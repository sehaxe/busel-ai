"""
🦩 BUSEL LOCAL CHAT v1.8 - FULL PROMPT INFERENCE
КРИТИЧЕСКИЙ ФИКС: Используем ВЕСЬ промпт без обрезки до кратного stride.
Патчер может работать с любой длиной входа благодаря левому паддингу.
"""
import os
import sys
import argparse
import yaml
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if os.path.basename(current_dir) in ["tools", "tests", "scripts"]:
    project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from model.patching import StridedFastBLTPatcher
from model.backbone import buselModel
from multimodal.special_tokens import vocab_size as _vocab_size


class InferenceConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 259)
        self.d_model = kwargs.get("d_model", 384)
        self.n_layers = kwargs.get("n_layers", 8)
        self.n_heads = kwargs.get("n_heads", 6)
        self.expert_hidden = kwargs.get("expert_hidden", 768)
        self.num_experts = kwargs.get("num_experts", 4)
        self.top_k = kwargs.get("top_k", 2)


def resolve_config(checkpoint_path: str, profile_override: str = None):
    cfg_dict = None
    source = None
    ckpt_profile = None
    config_path = os.path.join(project_root, "configs", "default.yaml")

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"🔍 Inspecting checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "config" in ckpt and ckpt["config"]:
            cfg_dict = ckpt["config"]
            source = "checkpoint['config']"
            ckpt_profile = cfg_dict.get("profile", ckpt.get("profile"))
        else:
            ckpt_profile = ckpt.get("profile")
            source = f"configs/default.yaml (profile: {ckpt_profile})"

    effective_profile = profile_override or ckpt_profile

    if cfg_dict is None:
        if not effective_profile:
            effective_profile = "shpak"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
        if effective_profile not in full_cfg.get("profiles", {}):
            raise ValueError(f"Profile '{effective_profile}' not found in configs/default.yaml")
        model_cfg = full_cfg["profiles"][effective_profile]["model"]
        cfg_dict = dict(model_cfg)

    cfg = InferenceConfig(**cfg_dict)
    print(f"✅ Resolved config from {source}")
    print(f"   🧠 d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}")
    print(f"   🔀 MoE: {cfg.num_experts} experts (top-{cfg.top_k}), hidden={cfg.expert_hidden}")
    print(f"   📚 Vocab: {cfg.vocab_size} (byte-level)")
    return cfg, effective_profile


def _strip_compile_prefix(sd):
    """Strip torch.compile state_dict prefixes (_orig_mod., compiled_model., _dynamo.).

    Training wraps the model via `torch.compile(model, fullgraph=False)` which causes
    `model.state_dict()` to return keys prefixed with `_orig_mod.`. Inference builds a
    fresh un-compiled model, so without this strip the keys never match and
    `load_state_dict(strict=False)` silently ignores ALL weights — the model runs with
    random init and produces garbage output.
    """
    if not sd:
        return sd
    out = {}
    for k, v in sd.items():
        new_k = k
        for prefix in ("_orig_mod.", "compiled_model.", "_dynamo."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        out[new_k] = v
    return out


def find_latest_checkpoint():
    ckpt_dir = os.path.join(project_root, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return None
    candidates = []
    for f in os.listdir(ckpt_dir):
        if not (f.startswith("busel_") and f.endswith(".pt")):
            continue
        full = os.path.join(ckpt_dir, f)
        size = os.path.getsize(full)
        if size < 2_000_000:  # <2MB is always corrupt for any busel profile
            print(f"⚠️  Skipping corrupt checkpoint: {f} ({size / 1024:.0f} KB < 2MB)")
            continue
        if size < 10_000_000:
            print(f"⚠️  Skipping undersized checkpoint: {f} ({size / 1024 / 1024:.1f} MB < 10MB threshold)")
            continue
        candidates.append((os.path.getmtime(full), full))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def detect_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_model(cfg, checkpoint_path, device):
    print(f"\n⚙️  Building model on {device.upper()}...")
    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = buselModel(cfg).to(device)

    target_dtype = torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32)
    for obj in [model, patcher]:
        for module in obj.modules():
            if "RMSNorm" in module.__class__.__name__:
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.data = module.weight.data.to(target_dtype)

    loaded = False
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"💾 Loading weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        try:
            model_sd = _strip_compile_prefix(ckpt["model_state_dict"])
            patcher_sd = _strip_compile_prefix(ckpt["patcher_state_dict"])

            m_missing, m_unexpected = model.load_state_dict(model_sd, strict=False)
            p_missing, p_unexpected = patcher.load_state_dict(patcher_sd, strict=False)

            total_ckpt_keys = len(model_sd) + len(patcher_sd)
            total_loaded = (len(model_sd) - len(m_unexpected)) + (len(patcher_sd) - len(p_unexpected))
            total_missing = len(m_missing) + len(p_missing)

            if total_missing > 0 or (m_unexpected or p_unexpected):
                print(f"   ⚠️  Partial load: {total_loaded}/{total_ckpt_keys} keys matched "
                      f"({len(m_missing)+len(p_missing)} missing, "
                      f"{len(m_unexpected)+len(p_unexpected)} unexpected)")
            else:
                print(f"   ✅ All {total_loaded} keys loaded exactly")

            loaded = total_loaded > 0
            if loaded:
                print(f"   📊 step={ckpt.get('step', '?')} profile={ckpt.get('profile', '?')}")
        except Exception as e:
            print(f"   ❌ Failed to load weights: {e}")
    else:
        print("⚠️  No checkpoint provided — running with random weights.")

    model.eval()
    patcher.eval()
    return model, patcher, loaded


def apply_sampling(logits, temperature, top_p, repetition_penalty, already_generated):
    # Mask special tokens [256:VOCAB_SIZE] — they are not raw bytes and would crash bytearray.append() downstream.
    logits[256:_vocab_size()] = -float("inf")

    for tok in already_generated:
        if tok < logits.shape[-1]:
            if logits[tok] > 0:
                logits[tok] /= repetition_penalty
            else:
                logits[tok] *= repetition_penalty

    # temperature==0 → greedy argmax restricted to byte range [0:256] (specials already masked above).
    if temperature < 1e-6:
        return int(logits[:256].argmax().item())

    logits = logits / (temperature + 1e-8)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float("inf")

    probs = torch.softmax(logits, dim=-1)

    # Guard against `invalid multinomial distribution` if top_p + temperature collapse every valid token.
    probs_sum = probs.sum().item()
    if not (probs_sum > 0) or not torch.isfinite(probs).any():
        return int(logits[:256].argmax().item())

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_stream(model, patcher, prompt, device,
                    max_new_tokens=200, temperature=0.7,
                    top_p=0.9, repetition_penalty=1.15,
                    stop_on_newline=False):
    """
    🎯 ИСПРАВЛЕННЫЙ ИНФЕРЕНС:
    - Используем ВЕСЬ промпт без обрезки
    - Патчер работает с любой длиной благодаря левому паддингу
    """
    prompt_bytes = list(prompt.encode("utf-8"))
    vocab_size = _vocab_size()
    
    generated_bytes = []
    already_generated = set(prompt_bytes)
    undecoded_tail = bytearray()
    
    autocast_dtype = torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32)
    autocast_enabled = device in ("cuda", "mps")
    
    for _ in range(max_new_tokens):
        # 🎯 КРИТИЧЕСКИЙ ФИКС: Используем ВЕСЬ промпт, не обрезаем!
        input_ids = torch.tensor([prompt_bytes], dtype=torch.int32, device=device)
        
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
            patches = patcher(input_ids)
            (logits_t1, _, _, _), _ = model(patches, None)
        
        next_token_logits = logits_t1[0, -1, :].clone().float()
        next_token = apply_sampling(next_token_logits, temperature, top_p, repetition_penalty, already_generated)
        
        if next_token >= vocab_size:
            break
        if next_token >= 256:
            already_generated.add(next_token)
            prompt_bytes.append(next_token)
            continue
        if stop_on_newline and next_token == 10 and generated_bytes:
            break
        
        generated_bytes.append(next_token)
        already_generated.add(next_token)
        prompt_bytes.append(next_token)
        undecoded_tail.append(next_token)
        
        try:
            decoded = undecoded_tail.decode("utf-8")
            if decoded:
                yield decoded
                undecoded_tail = bytearray()
        except UnicodeDecodeError:
            pass
    
    if undecoded_tail:
        try:
            yield undecoded_tail.decode("utf-8", errors="replace")
        except Exception:
            pass


def print_banner(profile, device, loaded):
    status = "💾 trained weights" if loaded else "🎲 random weights"
    print()
    print("╔" + "═" * 66 + "╗")
    print("║" + "🦩  BUSEL LOCAL CHAT v1.8 (FULL PROMPT)".center(66) + "║")
    print("║" + f"Profile: {profile}  |  Device: {device.upper()}  |  {status}".center(66) + "║")
    print("╠" + "═" * 66 + "╣")
    print("║" + "  /exit — выход  /reset — контекст  /raw — raw mode".ljust(66) + "║")
    print("║" + "  /temp 0.7  /topp 0.9  /rep 1.15  /max 150".ljust(66) + "║")
    print("╚" + "═" * 66 + "╝")
    print()


def interactive_loop(model, patcher, device, profile, loaded):
    print_banner(profile, device, loaded)
    temperature, top_p, repetition_penalty, max_new_tokens = 0.7, 0.9, 1.15, 150
    raw_mode, context = False, ""

    while True:
        try:
            user_input = input("\033[1;36mYou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit", "/q"):
            print("👋 Bye!")
            break
        if user_input.lower() == "/reset":
            context = ""
            print("🧹 Контекст очищен.")
            continue
        if user_input.lower() == "/raw":
            raw_mode = not raw_mode
            print(f"🔀 Raw mode: {'ON' if raw_mode else 'OFF'}")
            continue
        if user_input.lower().startswith("/temp"):
            try:
                temperature = float(user_input.split()[1])
                print(f"🌡️  temperature = {temperature}")
            except:
                print("❌ usage: /temp 0.7")
            continue
        if user_input.lower().startswith("/topp"):
            try:
                top_p = float(user_input.split()[1])
                print(f"🎯 top_p = {top_p}")
            except:
                print("❌ usage: /topp 0.9")
            continue
        if user_input.lower().startswith("/rep"):
            try:
                repetition_penalty = float(user_input.split()[1])
                print(f"🔁 rep = {repetition_penalty}")
            except:
                print("❌ usage: /rep 1.15")
            continue
        if user_input.lower().startswith("/max"):
            try:
                max_new_tokens = int(user_input.split()[1])
                print(f"📏 max = {max_new_tokens}")
            except:
                print("❌ usage: /max 150")
            continue

        prompt = (context + user_input) if raw_mode else (context + f"User: {user_input}\nAssistant: ")

        print("\033[1;33mBusel>\033[0m ", end="", flush=True)
        generated_text = ""
        try:
            for chunk in generate_stream(
                model, patcher, prompt, device,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, repetition_penalty=repetition_penalty,
                stop_on_newline=not raw_mode,
            ):
                print(chunk, end="", flush=True)
                generated_text += chunk
        except KeyboardInterrupt:
            print(" [прервано]")
        print()
        context = (prompt + generated_text) if raw_mode else (prompt + generated_text + "\n")


def main():
    parser = argparse.ArgumentParser(description="busel Local Interactive Chat")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--profile", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default=None)
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(project_root, ckpt)
    if ckpt is None:
        ckpt = find_latest_checkpoint()
        if ckpt:
            print(f"📂 Auto-selected latest checkpoint: {ckpt}")
        else:
            print("⚠️  No checkpoint found. Will use random weights.")

    cfg, profile = resolve_config(ckpt, args.profile)
    device = args.device or detect_device()
    print(f"🖥️  Device: {device.upper()}")
    model, patcher, loaded = load_model(cfg, ckpt, device)

    try:
        interactive_loop(model, patcher, device, profile, loaded)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()