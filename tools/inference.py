"""
🦩 BUSEL LOCAL CHAT — FULL PROMPT INFERENCE
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
from tools.tool_executor import (
    default_tool_registry,
    parse_tool_calls,
    execute_tool_calls,
    format_tool_results,
    format_tool_call_for_user,
    interactive_confirm,
    denied_result,
    strip_ansi,
    MAX_TOOL_CALLS_PER_TURN,
    MAX_TOOL_OUTPUT_BYTES,
)


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
    return "cpu"


def load_model(cfg, checkpoint_path, device, compile_inference: bool = False):
    print(f"\n⚙️  Building model on {device.upper()}..." + ("  (compile ON)" if compile_inference else ""))
    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = buselModel(cfg).to(device)

    target_dtype = torch.bfloat16 if device == "cuda" else torch.float32
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
            from model.checkpoint import load_state_dict_safely
            model_sd = ckpt["model_state_dict"]
            patcher_sd = ckpt["patcher_state_dict"]
            m_missing, m_unexpected = load_state_dict_safely(model, model_sd, strict=False)
            p_missing, p_unexpected = load_state_dict_safely(patcher, patcher_sd, strict=False)

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

    if compile_inference:
        model = torch.compile(model, fullgraph=False, dynamic=True, mode="default")
        patcher = torch.compile(patcher, fullgraph=False, dynamic=True, mode="default")

    return model, patcher, loaded


def apply_sampling(logits, temperature, top_p, repetition_penalty, already_generated):
    # Mask special tokens [256:VOCAB_SIZE] — they are not raw bytes and would crash bytearray.append() downstream.
    logits[256:_vocab_size()] = -float("inf")

    if already_generated:
        indices_list = [t for t in already_generated if t < logits.shape[-1]]
        if indices_list:
            indices = torch.tensor(indices_list, dtype=torch.long, device=logits.device)
            scores = logits[indices]
            scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
            logits[indices] = scores

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
    
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    autocast_enabled = device == "cuda"
    
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
    print("║" + "🦩  BUSEL LOCAL CHAT".center(66) + "║")
    print("║" + f"Profile: {profile}  |  Device: {device.upper()}  |  {status}".center(66) + "║")
    print("╠" + "═" * 66 + "╣")
    print("║" + "  /exit /reset /raw  /tools  /auto on|off".ljust(66) + "║")
    print("║" + "  /temp 0.7  /topp 0.9  /rep 1.15  /max 150".ljust(66) + "║")
    print("╚" + "═" * 66 + "╝")
    print()


def parse_repl_command_value(parts, value_type):
    """Parse a `/<cmd> <value>` REPL command. Returns (new_value, None) on success, (None, usage_msg) on failure.

    The narrow except is critical: a bare `except:` would swallow KeyboardInterrupt
    and SystemExit, hanging the REPL or preventing shutdown. The only legitimate
    failures here are IndexError (no arg) and ValueError (bad value).
    """
    try:
        return value_type(parts[1]), None
    except (ValueError, IndexError):
        return None, f"❌ usage: /{parts[0][1:]} <value>"


_REPL_PARAMS = [
    ("/temp", "temperature", float, "🌡️  temperature"),
    ("/topp", "top_p", float, "🎯 top_p"),
    ("/rep", "repetition_penalty", float, "🔁 rep"),
    ("/max", "max_new_tokens", int, "📏 max"),
]


def interactive_loop(model, patcher, device, profile, loaded):
    print_banner(profile, device, loaded)
    params = {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.15, "max_new_tokens": 150}
    raw_mode, context = False, ""
    auto_approve = False
    tool_registry = default_tool_registry()

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
        if user_input.lower() == "/tools":
            names = tool_registry.names()
            print(f"🔧 Available tools ({len(names)}): {', '.join(names)}")
            print(f"   Max tool calls per turn: {MAX_TOOL_CALLS_PER_TURN}")
            print(f"   Auto-approve: {'ON' if auto_approve else 'OFF'}")
            continue
        if user_input.lower().startswith("/auto"):
            parts = user_input.split()
            if len(parts) == 1:
                auto_approve = not auto_approve
            elif parts[1].lower() in ("on", "1", "yes", "y", "true"):
                auto_approve = True
            elif parts[1].lower() in ("off", "0", "no", "n", "false"):
                auto_approve = False
            else:
                print("❌ usage: /auto [on|off]")
                continue
            print(f"🤖 Auto-approve tool calls: {'ON' if auto_approve else 'OFF'}")
            continue
        for cmd, var_name, value_type, label in _REPL_PARAMS:
            if user_input.lower().startswith(cmd):
                new_value, err = parse_repl_command_value(user_input.split(), value_type)
                if err is not None:
                    print(err)
                else:
                    params[var_name] = new_value
                    print(f"{label} = {new_value}")
                continue

        prompt = (context + user_input) if raw_mode else (context + f"User: {user_input}\nAssistant: ")

        # Multi-pass: model can emit tool calls, get results, continue generating.
        # Each pass is one `generate_stream` call. Up to MAX_TOOL_CALLS_PER_TURN
        # tool-call rounds per user turn (loop guard).
        for pass_idx in range(MAX_TOOL_CALLS_PER_TURN + 1):
            print("\033[1;33mBusel>\033[0m ", end="", flush=True)
            generated_text = ""
            try:
                for chunk in generate_stream(
                    model, patcher, prompt, device,
                    max_new_tokens=params["max_new_tokens"], temperature=params["temperature"],
                    top_p=params["top_p"], repetition_penalty=params["repetition_penalty"],
                    # Tool-call envelopes span multiple lines; the model must
                    # not be cut off at the first newline mid-envelope.
                    stop_on_newline=False,
                ):
                    print(chunk, end="", flush=True)
                    generated_text += chunk
            except KeyboardInterrupt:
                print(" [прервано]")
                break
            print()

            context = (prompt + generated_text) if raw_mode else (prompt + generated_text + "\n")

            calls = parse_tool_calls(generated_text)
            if not calls:
                # No tool calls → this is the final assistant turn for this user message.
                break

            if pass_idx >= MAX_TOOL_CALLS_PER_TURN:
                print(
                    f"⚠️  Max tool calls per turn ({MAX_TOOL_CALLS_PER_TURN}) reached. "
                    f"Stopping tool loop — returning to you."
                )
                break

            # Per-call confirmation: default N = deny. The user can flip the
            # session default to "always approve" via /auto on.
            results = []
            for call in calls:
                confirmed = interactive_confirm(call, auto_approve=auto_approve)
                if confirmed:
                    raw_output = tool_registry.call(call["name"], call["params"])
                    output = strip_ansi(str(raw_output))[:MAX_TOOL_OUTPUT_BYTES]
                    print(f"   ✅ {format_tool_call_for_user(call)}")
                    print(f"      → {output[:300]}{'...' if len(output) > 300 else ''}")
                    results.append({**call, "output": output})
                else:
                    print(f"   🚫 Denied: {format_tool_call_for_user(call)}")
                    results.append(denied_result(call))

            # Next pass sees `<function_results>...</function_results>` appended
            # to the conversation and continues with the final response.
            results_block = format_tool_results(results)
            context = context + results_block + "\n"
            prompt = context


def main():
    parser = argparse.ArgumentParser(description="busel Local Interactive Chat")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--profile", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default=None)
    parser.add_argument("--compile-inference", action="store_true",
                        help="Wrap model+patcher in torch.compile (dynamic=True, mode=default) for ~1.2x gen loop speedup")
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
    model, patcher, loaded = load_model(cfg, ckpt, device, compile_inference=args.compile_inference)

    try:
        interactive_loop(model, patcher, device, profile, loaded)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()