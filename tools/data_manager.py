"""
⚙️ BYSEL OMNIVORE DATA_MANAGER v7.0 (SHPAK PRESETS ENGINE)
Содержит автоматический расчет Chinchilla-оптимальности и загрузку по пресету Shpak.
"""

import os
import json
import base64
import shutil
import urllib.request
import typer

DATA_DIR = "data_train"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
JSONL_PATH = os.path.join(DATA_DIR, "dataset.jsonl")

# Реестр умных пресетов на базе Generalized Chinchilla Scaling Laws (80 байт на параметр)
PRESETS = {
    "shpak": {
        "text_limit": 768000,   # ~3.84B байт-токенов (истинная Chinchilla-оптимальность для 48M параметров)
        "sft_limit": 8000,      # 8K высококачественных диалогов Smoltalk для выравнивания чат-бота
        "vision_limit": 1000    # 1000 изображений для мультимодальных тестов
    }
}


def ensure_directories():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def _download_vision(limit: int, dataset_name: str):
    from datasets import load_dataset  # Lazy import
    ensure_directories()
    typer.echo(typer.style(f"📥 Connecting to HF and streaming '{dataset_name}'...", fg=typer.colors.CYAN))
    
    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        typer.echo(typer.style(f"❌ Failed to load dataset: {e}", fg=typer.colors.RED, bold=True))
        return
        
    count = 0
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        for item in dataset:
            if count >= limit:
                break
            try:
                img = item["image"]
                caption = item["sentences"]["raw"][0].strip() if "sentences" in item else item.get("caption", "").strip()
                if not caption:
                    continue
                    
                img_filename = f"images/coco_{count}.jpg"
                img_path = os.path.join(DATA_DIR, img_filename)
                img.save(img_path)
                
                line = {"image": img_filename, "text": caption}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
                if count % 100 == 0:
                    typer.echo(f"   Downloaded: {count}/{limit} images...")
            except Exception:
                continue
    typer.echo(typer.style(f"✅ Successfully saved {count} samples to '{JSONL_PATH}'", fg=typer.colors.GREEN))


def _download_text(limit: int, source: str):
    from datasets import load_dataset  # Lazy import
    ensure_directories()
    
    source_clean = source.lower().strip()
    
    if source_clean == "tinystories":
        dataset_name, split_name, name_param, text_key = "roneneldan/TinyStories", "train", None, "text"
        output_file = os.path.join(DATA_DIR, "pretrain_tinystories.txt")
    elif source_clean == "fineweb":
        dataset_name, split_name, name_param, text_key = "HuggingFaceFW/fineweb-edu", "train", "sample-10BT", "text"
        output_file = os.path.join(DATA_DIR, "pretrain_fineweb.txt")
    elif source_clean in ["smollm", "cosmopedia"]:
        dataset_name, split_name, name_param, text_key = "HuggingFaceTB/smollm-corpus", "train", "cosmopedia-v2", "text"
        output_file = os.path.join(DATA_DIR, "pretrain_cosmopedia.txt")
    else:
        typer.echo(typer.style("❌ Unsupported source! Choose 'smollm', 'fineweb', or 'tinystories'.", fg=typer.colors.RED))
        return

    typer.echo(typer.style(f"📥 Streaming '{dataset_name}' from Hugging Face...", fg=typer.colors.CYAN))
    try:
        if name_param:
            dataset = load_dataset(dataset_name, name=name_param, split=split_name, streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split_name, streaming=True)
    except Exception as e:
        typer.echo(typer.style(f"❌ Load error: {e}", fg=typer.colors.RED))
        return

    count = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            if count >= limit:
                break
            try:
                text_content = item[text_key].strip()
                if not text_content: 
                    continue
                f.write(text_content + "\n\n")
                count += 1
                if count % 2000 == 0:
                    typer.echo(f"   Saved: {count}/{limit} texts...")
            except Exception:
                continue
    typer.echo(typer.style(f"✅ Successfully saved {count} texts to '{output_file}'", fg=typer.colors.GREEN))


def _download_sft(limit: int, source: str):
    from datasets import load_dataset  # Lazy import
    ensure_directories()
    
    source_clean = source.lower().strip()
    
    if source_clean == "alpaca":
        dataset_name = "tatsu-lab/alpaca"
        output_file = os.path.join(DATA_DIR, "sft_alpaca.jsonl")
    elif source_clean == "smoltalk":
        dataset_name = "HuggingFaceTB/smoltalk"
        output_file = os.path.join(DATA_DIR, "sft_smoltalk.jsonl")
    else:
        typer.echo(typer.style("❌ Unsupported SFT source! Choose 'smoltalk' or 'alpaca'.", fg=typer.colors.RED))
        return
    
    typer.echo(typer.style(f"📥 Streaming English instruction dataset '{dataset_name}'...", fg=typer.colors.CYAN))
    try:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        typer.echo(typer.style(f"❌ Failed to load SFT dataset: {e}", fg=typer.colors.RED))
        return

    count = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for item in dataset:
            if count >= limit:
                break
            try:
                if source_clean == "alpaca":
                    instruction = item.get("instruction", "").strip()
                    inp = item.get("input", "").strip()
                    output = item.get("output", "").strip()
                    if not instruction or not output: 
                        continue
                    
                    full_prompt = f"User: {instruction}"
                    if inp: 
                        full_prompt += f"\nContext: {inp}"
                    full_prompt += f"\nAssistant: {output}"
                else:  # smoltalk
                    messages = item.get("messages", [])
                    if not messages: 
                        continue
                    full_prompt = ""
                    for msg in messages:
                        role = msg.get("role", "user").capitalize()
                        content = msg.get("content", "").strip()
                        full_prompt += f"{role}: {content}\n"
                
                line = {"text": full_prompt.strip()}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
                if count % 1000 == 0:
                    typer.echo(f"   Converted: {count}/{limit} instructions...")
            except Exception:
                continue
    typer.echo(typer.style(f"✅ SFT dataset ({count} instructions) successfully saved to '{output_file}'", fg=typer.colors.GREEN))


def download_all(
    text_limit: int = typer.Option(5000, "--text-limit", "-t", help="Limit for pretrain text"),
    sft_limit: int = typer.Option(3000, "--sft-limit", "-s", help="Limit for SFT instructions"),
    vision_limit: int = typer.Option(1000, "--vision-limit", "-v", help="Limit for COCO images"),
    preset: str = typer.Option(None, "--preset", "-p", help="Automatic profile preset: 'shpak'")
):
    if preset:
        preset_clean = preset.lower().strip()
        if preset_clean in PRESETS:
            text_limit = PRESETS[preset_clean]["text_limit"]
            sft_limit = PRESETS[preset_clean]["sft_limit"]
            vision_limit = PRESETS[preset_clean]["vision_limit"]
            typer.echo(typer.style(f"🦁 PRESET DETECTED: {preset_clean}", fg=typer.colors.GREEN, bold=True))
            typer.echo(typer.style(f"📊 Generalized Chinchilla config: Text={text_limit}, SFT={sft_limit}, Vision={vision_limit}", fg=typer.colors.CYAN))
        else:
            typer.echo(typer.style(f"⚠️ Unknown preset '{preset}'! Using manual limits.", fg=typer.colors.YELLOW))

    typer.echo(typer.style("\n📥 STARTING BULK DATASET DOWNLOAD...", fg=typer.colors.CYAN, bold=True))
    _download_text(text_limit, "smollm")
    _download_sft(sft_limit, "smoltalk")
    _download_vision(vision_limit, "HuggingFaceM4/COCO")


def download_vision(
    limit: int = typer.Option(1000, "--limit", "-l", help="Number of images to download"),
    dataset_name: str = typer.Option("HuggingFaceM4/COCO", "--dataset", "-d", help="Hugging Face dataset name")
):
    _download_vision(limit, dataset_name)


def download_text(
    limit: int = typer.Option(5000, "--limit", "-l", help="Number of pretrain texts to download"),
    source: str = typer.Option("smollm", "--source", "-s", help="Source: 'smollm', 'fineweb', 'tinystories'"),
    preset: str = typer.Option(None, "--preset", "-p", help="Automatic profile preset: 'shpak'")
):
    if preset:
        preset_clean = preset.lower().strip()
        if preset_clean in PRESETS:
            limit = PRESETS[preset_clean]["text_limit"]
            typer.echo(typer.style(f"🦁 PRESET DETECTED: {preset_clean}", fg=typer.colors.GREEN, bold=True))
            typer.echo(typer.style(f"📊 Generalized Chinchilla pretrain volume (~80 bytes/param): Downloading {limit} docs...", fg=typer.colors.CYAN))
        else:
            typer.echo(typer.style(f"⚠️ Unknown preset '{preset}'! Using manual limits.", fg=typer.colors.YELLOW))

    _download_text(limit, source)


def download_sft(
    limit: int = typer.Option(3000, "--limit", "-l", help="Number of SFT instructions to download"),
    source: str = typer.Option("smoltalk", "--source", "-s", help="SFT Source: 'smoltalk' or 'alpaca'"),
    preset: str = typer.Option(None, "--preset", "-p", help="Automatic profile preset: 'shpak'")
):
    if preset:
        preset_clean = preset.lower().strip()
        if preset_clean in PRESETS:
            limit = PRESETS[preset_clean]["sft_limit"]
            typer.echo(typer.style(f"🦁 PRESET DETECTED: {preset_clean}", fg=typer.colors.GREEN, bold=True))
            typer.echo(typer.style(f"📊 SFT aligned volume: Downloading {limit} instructions...", fg=typer.colors.CYAN))
        else:
            typer.echo(typer.style(f"⚠️ Unknown preset '{preset}'! Using manual limits.", fg=typer.colors.YELLOW))

    _download_sft(limit, source)


def label_vision(
    source_dir: str = typer.Option("my_photos", "--dir", "-s", help="Directory with raw images"),
    model: str = typer.Option("moondream", "--model", "-m", help="Local vision model in Ollama")
):
    ensure_directories()
    if not os.path.exists(source_dir):
        os.makedirs(source_dir, exist_ok=True)
        typer.echo(typer.style(f"📁 Created empty directory '{source_dir}'. Place your images there.", fg=typer.colors.YELLOW))
        raise typer.Exit()
        
    supported_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    images = [f for f in os.listdir(source_dir) if f.lower().endswith(supported_exts)]
    
    if not images:
        typer.echo(typer.style(f"📂 No images found in '{source_dir}'.", fg=typer.colors.YELLOW))
        raise typer.Exit()
        
    typer.echo(typer.style(f"🤖 Found {len(images)} images. Querying Ollama...", fg=typer.colors.CYAN))
    
    count = 0
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        for filename in images:
            src_path = os.path.join(source_dir, filename)
            with open(src_path, "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                
            payload = {"model": model, "prompt": "Describe this image in detail. Keep it to 1-2 descriptive sentences.", "images": [img_b64], "stream": False}
            req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    caption = res["response"].strip()
            except Exception as e:
                typer.echo(typer.style(f"❌ Ollama request failed for {filename}: {e}", fg=typer.colors.RED))
                continue
                
            target_filename = f"images/my_{filename}"
            shutil.copy2(src_path, os.path.join(DATA_DIR, target_filename))
            f.write(json.dumps({"image": target_filename, "text": caption}, ensure_ascii=False) + "\n")
            typer.echo(f"   Labeled '{filename}' -> \"{caption}\"")
            count += 1
    typer.echo(typer.style(f"✅ Done! {count} images labeled in '{JSONL_PATH}'", fg=typer.colors.GREEN))