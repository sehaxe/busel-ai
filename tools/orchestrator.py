"""
⚙️ busel ORCHESTRATOR v6.0 (PATH FIX)
Содержит команды запуска обучения, автопилота, профайлера и API-сервера.
"""

import os
import sys
import subprocess
import typer

DATA_DIR = "data_train"


def load_env(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")


def print_tui_header():
    typer.echo(typer.style("╔═══════════════════════════════════════════════════════════════════════════╗", fg=typer.colors.MAGENTA, bold=True))
    typer.echo(typer.style("║                            busel OMNI-LLM v4.0                            ║", fg=typer.colors.CYAN, bold=True))
    typer.echo(typer.style("║                 Sovereign 1-bit Any-to-Text AI Framework                  ║", fg=typer.colors.CYAN, bold=True))
    typer.echo(typer.style("╚═══════════════════════════════════════════════════════════════════════════╝", fg=typer.colors.MAGENTA, bold=True))


def autopilot(
    profile_name: str = typer.Option("shpak", "--profile", "-p", help="Profile name: shpak or zubr")
):
    print_tui_header()
    load_env()

    want_monitoring = typer.confirm("📊 Do you want to enable local logging & TensorBoard monitoring?", default=True)
    if want_monitoring:
        typer.echo(typer.style("📈 Monitoring activated. Run 'tensorboard --logdir=checkpoints' to view logs.\n", fg=typer.colors.GREEN))

    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        typer.echo(typer.style("📁 Directory 'data_train' is empty. Starting automatic download...", fg=typer.colors.YELLOW, bold=True))
        from tools.data_manager import _download_text, _download_sft, _download_vision
        _download_text(80000, "smollm")
        _download_sft(5000, "smoltalk")
        _download_vision(500, "HuggingFaceM4/COCO")
    else:
        typer.echo(typer.style("📁 Training data found. Skipping download.", fg=typer.colors.GREEN))

    typer.echo(typer.style("\n📊 Launching hardware express-profiler for MPS/CUDA testing...", fg=typer.colors.CYAN, bold=True))
    result = subprocess.run([sys.executable, "tests/profiler_run.py"])
    if result.returncode != 0:
        typer.echo(typer.style("❌ Hardware test failed! Please check your GPU/accelerator.", fg=typer.colors.RED, bold=True))
        raise typer.Exit(code=1)
        
    typer.echo("=" * 80)
    
    typer.echo(typer.style(f"🔥 AUTOPILOT: Launching main training loop [{profile_name.upper()}]...", fg=typer.colors.GREEN, bold=True))
    subprocess.run([sys.executable, "train.py", "--profile", profile_name])


def train(
    profile_name: str = typer.Option("shpak", "--profile", "-p", help="Profile: shpak or zubr"),
    resume: str = typer.Option(None, "--resume", "-r", help="Path to checkpoint for resuming")
):
    args = [sys.executable, "train.py", "--profile", profile_name]
    if resume:
        args.extend(["--resume", resume])
    subprocess.run(args)


def profile():
    subprocess.run([sys.executable, "tests/profiler_run.py"])


def serve(
    host: str = typer.Option("127.0.0.1", help="Server host"),
    port: int = typer.Option(8000, help="Server port")
):
    import uvicorn  # Lazy import
    typer.echo(typer.style(f"🔥 Starting API server on http://{host}:{port}", fg=typer.colors.MAGENTA, bold=True))
    uvicorn.run("services.inference_api:app", host=host, port=port, reload=False)