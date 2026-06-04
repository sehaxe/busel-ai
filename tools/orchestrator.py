"""
⚙️ busel ORCHESTRATOR v6.1 — Multi-Stage Pipeline
Содержит команды запуска обучения, автопилота, профайлера, и pipeline.
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
    typer.echo(typer.style("║                            busel OMNI-LLM v6.1                            ║", fg=typer.colors.CYAN, bold=True))
    typer.echo(typer.style("║                 Sovereign 1-bit Any-to-Text AI Framework                  ║", fg=typer.colors.CYAN, bold=True))
    typer.echo(typer.style("╚═══════════════════════════════════════════════════════════════════════════╝", fg=typer.colors.MAGENTA, bold=True))


def _build_shim_yaml(profile: str, resume: str, max_steps, warmup_steps) -> str:
    """Build a temp pipeline YAML (pretrain-only + overrides); return the temp dir path."""
    import tempfile
    import yaml as _yaml
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "pipelines", "pretrain-only.yaml")
    with open(src) as f:
        cfg = _yaml.safe_load(f)
    stage = cfg["stages"][0]
    stage.setdefault("params", {})
    if profile:
        stage["params"]["profile_name"] = profile
    if max_steps is not None:
        stage["params"]["max_steps"] = max_steps
    if warmup_steps is not None:
        stage["params"]["warmup_steps"] = warmup_steps
    if resume:
        stage["resume"] = resume
    tmpdir = tempfile.mkdtemp(prefix="busel_shim_")
    tmp_yaml = os.path.join(tmpdir, "shim.yaml")
    with open(tmp_yaml, "w") as f:
        _yaml.dump(cfg, f)
    return tmpdir


def train_single_profile(args_list):
    """Translate legacy train.py CLI args into a pipeline run.

    Supported: --profile, --resume, --max-steps, --warmup-steps.
    Other flags (--no-compile, --compile-mode, --no-checkpointing, --seed) are dropped
    because the pipeline runner owns those knobs (see configs/pipelines/*.yaml).
    """
    import argparse
    import shutil
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--profile", "-p", default="shpak")
    p.add_argument("--resume", "-r", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    args, _unknown = p.parse_known_args(args_list)

    tmpdir = _build_shim_yaml(args.profile, args.resume, args.max_steps, args.warmup_steps)
    try:
        pipeline(name="shim", start_stage=None, config_dir=tmpdir)
        return 0
    except SystemExit as e:
        return e.code or 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
    train_single_profile(["--profile", profile_name])


def train(
    profile_name: str = typer.Option("shpak", "--profile", "-p", help="Profile: shpak or zubr"),
    resume: str = typer.Option(None, "--resume", "-r", help="Path to checkpoint for resuming")
):
    args = ["--profile", profile_name]
    if resume:
        args.extend(["--resume", resume])
    train_single_profile(args)


def train_all(
    start_stage: str = typer.Option(None, "--start-stage", help="Resume from this stage name (e.g. 'sft', 'dpo')"),
):
    """🚀 ONE-CLICK FULL TRAINING: pretrain → SFT → DPO → eval.

    Runs the `full` pipeline (configs/pipelines/full.yaml). Requires that
    the 4 HF data presets are already downloaded — run
    `uv run cli.py download-data` first.
    """
    pipeline(name="full", start_stage=start_stage, config_dir="configs/pipelines")


def profile():
    subprocess.run([sys.executable, "tests/profiler_run.py"])


def pipeline(
    name: str = typer.Option(..., "--name", "-n", help="Pipeline name (configs/pipelines/<name>.yaml)"),
    start_stage: str = typer.Option(None, "--start-stage", help="Resume from this stage name"),
    config_dir: str = typer.Option("configs/pipelines", "--config-dir", help="Where to look for pipeline YAMLs"),
):
    """Run a multi-stage training pipeline.

    Loads configs/pipelines/<name>.yaml, instantiates each registered
    stage via training/stages, and runs setup → run → finalize in order.
    Per-stage checkpoints are saved automatically.
    """
    from training.stages import load_pipeline_yaml, get_stage
    from busel_logging import setup_logging, log_event
    from training.stages.base import StageState

    print_tui_header()
    load_env()
    setup_logging()

    yaml_path = os.path.join(config_dir, f"{name}.yaml")
    if not os.path.exists(yaml_path):
        typer.echo(typer.style(f"❌ Pipeline YAML not found: {yaml_path}", fg=typer.colors.RED, bold=True))
        typer.echo(typer.style(f"   Available pipelines in {config_dir}:", fg=typer.colors.YELLOW))
        if os.path.isdir(config_dir):
            for f in sorted(os.listdir(config_dir)):
                if f.endswith(".yaml"):
                    typer.echo(f"     - {f[:-5]}")
        raise typer.Exit(code=1)

    pipeline_cfg = load_pipeline_yaml(yaml_path)
    log_event("pipeline_start", pipeline=pipeline_cfg.name, num_stages=len(pipeline_cfg.stages))

    typer.echo(typer.style(f"🛸 Pipeline: {pipeline_cfg.name} ({len(pipeline_cfg.stages)} stages)", fg=typer.colors.CYAN, bold=True))
    for i, s in enumerate(pipeline_cfg.stages, 1):
        typer.echo(typer.style(f"   {i}. {s.name}  data={s.data_preset or '-'}  resume={s.resume or '-'}", fg=typer.colors.CYAN))

    import yaml as _yaml
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        _default_profiles = _yaml.safe_load(f).get("profiles", {})

    def _resolve_resume(stage_name: str, default_resume: str | None) -> str | None:
        if default_resume:
            return default_resume
        candidate = f"checkpoints/busel_{pipeline_cfg.name}_{stage_name}_FINAL.pt"
        return candidate if os.path.exists(candidate) else None

    state = StageState()
    skipping = bool(start_stage)
    running_resume: str | None = None

    for i, stage_spec in enumerate(pipeline_cfg.stages, 1):
        if skipping:
            if stage_spec.name == start_stage:
                skipping = False
            else:
                typer.echo(typer.style(f"⏭  Skipping stage {i}/{len(pipeline_cfg.stages)}: {stage_spec.name}", fg=typer.colors.YELLOW))
                continue

        typer.echo(typer.style(f"\n🚀 Stage {i}/{len(pipeline_cfg.stages)}: {stage_spec.name}", fg=typer.colors.GREEN, bold=True))
        log_event("stage_start", pipeline=pipeline_cfg.name, stage=stage_spec.name, index=i)

        stage_cls = get_stage(stage_spec.name)
        stage = stage_cls()

        merged_params = {**pipeline_cfg.global_params, **stage_spec.params}
        profile_name = merged_params.pop("profile_name", stage_spec.data_preset or "shpak")
        profile_dict = _default_profiles.get(profile_name)
        if profile_dict is None:
            raise ValueError(f"Profile {profile_name!r} not in configs/default.yaml")

        resume = _resolve_resume(stage_spec.name, stage_spec.resume)
        if running_resume and stage_spec.name != "pretrain" and not stage_spec.resume:
            resume = running_resume

        try:
            stage.setup(
                profile=profile_dict,
                profile_name=profile_name,
                resume=resume,
                stage_params=merged_params,
            )
        except Exception as e:
            typer.echo(typer.style(f"❌ Stage {stage_spec.name} setup() failed: {type(e).__name__}: {e}", fg=typer.colors.RED))
            log_event("stage_failed", stage=stage_spec.name, phase="setup", error=str(e))
            raise typer.Exit(code=1)

        try:
            state = stage.run(state)
        except SystemExit:
            raise
        except Exception as e:
            typer.echo(typer.style(f"❌ Stage {stage_spec.name} run() failed: {type(e).__name__}: {e}", fg=typer.colors.RED))
            log_event("stage_failed", stage=stage_spec.name, phase="run", error=str(e))
            raise typer.Exit(code=1)

        try:
            state = stage.finalize(state)
        except Exception as e:
            typer.echo(typer.style(f"❌ Stage {stage_spec.name} finalize() failed: {type(e).__name__}: {e}", fg=typer.colors.RED))
            log_event("stage_failed", stage=stage_spec.name, phase="finalize", error=str(e))
            raise typer.Exit(code=1)

        if state.last_checkpoint_path:
            running_resume = state.last_checkpoint_path

    log_event("pipeline_complete", pipeline=pipeline_cfg.name, total_stages=len(pipeline_cfg.stages))
    typer.echo(typer.style(f"\n🎉 Pipeline {pipeline_cfg.name} complete! {len(pipeline_cfg.stages)} stages succeeded.", fg=typer.colors.GREEN, bold=True))