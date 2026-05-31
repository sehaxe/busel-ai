import typer
import os
import uvicorn
from tests.profiler_run import run_profile_test

app = typer.Typer(help="Bysel CLI Engine - Sovereign 1-bit Omni-LLM")

@app.command()
def train(
    mode: str = typer.Option(..., "--mode", "-m", help="Стадия: pretrain, sft или dpo"),
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Конфиг"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Имя датасета"),
    autopilot: bool = typer.Option(True, help="Включить автопилот")
):
    typer.echo(typer.style(f"🚀 Запуск обучения [{mode.upper()}] для bysel...", fg=typer.colors.GREEN, bold=True))

@app.command()
def profile():
    typer.echo(typer.style("📊 Запуск профилировщика...", fg=typer.colors.CYAN, bold=True))
    run_profile_test()

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Хост"),
    port: int = typer.Option(8000, help="Порт")
):
    typer.echo(typer.style(f"🔥 Запуск сервера на http://{host}:{port}", fg=typer.colors.MAGENTA, bold=True))
    uvicorn.run("services.inference_api:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    app()
