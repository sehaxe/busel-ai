"""
🧙 BYSEL SETUP WIZARD - Interactive .env configuration
"""
import os
import re
import typer
from pathlib import Path

ENV_FILE = Path(".env")

def load_env() -> dict:
    """Загружает существующие переменные из .env"""
    env_vars = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars

def save_env(env_vars: dict):
    """Сохраняет переменные в .env"""
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write("# ╔═══════════════════════════════════════════════════════════════╗\n")
        f.write("# ║  BYSEL (Бусел) - Sovereign Omni-LLM Configuration            ║\n")
        f.write("# ╚═══════════════════════════════════════════════════════════════╝\n\n")
        
        f.write("# 🤖 Telegram Bot Token (from @BotFather)\n")
        f.write(f'TELEGRAM_BOT_TOKEN="{env_vars.get("TELEGRAM_BOT_TOKEN", "")}"\n\n')
        
        f.write("# 👑 Admin User IDs (comma-separated Telegram user IDs)\n")
        f.write("# Get your ID from @userinfobot or @getmyid_bot\n")
        f.write(f'TELEGRAM_ADMIN_IDS="{env_vars.get("TELEGRAM_ADMIN_IDS", "")}"\n\n')
        
        f.write("# 🌐 Inference API URL\n")
        f.write(f'INFERENCE_API_URL="{env_vars.get("INFERENCE_API_URL", "http://127.0.0.1:8000")}"\n\n')
        
        f.write("# ⚙️ Training defaults\n")
        f.write(f'DEFAULT_PROFILE="{env_vars.get("DEFAULT_PROFILE", "shpak")}"\n')

def validate_bot_token(token: str) -> bool:
    """Проверяет формат токена бота."""
    return bool(re.match(r'^\d+:[A-Za-z0-9_-]{35,}$', token))

def validate_admin_ids(ids_str: str) -> bool:
    """Проверяет формат списка ID админов."""
    if not ids_str.strip():
        return False
    parts = [p.strip() for p in ids_str.split(",")]
    return all(p.isdigit() for p in parts if p)

def run_setup_wizard(force: bool = False):
    """Интерактивная настройка .env файла."""
    env_vars = load_env()
    
    # Проверяем, нужна ли настройка
    has_token = env_vars.get("TELEGRAM_BOT_TOKEN", "").strip()
    has_admins = env_vars.get("TELEGRAM_ADMIN_IDS", "").strip()
    
    if has_token and has_admins and not force:
        return env_vars
    
    typer.echo(typer.style("\n╔═══════════════════════════════════════════════════════════════╗", fg=typer.colors.MAGENTA, bold=True))
    typer.echo(typer.style("║  🧙 BYSEL SETUP WIZARD - First Time Configuration             ║", fg=typer.colors.CYAN, bold=True))
    typer.echo(typer.style("╚═══════════════════════════════════════════════════════════════╝", fg=typer.colors.MAGENTA, bold=True))
    typer.echo()
    
    # 1. Bot Token
    typer.echo(typer.style("📱 Шаг 1: Telegram Bot Token", fg=typer.colors.GREEN, bold=True))
    typer.echo("   Получите токен у @BotFather в Telegram:")
    typer.echo("   1. Напишите @BotFather")
    typer.echo("   2. Отправьте /newbot")
    typer.echo("   3. Следуйте инструкциям")
    typer.echo("   4. Скопируйте токен вида: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz")
    typer.echo()
    
    while True:
        if has_token:
            masked = has_token[:10] + "..." + has_token[-5:]
            token = typer.prompt(
                f"   Введите токен бота (Enter чтобы оставить {masked})",
                default="",
                show_default=False
            ).strip()
            if not token:
                token = has_token
        else:
            token = typer.prompt(
                "   Введите токен бота",
                default="",
                show_default=False
            ).strip()
        
        if validate_bot_token(token):
            env_vars["TELEGRAM_BOT_TOKEN"] = token
            typer.echo(typer.style("   ✅ Токен принят!", fg=typer.colors.GREEN))
            break
        else:
            typer.echo(typer.style("   ❌ Неверный формат токена. Попробуйте еще раз.", fg=typer.colors.RED))
    
    typer.echo()
    
    # 2. Admin IDs
    typer.echo(typer.style("👑 Шаг 2: Admin User IDs", fg=typer.colors.GREEN, bold=True))
    typer.echo("   Узнайте свой Telegram ID:")
    typer.echo("   • Напишите @userinfobot или @getmyid_bot")
    typer.echo("   • Скопируйте числовой ID (например: 123456789)")
    typer.echo("   • Можно указать несколько ID через запятую")
    typer.echo()
    
    while True:
        if has_admins:
            admin_ids = typer.prompt(
                f"   Введите ID админов (Enter чтобы оставить {has_admins})",
                default="",
                show_default=False
            ).strip()
            if not admin_ids:
                admin_ids = has_admins
        else:
            admin_ids = typer.prompt(
                "   Введите ID админов",
                default="",
                show_default=False
            ).strip()
        
        if validate_admin_ids(admin_ids):
            env_vars["TELEGRAM_ADMIN_IDS"] = admin_ids
            admin_list = [int(x.strip()) for x in admin_ids.split(",")]
            typer.echo(typer.style(f"   ✅ Админы: {admin_list}", fg=typer.colors.GREEN))
            break
        else:
            typer.echo(typer.style("   ❌ Неверный формат. Введите числа через запятую.", fg=typer.colors.RED))
    
    typer.echo()
    
    # 3. Optional settings
    typer.echo(typer.style("⚙️  Шаг 3: Дополнительные настройки (опционально)", fg=typer.colors.YELLOW, bold=True))
    
    api_url = typer.prompt(
        "   URL инференс API",
        default=env_vars.get("INFERENCE_API_URL", "http://127.0.0.1:8000"),
        show_default=True
    )
    env_vars["INFERENCE_API_URL"] = api_url
    
    profile = typer.prompt(
        "   Профиль обучения по умолчанию",
        default=env_vars.get("DEFAULT_PROFILE", "shpak"),
        show_default=True
    )
    env_vars["DEFAULT_PROFILE"] = profile
    
    # Сохраняем
    save_env(env_vars)
    
    typer.echo()
    typer.echo(typer.style("╔═══════════════════════════════════════════════════════════════╗", fg=typer.colors.GREEN, bold=True))
    typer.echo(typer.style("║  ✅ Конфигурация сохранена в .env                             ║", fg=typer.colors.GREEN, bold=True))
    typer.echo(typer.style("╚═══════════════════════════════════════════════════════════════╝", fg=typer.colors.GREEN, bold=True))
    typer.echo()
    
    return env_vars

def get_admin_ids() -> list:
    """Возвращает список ID админов из окружения."""
    admin_str = os.environ.get("TELEGRAM_ADMIN_IDS", "")
    if not admin_str:
        return []
    return [int(x.strip()) for x in admin_str.split(",") if x.strip().isdigit()]

def is_admin(user_id: int) -> bool:
    """Проверяет, является ли пользователь админом."""
    return user_id in get_admin_ids()