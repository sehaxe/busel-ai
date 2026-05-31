"""
🤖 BYSEL SOVEREIGN BOT v2.0 - Full-featured Training Commander
"""
import os
import sys
import json
import time
import asyncio
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional

from aiogram import Bot, Dispatcher, types, Router, F
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
from aiogram.enums import ParseMode, ChatAction

# Пути
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram_bot.state_manager import (
    get_state, set_status, is_alive, get_latest_metrics,
    get_metrics_history, estimate_eta
)
from telegram_bot.setup_wizard import get_admin_ids, is_admin, run_setup_wizard

# ═══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════

API_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
INFERENCE_URL = os.environ.get("INFERENCE_API_URL", "http://127.0.0.1:8000")

# Rate limiting settings
PUBLIC_RATE_LIMIT = 5  # запросов в минуту
PUBLIC_RATE_WINDOW = 60  # секунд
MAX_GENERATION_LENGTH = 200  # для публичных пользователей
ADMIN_MAX_LENGTH = 1000  # для админов

# ═══════════════════════════════════════════════════════════════
# FSM STATES
# ═══════════════════════════════════════════════════════════════

class BotStates(StatesGroup):
    waiting_for_chat = State()
    waiting_for_profile = State()
    confirming_stop = State()

# ═══════════════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.requests = defaultdict(list)
    
    def check(self, user_id: int) -> tuple[bool, int]:
        """Возвращает (разрешено, секунд до сброса)."""
        now = time.time()
        # Очищаем старые записи
        self.requests[user_id] = [
            t for t in self.requests[user_id]
            if now - t < self.window
        ]
        
        if len(self.requests[user_id]) >= self.limit:
            oldest = self.requests[user_id][0]
            wait_time = int(self.window - (now - oldest))
            return False, wait_time
        
        self.requests[user_id].append(now)
        return True, 0

rate_limiter = RateLimiter(PUBLIC_RATE_LIMIT, PUBLIC_RATE_WINDOW)

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def format_duration(seconds: float) -> str:
    """Форматирует длительность в человекочитаемый вид."""
    if seconds < 60:
        return f"{seconds:.0f}с"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}мин"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}ч {mins}мин"
    else:
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{days}д {hours}ч"

def format_number(n: float) -> str:
    """Форматирует большие числа."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))

def get_status_emoji(status: str) -> str:
    """Возвращает emoji для статуса."""
    emojis = {
        "idle": "💤",
        "running": "🔥",
        "paused": "⏸️",
        "stopped": "🛑",
        "finished": "🎉",
        "error": "❌"
    }
    return emojis.get(status, "❓")

def build_main_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Строит главную клавиатуру с учетом прав."""
    buttons = [
        [InlineKeyboardButton(text="📊 Статус", callback_data="status")],
        [InlineKeyboardButton(text="📈 График", callback_data="graph")],
        [InlineKeyboardButton(text="💬 Чат с Буслом", callback_data="chat")],
    ]
    
    if is_admin(user_id):
        buttons.extend([
            [
                InlineKeyboardButton(text="▶️ Старт", callback_data="start_training"),
                InlineKeyboardButton(text="⏸️ Пауза", callback_data="pause"),
            ],
            [
                InlineKeyboardButton(text="▶️ Возобновить", callback_data="resume"),
                InlineKeyboardButton(text="🛑 Стоп", callback_data="stop"),
            ],
            [InlineKeyboardButton(text="📋 Логи", callback_data="logs")],
        ])
    
    buttons.append([InlineKeyboardButton(text="ℹ️ Помощь", callback_data="help")])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def call_inference_api(prompt: str, max_length: int = 200, temperature: float = 0.8) -> str:
    """Вызывает API инференса."""
    try:
        payload = json.dumps({
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.15
        }).encode("utf-8")
        
        req = urllib.request.Request(
            f"{INFERENCE_URL}/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("generated_text", "[Пустой ответ]")
    except urllib.error.URLError as e:
        return f"❌ API недоступен: {e}\n\nЗапустите: `uv run cli.py serve`"
    except Exception as e:
        return f"❌ Ошибка: {e}"

def generate_status_report() -> str:
    """Генерирует текстовый отчет о статусе."""
    state = get_state()
    status = state.get("status", "idle")
    metrics = get_latest_metrics()
    eta_info = estimate_eta()
    
    # Статус
    emoji = get_status_emoji(status)
    status_text = {
        "idle": "Ожидание запуска",
        "running": "Обучение активно",
        "paused": "На паузе",
        "stopped": "Остановлено",
        "finished": "Завершено успешно"
    }.get(status, status)
    
    lines = [
        f"{emoji} *Бусел — {status_text}*",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]
    
    if status in ["running", "paused"] or metrics:
        step = state.get("current_step", metrics.get("step", 0))
        max_steps = state.get("max_steps", 0)
        profile = state.get("profile", "unknown")
        
        progress = (step / max_steps * 100) if max_steps > 0 else 0
        progress_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
        
        lines.extend([
            f"📁 *Профиль:* `{profile}`",
            f"🎯 *Прогресс:* `{step:,}/{max_steps:,}`",
            f"`{progress_bar}` {progress:.1f}%",
            "",
        ])
        
        if metrics:
            lines.extend([
                f"📉 *Лосс:* `{metrics.get('loss', 0):.4f}`",
                f"🧬 *Aux Loss:* `{metrics.get('aux_loss', 0):.4f}`",
                f"⚡ *Скорость:* `{metrics.get('speed', 0):.0f} tok/s`",
                f"🎓 *Learning Rate:* `{metrics.get('lr', 0):.2e}`",
            ])
            
            if metrics.get("vram"):
                lines.append(f"💾 *VRAM:* `{metrics['vram']:.0f} MB`")
            
            lines.append("")
        
        # ETA
        if status == "running" and eta_info.get("eta_str"):
            lines.extend([
                f"⏱️ *ETA:* `{eta_info['eta_str']}`",
                f"📊 *Осталось шагов:* `{eta_info.get('steps_remaining', 0):,}`",
            ])
        
        # Время работы
        started_at = state.get("started_at")
        if started_at:
            total_pause = state.get("total_pause_time", 0)
            elapsed = time.time() - started_at - total_pause
            lines.append(f"⏰ *Время работы:* `{format_duration(elapsed)}`")
        
        if state.get("paused_at") and status == "paused":
            pause_duration = time.time() - state["paused_at"]
            lines.append(f"⏸️ *На паузе:* `{format_duration(pause_duration)}`")
    
    elif status == "idle":
        lines.extend([
            "",
            "🚀 Бусел готов к обучению!",
            "Используйте /start\\_training для запуска",
        ])
    
    # Heartbeat check
    if status == "running" and not is_alive():
        lines.extend([
            "",
            "⚠️ *Сердцебиение потеряно!*"
        ])
    
    lines.append("")
    lines.append(f"🕐 _{datetime.now().strftime('%H:%M:%S')}_")
    
    return "\n".join(lines)

async def send_training_control(action: str, admin_id: int) -> str:
    """Отправляет команду управления обучением."""
    state = get_state()
    current_status = state.get("status", "idle")
    
    if action == "pause":
        if current_status != "running":
            return f"❌ Нельзя поставить на паузу (статус: {current_status})"
        set_status("paused")
        return "⏸️ Обучение поставлено на паузу"
    
    elif action == "resume":
        if current_status != "paused":
            return f"❌ Нельзя возобновить (статус: {current_status})"
        set_status("running")
        return "▶️ Обучение возобновлено"
    
    elif action == "stop":
        if current_status not in ["running", "paused"]:
            return f"❌ Нельзя остановить (статус: {current_status})"
        set_status("stopped")
        return "🛑 Обучение остановлено (сохранен checkpoint)"
    
    elif action == "start":
        if current_status == "running":
            return "❌ Обучение уже запущено"
        return "🚀 Для запуска используйте: `uv run cli.py train --profile shpak`"
    
    return "❓ Неизвестная команда"

# ═══════════════════════════════════════════════════════════════
# РОУТЕР И ОБРАБОТЧИКИ
# ═══════════════════════════════════════════════════════════════

router = Router()

@router.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext):
    """Приветствие и главное меню."""
    user_id = message.from_user.id
    is_adm = is_admin(user_id)
    
    welcome_text = f"""
🦩 *Привет, {message.from_user.first_name}!*

Я — *Бусел*, суверенный 1-битный AI-ассистент.

{"👑 *Вы администратор*" if is_adm else "👤 *Вы гость*"}

📊 Мониторинг обучения
💬 Чат с моделью
{"🎛️ Управление обучением" if is_adm else ""}

Выберите действие:
"""
    
    await message.answer(
        welcome_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=build_main_keyboard(user_id)
    )

@router.message(Command("help"))
@router.callback_query(F.data == "help")
async def cmd_help(event: types.Message | types.CallbackQuery):
    """Справка по командам."""
    user_id = event.from_user.id
    is_adm = is_admin(user_id)
    
    help_text = """
ℹ️ *Справка по командам*

📊 *Мониторинг:*
/status — Текущий статус обучения
/graph — График прогресса
/metrics — Последние метрики

💬 *Общение:*
/chat <текст> — Поговорить с Буслом
Просто напишите сообщение для диалога
"""
    
    if is_adm:
        help_text += """
🎛️ *Управление (только админ):*
/start\\_training — Инструкция по запуску
/pause — Поставить на паузу
/resume — Возобновить обучение
/stop — Остановить обучение
/logs — Последние логи

⚙️ *Настройки:*
/admins — Список админов
/setup — Перезапустить настройку
"""
    else:
        help_text += f"""
⏱️ *Лимиты:*
• {PUBLIC_RATE_LIMIT} запросов в минуту
• До {MAX_GENERATION_LENGTH} символов ответа
"""
    
    if isinstance(event, types.CallbackQuery):
        await event.message.edit_text(help_text, parse_mode=ParseMode.MARKDOWN)
        await event.answer()
    else:
        await event.answer(help_text, parse_mode=ParseMode.MARKDOWN)

@router.message(Command("status"))
@router.callback_query(F.data == "status")
async def cmd_status(event: types.Message | types.CallbackQuery):
    """Статус обучения."""
    if isinstance(event, types.CallbackQuery):
        await event.answer("📊 Обновляю статус...")
        await event.message.edit_text(
            generate_status_report(),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=build_main_keyboard(event.from_user.id)
        )
    else:
        await event.answer(
            generate_status_report(),
            parse_mode=ParseMode.MARKDOWN
        )

@router.message(Command("graph"))
@router.callback_query(F.data == "graph")
async def cmd_graph(event: types.Message | types.CallbackQuery):
    """График обучения."""
    from tools.plotter import generate_report_plot
    
    log_path = "checkpoints/metrics.jsonl"
    plot_path = "checkpoints/training_report.png"
    
    if isinstance(event, types.CallbackQuery):
        await event.answer("📈 Генерирую график...")
        msg = event.message
    else:
        msg = event
        await msg.answer("📈 Генерирую график...")
    
    success = generate_report_plot(log_path, plot_path)
    
    if success and os.path.exists(plot_path):
        photo = FSInputFile(plot_path)
        await msg.answer_photo(
            photo,
            caption="📊 *Отчет Бусла*",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await msg.answer("❌ Нет данных для графика. Начните обучение!")

@router.message(Command("logs"))
@router.callback_query(F.data == "logs")
async def cmd_logs(event: types.Message | types.CallbackQuery, state: FSMContext):
    """Последние логи (только админ)."""
    user_id = event.from_user.id
    
    if not is_admin(user_id):
        if isinstance(event, types.CallbackQuery):
            await event.answer("🚫 Только для админов", show_alert=True)
        else:
            await event.answer("🚫 Эта команда доступна только администраторам")
        return
    
    metrics = get_metrics_history(max_points=10)
    
    if not metrics:
        text = "📋 Логов пока нет"
    else:
        lines = ["📋 *Последние 10 шагов:*\n"]
        for m in metrics[-10:]:
            lines.append(
                f"`{m.get('step', 0):05d}` | "
                f"Loss: `{m.get('loss', 0):.3f}` | "
                f"Speed: `{m.get('speed', 0):.0f}` tok/s"
            )
        text = "\n".join(lines)
    
    if isinstance(event, types.CallbackQuery):
        await event.answer()
        await event.message.edit_text(text, parse_mode=ParseMode.MARKDOWN)
    else:
        await event.answer(text, parse_mode=ParseMode.MARKDOWN)

@router.message(Command("pause"))
@router.callback_query(F.data == "pause")
async def cmd_pause(event: types.Message | types.CallbackQuery, state: FSMContext):
    """Пауза обучения."""
    user_id = event.from_user.id
    
    if not is_admin(user_id):
        if isinstance(event, types.CallbackQuery):
            await event.answer("🚫 Только для админов", show_alert=True)
        else:
            await event.answer("🚫 Только администраторы могут управлять обучением")
        return
    
    result = await send_training_control("pause", user_id)
    
    if isinstance(event, types.CallbackQuery):
        await event.answer()
        await event.message.edit_text(
            result,
            reply_markup=build_main_keyboard(user_id)
        )
    else:
        await event.answer(result)

@router.message(Command("resume"))
@router.callback_query(F.data == "resume")
async def cmd_resume(event: types.Message | types.CallbackQuery, state: FSMContext):
    """Возобновление обучения."""
    user_id = event.from_user.id
    
    if not is_admin(user_id):
        if isinstance(event, types.CallbackQuery):
            await event.answer("🚫 Только для админов", show_alert=True)
        else:
            await event.answer("🚫 Только администраторы могут управлять обучением")
        return
    
    result = await send_training_control("resume", user_id)
    
    if isinstance(event, types.CallbackQuery):
        await event.answer()
        await event.message.edit_text(
            result,
            reply_markup=build_main_keyboard(user_id)
        )
    else:
        await event.answer(result)

@router.message(Command("stop"))
@router.callback_query(F.data == "stop")
async def cmd_stop(event: types.Message | types.CallbackQuery, state: FSMContext):
    """Остановка обучения с подтверждением."""
    user_id = event.from_user.id
    
    if not is_admin(user_id):
        if isinstance(event, types.CallbackQuery):
            await event.answer("🚫 Только для админов", show_alert=True)
        else:
            await event.answer("🚫 Только администраторы могут управлять обучением")
        return
    
    confirm_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Да, остановить", callback_data="confirm_stop"),
            InlineKeyboardButton(text="❌ Отмена", callback_data="cancel_stop"),
        ]
    ])
    
    text = "⚠️ *Вы уверены, что хотите остановить обучение?*\n\nБудет сохранен emergency checkpoint."
    
    if isinstance(event, types.CallbackQuery):
        await event.answer()
        await event.message.edit_text(
            text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=confirm_keyboard
        )
    else:
        await event.answer(text, parse_mode=ParseMode.MARKDOWN, reply_markup=confirm_keyboard)
    
    await state.set_state(BotStates.confirming_stop)

@router.callback_query(F.data == "confirm_stop", StateFilter(BotStates.confirming_stop))
async def confirm_stop_callback(callback: types.CallbackQuery, state: FSMContext):
    """Подтверждение остановки."""
    await state.clear()
    result = await send_training_control("stop", callback.from_user.id)
    await callback.message.edit_text(
        result,
        reply_markup=build_main_keyboard(callback.from_user.id)
    )
    await callback.answer()

@router.callback_query(F.data == "cancel_stop", StateFilter(BotStates.confirming_stop))
async def cancel_stop_callback(callback: types.CallbackQuery, state: FSMContext):
    """Отмена остановки."""
    await state.clear()
    await callback.message.edit_text(
        "❌ Остановка отменена",
        reply_markup=build_main_keyboard(callback.from_user.id)
    )
    await callback.answer()

@router.message(Command("start_training"))
@router.callback_query(F.data == "start_training")
async def cmd_start_training(event: types.Message | types.CallbackQuery, state: FSMContext):
    """Инструкция по запуску обучения."""
    user_id = event.from_user.id
    
    if not is_admin(user_id):
        if isinstance(event, types.CallbackQuery):
            await event.answer("🚫 Только для админов", show_alert=True)
        else:
            await event.answer("🚫 Только администраторы могут запускать обучение")
        return
    
    current_state = get_state()
    
    if current_state.get("status") == "running":
        text = "🔥 Обучение уже запущено!"
    else:
        text = """
🚀 *Запуск обучения*

Для старта выполните в терминале:

```
uv run cli.py train --profile shpak
```

Или для другого профиля:
```
uv run cli.py train --profile zubr
```

После запуска используйте /status для мониторинга.
"""
    
    if isinstance(event, types.CallbackQuery):
        await event.answer()
        await event.message.edit_text(text, parse_mode=ParseMode.MARKDOWN)
    else:
        await event.answer(text, parse_mode=ParseMode.MARKDOWN)

@router.message(Command("chat"))
async def cmd_chat(message: types.Message, state: FSMContext):
    """Начало чата с моделью."""
    args = message.text.split(maxsplit=1)
    
    if len(args) > 1:
        # Есть текст сразу
        await process_chat_message(message, args[1])
    else:
        await message.answer(
            "💬 *Режим чата с Буслом*\n\n"
            "Просто отправьте мне сообщение, и я отвечу!\n"
            "Для выхода: /cancel",
            parse_mode=ParseMode.MARKDOWN
        )
        await state.set_state(BotStates.waiting_for_chat)

@router.message(Command("cancel"))
async def cmd_cancel(message: types.Message, state: FSMContext):
    """Отмена текущего действия."""
    await state.clear()
    await message.answer(
        "✅ Отменено",
        reply_markup=build_main_keyboard(message.from_user.id)
    )

@router.message(Command("admins"))
async def cmd_admins(message: types.Message):
    """Список админов."""
    if not is_admin(message.from_user.id):
        await message.answer("🚫 Только для админов")
        return
    
    admin_ids = get_admin_ids()
    text = "👑 *Администраторы:*\n\n"
    for aid in admin_ids:
        marker = " ← это вы" if aid == message.from_user.id else ""
        text += f"• `{aid}`{marker}\n"
    
    await message.answer(text, parse_mode=ParseMode.MARKDOWN)

@router.message(StateFilter(BotStates.waiting_for_chat))
async def process_chat_state(message: types.Message, state: FSMContext):
    """Обработка сообщений в режиме чата."""
    await process_chat_message(message, message.text)

async def process_chat_message(message: types.Message, text: str):
    """Основная логика чата с моделью."""
    user_id = message.from_user.id
    is_adm = is_admin(user_id)
    
    # Rate limiting для не-админов
    if not is_adm:
        allowed, wait_time = rate_limiter.check(user_id)
        if not allowed:
            await message.answer(
                f"⏱️ *Лимит превышен!*\n\n"
                f"Подождите {wait_time} секунд.\n\n"
                f"Лимит: {PUBLIC_RATE_LIMIT} запросов в минуту",
                parse_mode=ParseMode.MARKDOWN
            )
            return
    
    # Показываем "печатает..."
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    
    max_length = ADMIN_MAX_LENGTH if is_adm else MAX_GENERATION_LENGTH
    
    # Вызываем API в отдельном потоке
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: call_inference_api(text, max_length=max_length)
    )
    
    # Форматируем ответ
    if response.startswith("❌"):
        await message.answer(response)
    else:
        # Обрезаем если слишком длинный
        if len(response) > 3900:
            response = response[:3900] + "..."
        
        footer = ""
        if not is_adm:
            remaining = PUBLIC_RATE_LIMIT - len(rate_limiter.requests[user_id])
            footer = f"\n\n_Запросов осталось: {remaining}_"
        
        await message.answer(
            f"🦩 *Бусел отвечает:*\n\n{response}{footer}",
            parse_mode=ParseMode.MARKDOWN
        )

# Обработка обычных сообщений (без команды) — автоматический чат
@router.message(F.text & ~F.text.startswith("/"))
async def handle_regular_message(message: types.Message, state: FSMContext):
    """Автоматический чат для обычных сообщений."""
    current_state = await state.get_state()
    
    # Если не в режиме чата и не админ — игнорируем
    if current_state != BotStates.waiting_for_chat and not is_admin(message.from_user.id):
        await message.answer(
            "👋 Используйте /start для начала работы или /chat для общения!",
            reply_markup=build_main_keyboard(message.from_user.id)
        )
        return
    
    await process_chat_message(message, message.text)

# ═══════════════════════════════════════════════════════════════
# ЗАПУСК
# ═══════════════════════════════════════════════════════════════

async def main():
    """Главная функция запуска бота."""
    if not API_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN не установлен!")
        print("   Запустите: uv run cli.py bot")
        sys.exit(1)
    
    bot = Bot(token=API_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    
    # Уведомляем админов о запуске
    admin_ids = get_admin_ids()
    for admin_id in admin_ids:
        try:
            await bot.send_message(
                admin_id,
                "🦩 *Бусел онлайн!*\n\nБот запущен и готов к работе.",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception:
            pass
    
    print(f"🤖 Бусел-бот запущен!")
    print(f"   Админы: {admin_ids}")
    print(f"   API: {INFERENCE_URL}")
    
    # Удаляем webhook и запускаем polling
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())