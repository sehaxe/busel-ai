"""
🤖 SOVEREIGN TELEGRAM BOT v1.0
Поддерживает команды /start и /status (генерация красивых графиков на лету).
"""

import os
import sys
import json
import urllib.request
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command

# Корректируем пути
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
API_URL = "http://127.0.0.1:8000/generate"

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def send_welcome(message: types.Message):
    await message.reply(
        "Hello! 👋\n\n"
        "I am Bysel (v4.0), a sovereign Any-to-Text AI powered by NVIDIA's GDN-2 hybrid linear attention.\n\n"
        "Available commands:\n"
        "📊 /status - Generate training progress report & beautiful cyberpunk charts\n"
        "✍️ Or just send me any text to generate a completion!"
    )


@dp.message(Command("status"))
async def send_status_report(message: types.Message):
    """
    Генерирует на лету неоновый график обучения и отправляет красивый отчет.
    """
    from tools.plotter import generate_report_plot
    
    log_path = "checkpoints/metrics.jsonl"
    plot_path = "checkpoints/training_report.png"
    
    # 1. Генерируем график
    success = generate_report_plot(log_path, plot_path)
    
    if not success or not os.path.exists(log_path):
        await message.reply("📊 *No training metrics found yet.* Start the training loop first!", parse_mode="Markdown")
        return
        
    # 2. Считываем показатели последнего шага
    last_metrics = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_metrics = json.loads(line)
                
    if not last_metrics:
        await message.reply("❌ Failed to parse training metrics.")
        return
        
    # 3. Формируем стильный TUI-отчет
    report_text = (
        f"📊 *Bysel Training Status Report*\n"
        f"───────────────────────────\n"
        f"🚶 *Current Step:* `{last_metrics['step']:05d}`\n"
        f"📉 *Total Loss:* `{last_metrics['loss']:.4f}`\n"
        f"🧬 *Aux Loss:* `{last_metrics['aux_loss']:.4f}`\n"
        f"🚀 *Instant Speed:* `{last_metrics['speed']:.0f} tokens/s`\n"
        f"⚡ *Learning Rate:* `{last_metrics['lr']:.5f}`\n"
        f"💾 *VRAM Usage:* `{last_metrics['vram']:.0f} MB`\n"
        f"───────────────────────────\n"
        f"📈 _Dynamic training curves are attached in the image below!_"
    )
    
    # 4. Отправляем график и текст в Telegram
    photo = types.FSInputFile(plot_path)
    await message.reply_photo(photo, caption=report_text, parse_mode="Markdown")


@dp.message()
async def generate_response(message: types.Message):
    prompt = message.text
    payload = json.dumps({
        "prompt": prompt,
        "max_length": 120,
        "temperature": 0.7
    }).encode("utf-8")
    
    req = urllib.request.Request(
        API_URL, 
        data=payload, 
        headers={"Content-Type": "application/json"}
    )
    
    status_msg = await message.reply("⚡ *Bysel is thinking...*", parse_mode="Markdown")
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: urllib.request.urlopen(req, timeout=30).read()
        )
        data = json.loads(response.decode("utf-8"))
        generated_text = data["generated_text"]
        
        await status_msg.edit_text(generated_text)
    except Exception as e:
        await status_msg.edit_text(f"❌ Failed to reach model: {e}. Make sure 'cli.py serve' is running!")


async def main():
    print("🤖 Telegram Bot Bysel is running...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    if API_TOKEN == "YOUR_BOT_TOKEN":
        print("❌ ERROR: Define TELEGRAM_BOT_TOKEN in .env before running the bot!")
        sys.exit(1)
    asyncio.run(main())