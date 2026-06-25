"""
📈 busel
Генерирует минималистичные графики в стиле Google Cloud / Vertex AI.
"""

import os
import json
import numpy as np


def _ema(data, alpha=0.15):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed


def _load_metrics(log_path):
    """Parse JSONL metrics log into 6 parallel numpy arrays. Returns None if the file is missing or empty."""
    if not os.path.exists(log_path):
        return None
    steps, losses, aux_losses, speeds, lrs, vrams = [], [], [], [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                steps.append(data["step"])
                losses.append(data["loss"])
                aux_losses.append(data["aux_loss"])
                speeds.append(data["speed"])
                lrs.append(data["lr"])
                vrams.append(data.get("vram", 0.0))
            except Exception:
                continue
    if not steps:
        return None
    return tuple(np.array(x) for x in (steps, losses, aux_losses, speeds, lrs, vrams))


def _compute_dashboard_stats(steps, losses, speeds, vrams, cumulative_tokens):
    return {
        "min_loss_val": float(np.min(losses)),
        "min_loss_step": int(steps[np.argmin(losses)]),
        "avg_speed_val": float(np.mean(speeds[10:]) if len(speeds) > 10 else np.mean(speeds)),
        "max_vram_val": float(np.max(vrams)),
        "total_tokens_val": float(cumulative_tokens[-1]),
    }


def generate_report_plot(log_path="checkpoints/metrics.jsonl", output_path="checkpoints/training_report.png",
                         tokens_per_step: int = 262144):
    """Parse JSONL metrics log and render a 3-panel flat Google Material chart."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    metrics = _load_metrics(log_path)
    if metrics is None:
        return False
    steps, losses, aux_losses, speeds, lrs, vrams = metrics

    cumulative_tokens = steps * tokens_per_step

    loss_smoothed = _ema(losses, alpha=0.2)
    aux_smoothed = _ema(aux_losses, alpha=0.2)
    speed_smoothed = _ema(speeds, alpha=0.2)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['text.color'] = '#202124'
    plt.rcParams['axes.labelcolor'] = '#5F6368'
    plt.rcParams['xtick.color'] = '#5F6368'
    plt.rcParams['ytick.color'] = '#5F6368'

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 14), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    ax1.set_facecolor("#FFFFFF")
    ax1.plot(steps, losses, color="#1A73E8", alpha=0.15, linewidth=1.0)
    line_loss_smooth = ax1.plot(steps, loss_smoothed, color="#1A73E8", linewidth=2.0, label="Total Loss")
    ax1.set_ylabel("Total Loss", color="#1A73E8", fontweight="medium", fontsize=9)
    ax1.tick_params(axis='y', labelcolor="#1A73E8")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(steps, aux_losses, color="#EA4335", alpha=0.15, linewidth=1.0, linestyle="--")
    line_aux_smooth = ax1_twin.plot(steps, aux_smoothed, color="#EA4335", linewidth=1.5, linestyle="--", label="Aux Loss (MoE)")
    ax1_twin.set_ylabel("Aux Loss", color="#EA4335", fontweight="medium", fontsize=9)
    ax1_twin.tick_params(axis='y', labelcolor="#EA4335")

    ax1.set_title("Loss Convergence & Expert Balance", fontsize=10.5, fontweight="bold", color="#202124", loc="left", pad=10)
    lines_1 = line_loss_smooth + line_aux_smooth
    ax1.legend(lines_1, [l.get_label() for l in lines_1], loc="upper right", frameon=False, fontsize=8.5)

    ax2.set_facecolor("#FFFFFF")
    ax2.plot(steps, speeds, color="#34A853", alpha=0.15, linewidth=1.0)
    line_speed_smooth = ax2.plot(steps, speed_smoothed, color="#34A853", linewidth=2.0, label="Throughput")
    ax2.set_ylabel("Throughput (tokens/s)", color="#34A853", fontweight="medium", fontsize=9)
    ax2.tick_params(axis='y', labelcolor="#34A853")

    ax2_twin = ax2.twinx()
    line_vram = ax2_twin.plot(steps, vrams, color="#9333EA", alpha=0.5, linewidth=1.2, linestyle=":", label="VRAM Allocated")
    ax2_twin.set_ylabel("VRAM Allocated (MB)", color="#9333EA", fontweight="medium", fontsize=9)
    ax2_twin.tick_params(axis='y', labelcolor="#9333EA")

    ax2.set_title("System Compute Throughput & Memory", fontsize=10.5, fontweight="bold", color="#202124", loc="left", pad=10)
    lines_2 = line_speed_smooth + line_vram
    ax2.legend(lines_2, [l.get_label() for l in lines_2], loc="upper left", frameon=False, fontsize=8.5)

    ax3.set_facecolor("#FFFFFF")
    line_lr = ax3.plot(steps, lrs, color="#F9AB00", linewidth=1.8, linestyle="-.", label="Learning Rate")
    ax3.set_ylabel("Learning Rate", color="#F9AB00", fontweight="medium", fontsize=9)
    ax3.tick_params(axis='y', labelcolor="#F9AB00")
    ax3.set_xlabel("Training Steps", fontweight="medium", fontsize=9)

    ax3_twin = ax3.twinx()
    line_tokens = ax3_twin.plot(steps, cumulative_tokens / 1e3, color="#12B5CB", linewidth=1.2, linestyle=":", label="Processed Volume")
    ax3_twin.set_ylabel("Cumulative Volume (K tokens)", color="#12B5CB", fontweight="medium", fontsize=9)
    ax3_twin.tick_params(axis='y', labelcolor="#12B5CB")

    ax3.set_title("Learning Rate Decay & Cumulative Data", fontsize=10.5, fontweight="bold", color="#202124", loc="left", pad=10)
    lines_3 = line_lr + line_tokens
    ax3.legend(lines_3, [l.get_label() for l in lines_3], loc="upper right", frameon=False, fontsize=8.5)

    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#E0E0E0')
        ax.spines['bottom'].set_linewidth(1.0)
        ax.yaxis.grid(True, color='#F1F3F4', linestyle='-', linewidth=1.0)
        ax.xaxis.grid(False)
        ax.tick_params(axis='both', which='both', length=0)

    for ax_t in [ax1_twin, ax2_twin, ax3_twin]:
        ax_t.spines['top'].set_visible(False)
        ax_t.spines['right'].set_visible(False)
        ax_t.spines['left'].set_visible(False)
        ax_t.spines['bottom'].set_visible(False)
        ax_t.tick_params(axis='both', which='both', length=0)

    stats = _compute_dashboard_stats(steps, losses, speeds, vrams, cumulative_tokens)

    rect = patches.FancyBboxPatch(
        (0.06, 0.905), 0.88, 0.07,
        boxstyle="round,pad=0.0,rounding_size=0.015",
        facecolor="#F8F9FA", edgecolor="#F1F3F4", linewidth=1.0,
        transform=fig.transFigure, figure=fig, zorder=-1
    )
    fig.patches.append(rect)

    columns = [
        (0.16, "MINIMUM LOSS",    f"{stats['min_loss_val']:.4f}",      "#1A73E8", f"at Step {stats['min_loss_step']}"),
        (0.38, "COMPUTE SPEED",   f"{stats['avg_speed_val']:.1f} tok/s", "#34A853", "CUDA"),
        (0.62, "CUMULATIVE VOLUME", f"{stats['total_tokens_val'] / 1e3:.1f} Ktok", "#12B5CB", "Byte Tokens Processed"),
        (0.84, "PEAK MEMORY",     f"{stats['max_vram_val']:.1f} MB",   "#9333EA", "CUDA"),
    ]
    for x, label, value, color, sublabel in columns:
        fig.text(x, 0.955, label, fontsize=7.5, color="#5F6368", fontweight="bold", ha="center")
        fig.text(x, 0.932, value, fontsize=13, color=color, fontweight="bold", ha="center")
        fig.text(x, 0.915, sublabel, fontsize=7.5, color="#70757A", ha="center")

    plt.subplots_adjust(top=0.87, hspace=0.38)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches='tight')
    plt.close()
    return True