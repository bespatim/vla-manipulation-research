# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "matplotlib", "pillow", "imageio[ffmpeg]"]
# ///
"""Разбор результатов OpenVLA в LIBERO (вывод notebooks/libero_openvla_colab.ipynb).

Считает успех с 95% доверительным интервалом, классифицирует неудачи по записанной физике
и готовит картинки для главы 1:
  - series_failures.png        — кадры всех неудачных эпизодов серии
  - force_success_vs_empty.png — сила контакта с миской: успешный эпизод против «пустого захвата»
  - empty_gripper.gif          — пример «пустого захвата» с камерой на запястье
  - cookie_attempt{1,2}_steps.png — ключевые кадры попыток с коробкой печенья

    uv run scripts/analyze_libero_results.py --results results/libero_spatial --out chapter1/images
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Шрифт с кириллицей (встроенный шрифт PIL её не умеет)
FONT = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), 12)
SUITE = "libero_spatial"
TARGET = "akita_black_bowl_1"
CONTACT_N = 1.0          # порог «есть контакт», Н
PAPER = 84.7             # Приложение E, Таблица 12


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def episode_stats(log: pd.DataFrame) -> dict:
    eef = log[["eef_x", "eef_y", "eef_z"]].to_numpy()
    path = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    moved = np.flatnonzero(np.cumsum(path) > 0.05)          # шаг, когда захват прошёл 5 см
    bowl = log[[f"{TARGET}_x", f"{TARGET}_y", f"{TARGET}_z"]].to_numpy()
    plate = log[["plate_1_x", "plate_1_y"]].to_numpy()
    closed = log.act_env_6 > 0
    touch = log[f"force_{TARGET}"] > CONTACT_N
    return {
        "start_move_step": int(moved[0]) if len(moved) else None,
        "closed_steps": int(closed.sum()),
        "touch_target_steps": int(touch.sum()),
        "max_force_target_N": round(float(log[f"force_{TARGET}"].max()), 1),
        "max_lift_cm": round(100 * float(bowl[:, 2].max() - bowl[0, 2]), 1),
        "bowl_to_plate_cm": round(100 * float(np.linalg.norm(bowl[-1, :2] - plate[-1])), 1),
        "other_contact_steps": int((log.force_other > CONTACT_N).sum()),
    }


def failure_type(s: dict) -> str:
    if s["start_move_step"] is None or s["start_move_step"] > 150:
        return "замирание"
    if s["max_lift_cm"] > 5:
        return "донёс, но не поставил"
    if s["closed_steps"] > 0 and s["touch_target_steps"] == 0 and s["other_contact_steps"] < 10:
        return "пустой захват"
    if s["other_contact_steps"] >= 10:
        return "столкновение с другим предметом"
    return "другое"


def frame_sheet(videos, steps, out, size=192, wrist=False, cols=None):
    """Сетка кадров: строка на видео (или по `cols` кадров в строке). wrist=True — вместе с камерой запястья."""
    rows = []
    for label, path in videos:
        v = iio.imread(path)
        tiles = []
        for s in steps:
            f = v[min(s, len(v) - 1)]
            im = Image.fromarray(f if wrist else f[:, :256]).resize((size * (2 if wrist else 1), size))
            ImageDraw.Draw(im).text((4, 3), f"{label} · шаг {s}", fill=(220, 0, 0), font=FONT)
            tiles.append(np.asarray(im))
        cols_ = cols or len(tiles)
        rows += [np.concatenate(tiles[c:c + cols_], axis=1) for c in range(0, len(tiles), cols_)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)


def to_gif(mp4, out, stride=3, width=448):
    v = iio.imread(mp4)
    frames = [Image.fromarray(f).resize((width, width * f.shape[0] // f.shape[1])) for f in v[::stride]]
    pal = frames[len(frames) // 2].quantize(colors=128)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 * stride / 20), loop=0, optimize=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    series = args.results / "series"
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(series / "summary.csv")
    k, n = int(df.success.sum()), len(df)
    lo, hi = wilson(k, n)
    print(f"Успех: {100 * k / n:.1f}% ({k}/{n}), 95% ДИ [{lo:.0f}%, {hi:.0f}%] | у авторов {PAPER}%")
    print(f"Время на шаг: {df.infer_s_per_step.mean():.2f} с\n")

    print("| Задача | Команда | Успех |\n|---|---|---|")
    for t, g in df.groupby("task_id"):
        print(f"| {t} | {g.instruction.iloc[0]} | {int(g.success.sum())}/{len(g)} |")

    rows = []
    for _, r in df.iterrows():
        log = pd.read_csv(series / f"{SUITE}_task{r.task_id:02d}_trial{r.trial:02d}.csv")
        s = episode_stats(log)
        s.update(task=r.task_id, trial=r.trial, success=r.success, steps=r.steps,
                 type="успех" if r.success else failure_type(s))
        rows.append(s)
    stats = pd.DataFrame(rows)
    stats.to_csv(args.results / "series_episode_stats.csv", index=False)
    print("\nНеудачи:")
    print(stats[~stats.success][["task", "trial", "type", "start_move_step", "closed_steps",
                                 "touch_target_steps", "max_force_target_N", "max_lift_cm",
                                 "bowl_to_plate_cm"]].to_string(index=False))
    print("\n", stats[~stats.success].type.value_counts().to_string())

    fails = stats[~stats.success]
    frame_sheet([(f"з{t} п{i}", series / f"{SUITE}_task{t:02d}_trial{i:02d}.mp4")
                 for t, i in fails[["task", "trial"]].values],
                [40, 90, 140, 190, 219], args.out / "series_failures.png")

    # Сила контакта: успешный эпизод против «пустого захвата» на той же задаче
    empty = fails[fails.type == "пустой захват"]
    if len(empty):
        t, i = empty[["task", "trial"]].values[-1]
        ok = stats[(stats.task == t) & stats.success]
        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        cases = [(f"задача {t}, попытка {i}: провал", i, "tab:red")]
        if len(ok):
            cases.insert(0, (f"задача {t}, попытка {ok.trial.iloc[0]}: успех", ok.trial.iloc[0], "tab:green"))
        for label, trial, color in cases:
            log = pd.read_csv(series / f"{SUITE}_task{t:02d}_trial{trial:02d}.csv")
            ax[0].plot(log.step, (log.act_env_6 > 0).astype(int), color=color, label=label)
            ax[1].plot(log.step, log[f"force_{TARGET}"], color=color, label=label)
        ax[0].set_ylabel("команда\n«сжать»"); ax[0].set_yticks([0, 1]); ax[0].legend(fontsize=8)
        ax[1].set_ylabel("сила контакта\nс миской, Н"); ax[1].set_xlabel("шаг (20 Гц)")
        fig.suptitle("Модель сжимает захват в обоих случаях; сила контакта — только при успехе", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out / "force_success_vs_empty.png", dpi=120)
        to_gif(series / f"{SUITE}_task{t:02d}_trial{i:02d}.mp4", args.out / "empty_gripper.gif")
        print(f"\nпример пустого захвата: задача {t}, попытка {i}")

    custom = sorted((args.results / "custom").glob("*/"))
    for j, d in enumerate(custom, 1):
        mp4 = next(d.glob("*.mp4"))
        log = pd.read_csv(next(d.glob(f"{SUITE}_*.csv")))
        c = log[["cookies_1_x", "cookies_1_y", "cookies_1_z"]].to_numpy()
        b = log[[f"{TARGET}_x", f"{TARGET}_y", f"{TARGET}_z"]].to_numpy()
        print(f"\n{d.name}: сила на коробке печенья max {log.force_cookies_1.max():.1f} Н, "
              f"коробка сдвинута на {100 * np.linalg.norm(c[-1, :2] - c[0, :2]):.1f} см, "
              f"поднята на {100 * (c[:, 2].max() - c[0, 2]):.1f} см | сила на миске max "
              f"{log[f'force_{TARGET}'].max():.1f} Н, миска поднята на {100 * (b[:, 2].max() - b[0, 2]):.1f} см")
    named = {"pick_up_the_cookie_box": 1, "pick_up_the_red_cookie_box": 2}
    for d in custom:
        j = next((v for key, v in named.items() if d.name.startswith(key + "_")), None)
        if j is not None:
            frame_sheet([(f"попытка {j}", next(d.glob("*.mp4")))], [90, 125, 145, 165, 190, 219],
                        args.out / f"cookie_attempt{j}_steps.png", size=160, wrist=True, cols=3)


if __name__ == "__main__":
    main()
