# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "matplotlib", "pillow", "imageio[ffmpeg]"]
# ///
"""Разбор результатов OpenVLA-OFT в LIBERO (вывод notebooks/libero_oft_colab.ipynb) для главы 2.

Ожидаемая структура папки --results (как в Colab: /content/results/oft_libero_spatial):
  series/<условие>/          — модель авторов, «заморозка» входов (image_only, no_proprio, no_wrist, full)
  series_drop/<условие>/     — модель авторов, отключение входов флагами (drop_wrist, drop_proprio, drop_both)
  series_no_wrist/<условие>/ — модель, обученная без камеры на запястье (full, no_proprio)
Для сравнения берутся результаты OpenVLA из главы 1 (--openvla, results/libero_spatial/series).

    uv run scripts/analyze_oft_results.py --results results/oft_libero_spatial \
        --openvla results/libero_spatial/series --out chapter2/images
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

FONT = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), 12)
SUITE = "libero_spatial"
TARGET = "akita_black_bowl_1"

# (папка, условие, группа, подпись)
RUNS = [
    ("series", "full", "модель авторов", "2 камеры + датчики (как у авторов)"),
    ("series", "no_wrist", "заморозка", "камера запястья заморожена"),
    ("series", "no_proprio", "заморозка", "датчики заморожены"),
    ("series", "image_only", "заморозка", "запястье и датчики заморожены"),
    ("series_drop", "drop_wrist", "отключение", "камера запястья отключена"),
    ("series_drop", "drop_proprio", "отключение", "датчики отключены"),
    ("series_drop", "drop_both", "отключение", "запястье и датчики отключены"),
    ("series_no_wrist", "full", "обучена без запястья", "1 камера + датчики"),
    ("series_no_wrist", "no_proprio", "обучена без запястья", "датчики заморожены"),
]


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def failure_type(log: pd.DataFrame) -> str:
    """Те же правила, что в главе 1, плюс «ни разу не сжал захват»."""
    eef = log[["eef_x", "eef_y", "eef_z"]].to_numpy()
    moved = np.flatnonzero(np.cumsum(np.linalg.norm(np.diff(eef, axis=0), axis=1)) > 0.05)
    if not len(moved) or moved[0] > 150:
        return "замирание"
    if log[f"{TARGET}_z"].max() - log[f"{TARGET}_z"].iloc[0] > 0.05:
        return "донёс, но не поставил"
    if (log.act_env_6 > 0).sum() == 0:
        return "ни разу не сжал захват"
    touch = (log[f"force_{TARGET}"] > 1).sum()
    other = (log.force_other > 1).sum()
    if touch == 0 and other < 10:
        return "пустой захват"
    return "столкновение" if other >= 10 else "другое"


def episode_csv(folder: Path, task: int, trial: int) -> Path:
    return folder / f"{SUITE}_task{task:02d}_trial{trial:02d}.csv"


def load_runs(root: Path):
    rows, frames = [], {}
    for folder, cond, group, label in RUNS:
        d = root / folder / cond
        if not (d / "summary.csv").exists():
            continue
        s = pd.read_csv(d / "summary.csv")
        k, n = int(s.success.sum()), len(s)
        lo, hi = wilson(k, n)
        types = [failure_type(pd.read_csv(episode_csv(d, r.task_id, r.trial))) for _, r in s[~s.success].iterrows()]
        rows.append({"группа": group, "условие": label, "успех, %": round(100 * k / n, 1), "k": k, "n": n,
                     "ДИ_low": lo, "ДИ_high": hi, "шагов (успех)": s[s.success].steps.mean(),
                     "с на эпизод (успех)": s[s.success].wall_s.mean(), "с на запрос": s.infer_s_per_query.mean(),
                     "запросов на эпизод": s.queries.mean(),
                     "неудачи": {k_: int(v) for k_, v in pd.Series(types, dtype=object).value_counts().items()}})
        frames[(folder, cond)] = s
    return pd.DataFrame(rows), frames


def openvla_baseline(path: Path) -> dict:
    s = pd.read_csv(path / "summary.csv")
    k, n = int(s.success.sum()), len(s)
    lo, hi = wilson(k, n)
    return {"группа": "OpenVLA (глава 1)", "условие": "1 камера, старое дообучение", "успех, %": round(100 * k / n, 1),
            "k": k, "n": n, "ДИ_low": lo, "ДИ_high": hi, "шагов (успех)": s[s.success].steps.mean(),
            "с на эпизод (успех)": s[s.success].wall_s.mean(), "с на запрос": s.infer_s_per_step.mean(),
            "запросов на эпизод": s.steps.mean()}


def plot_success(table: pd.DataFrame, out: Path) -> None:
    t = table.iloc[::-1]
    colors = {"модель авторов": "#2b8a3e", "заморозка": "#c92a2a", "отключение": "#e8590c",
              "обучена без запястья": "#1971c2", "OpenVLA (глава 1)": "#868e96"}
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.barh(y, t["успех, %"], color=[colors[g] for g in t["группа"]])
    ax.errorbar(t["успех, %"], y, xerr=[t["успех, %"] - t["ДИ_low"], t["ДИ_high"] - t["успех, %"]],
                fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_yticks(y, [f"{g}: {c}  ({k}/{n})" for g, c, k, n in zip(t["группа"], t["условие"], t["k"], t["n"])],
                  fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("успех, % (чёрным — 95% доверительный интервал)")
    ax.set_title("LIBERO-Spatial, 10 задач: OpenVLA-OFT при отключении и заморозке входов", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_drift(root: Path, task: int, trial: int, out: Path) -> None:
    """Вид сверху: куда едет захват при живых, замороженных и отключённых входах."""
    cases = [("series", "full", "все входы живые", "#2b8a3e"),
             ("series", "no_wrist", "запястье заморожено", "#f08c00"),
             ("series", "no_proprio", "датчики заморожены", "#c92a2a"),
             ("series", "image_only", "запястье и датчики заморожены", "#862e9c"),
             ("series_drop", "drop_both", "запястье и датчики отключены", "#1971c2")]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ref = None
    for folder, cond, label, color in cases:
        p = episode_csv(root / folder / cond, task, trial)
        if not p.exists():
            continue
        log = pd.read_csv(p)
        ref = log if ref is None else ref
        ax.plot(log.eef_x, log.eef_y, color=color, lw=1.6, label=label)
        ax.plot(log.eef_x.iloc[-1], log.eef_y.iloc[-1], "o", color=color, ms=5)
    if ref is not None:
        ax.plot(ref.eef_x.iloc[0], ref.eef_y.iloc[0], "k^", ms=9, label="старт захвата")
        ax.plot(ref[f"{TARGET}_x"].iloc[0], ref[f"{TARGET}_y"].iloc[0], "s", color="black", ms=11, mfc="none",
                label="нужная миска")
        ax.plot(ref.plate_1_x.iloc[0], ref.plate_1_y.iloc[0], "o", color="black", ms=16, mfc="none", label="тарелка")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"Путь захвата (вид сверху), задача {task}, попытка {trial}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def to_gif(mp4: Path, out: Path, label: str, stride: int = 2, width: int = 448) -> None:
    v = iio.imread(mp4)
    frames = []
    for f in v[::stride]:
        im = Image.fromarray(f).resize((width, width * f.shape[0] // f.shape[1]))
        ImageDraw.Draw(im).text((6, 4), label, fill=(255, 255, 0), font=FONT)
        frames.append(im)
    pal = frames[len(frames) // 2].quantize(colors=128)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 * stride / 20), loop=0, optimize=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--openvla", type=Path, default=Path("results/libero_spatial/series"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--example-task", type=int, default=2, help="задача для графика пути и GIF")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table, _ = load_runs(args.results)
    if args.openvla.exists():
        table = pd.concat([table, pd.DataFrame([openvla_baseline(args.openvla)])], ignore_index=True)
    table.to_csv(args.results / "summary_all_conditions.csv", index=False)

    print("| Группа | Условие | Успех | 95% ДИ | шагов (успех) | с на эпизод (успех) | с на запрос | запросов |")
    print("|---|---|---|---|---|---|---|---|")
    for _, r in table.iterrows():
        print(f"| {r['группа']} | {r['условие']} | {r['успех, %']:.1f}% ({r.k}/{r.n}) | "
              f"[{r.ДИ_low:.0f}%, {r.ДИ_high:.0f}%] | {r['шагов (успех)']:.0f} | {r['с на эпизод (успех)']:.1f} | "
              f"{r['с на запрос']:.2f} | {r['запросов на эпизод']:.0f} |")
    print("\nТипы неудач:")
    for _, r in table.iterrows():
        if isinstance(r.get("неудачи"), dict) and r["неудачи"]:
            print(f"  {r['группа']}: {r['условие']}: {r['неудачи']}")

    plot_success(table, args.out / "oft_success_by_condition.png")
    plot_drift(args.results, args.example_task, 0, args.out / "oft_drift_topview.png")
    for folder, cond, label in [("series", "full", "все входы живые"),
                                ("series", "image_only", "запястье и датчики заморожены"),
                                ("series", "no_wrist", "камера запястья заморожена")]:
        mp4 = args.results / folder / cond / f"{SUITE}_task{args.example_task:02d}_trial00.mp4"
        if mp4.exists():
            to_gif(mp4, args.out / f"oft_{cond}_task{args.example_task:02d}.gif", label)
    print("\nкартинки:", sorted(x.name for x in args.out.iterdir()))


if __name__ == "__main__":
    main()
