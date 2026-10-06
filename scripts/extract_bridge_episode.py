# /// script
# requires-python = ">=3.10"
# dependencies = ["tfrecord", "protobuf", "numpy", "pillow", "scipy", "matplotlib"]
# ///
"""Извлечение одного эпизода BridgeData V2 (валидационная часть) без TensorFlow.

Скачивает один шард валидационной выборки (~130 МБ, ~54 эпизода) с сервера
Berkeley и сохраняет выбранный эпизод:
  - trajectory.csv  — поза захвата, кватернион и действие оператора на каждом шаге
  - frames.gif      — видео с основной неподвижной камеры (image_0)
  - frames/         — те же кадры отдельными PNG (их будем подавать в OpenVLA)
  - contact_sheet.png, trajectory.png — превью для README

Запуск:
    uv run scripts/extract_bridge_episode.py --list
    uv run scripts/extract_bridge_episode.py --episode 34
"""

import argparse
import csv
import io
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tfrecord import example_pb2
from tfrecord.reader import tfrecord_iterator

BASE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/"
ROOT = Path(__file__).resolve().parent.parent


def load_shard(shard: int) -> list:
    name = f"bridge_dataset-val.tfrecord-{shard:05d}-of-00128"
    path = ROOT / "data" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Скачиваю {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, path)
    episodes = []
    for rec in tfrecord_iterator(str(path)):
        ex = example_pb2.Example()
        ex.ParseFromString(rec)
        episodes.append(ex.features.feature)
    return episodes


def instruction(f) -> str:
    values = f["steps/language_instruction"].bytes_list.value
    return values[0].decode() if values and values[0] else ""


def save_episode(f, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # state: x, y, z [м], roll, pitch, yaw [рад], захват (1 — открыт, 0 — закрыт)
    state = np.array(f["steps/observation/state"].float_list.value).reshape(-1, 7)
    # action: dx, dy, dz [м], droll, dpitch, dyaw [рад], захват — команда оператора на шаге
    action = np.array(f["steps/action"].float_list.value).reshape(-1, 7)
    frames = [Image.open(io.BytesIO(b)).convert("RGB")
              for b in f["steps/observation/image_0"].bytes_list.value]
    task = instruction(f)

    # Углы в BridgeData — от R_ee · DEFAULT_ROTATIONᵀ (R = Rz·Ry·Rx), поэтому ориентация самого
    # захвата (ee_gripper_link в системе основания Interbotix) = R(углы) · DEFAULT_ROTATION.
    # Источник: rail-berkeley/bridge_data_robot, widowx_controller.py и transformation_utils.py
    default_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    r_ee = Rotation.from_euler("xyz", state[:, 3:6]).as_matrix() @ default_rotation
    quat = Rotation.from_matrix(r_ee).as_quat(scalar_first=True)

    with open(out / "trajectory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "x", "y", "z", "roll", "pitch", "yaw", "qw", "qx", "qy", "qz", "gripper",
                    "act_dx", "act_dy", "act_dz", "act_droll", "act_dpitch", "act_dyaw", "act_gripper"])
        for t in range(len(state)):
            w.writerow([t, *np.round(state[t, :6], 6), *np.round(quat[t], 6),
                        round(state[t, 6], 4), *np.round(action[t], 6)])

    (out / "frames").mkdir(exist_ok=True)
    for t, im in enumerate(frames):
        im.save(out / "frames" / f"{t:03d}.png")
    frames[0].save(out / "frames.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
    (out / "instruction.txt").write_text(task + "\n", encoding="utf-8")

    # Превью: 8 кадров равномерно по эпизоду
    idx = np.linspace(0, len(frames) - 1, 8).astype(int)
    sheet = Image.new("RGB", (256 * 4, 256 * 2), "white")
    for k, t in enumerate(idx):
        sheet.paste(frames[t], (256 * (k % 4), 256 * (k // 4)))
    sheet.save(out / "contact_sheet.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, name in enumerate("xyz"):
        axes[0].plot(state[:, i], label=name)
    axes[0].set(title="Положение захвата, м", xlabel="шаг (5 Гц)")
    axes[0].legend()
    axes[1].plot(state[:, 6], label="захват (state)")
    axes[1].plot(action[:, 6], "--", label="захват (action)")
    axes[1].set(title="Захват: 1 — открыт, 0 — закрыт", xlabel="шаг (5 Гц)")
    axes[1].legend()
    fig.suptitle(f"BridgeData V2 val: «{task}»")
    fig.tight_layout()
    fig.savefig(out / "trajectory.png", dpi=120)
    print(f"Сохранено в {out}: {len(frames)} шагов, задача «{task}»")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0, help="номер валидационного шарда 0..127")
    p.add_argument("--episode", type=int, default=34, help="номер эпизода внутри шарда")
    p.add_argument("--list", action="store_true", help="только показать эпизоды шарда")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    episodes = load_shard(args.shard)
    if args.list:
        for i, f in enumerate(episodes):
            n = len(f["steps/observation/image_0"].bytes_list.value)
            print(f"{i:3d} {n:3d} шагов | {instruction(f) or '<без инструкции>'}")
        return
    out = args.out or ROOT / "examples" / f"bridge_val{args.shard:03d}_ep{args.episode:02d}"
    save_episode(episodes[args.episode], out)


if __name__ == "__main__":
    main()
