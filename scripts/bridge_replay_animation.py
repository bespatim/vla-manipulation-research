# /// script
# requires-python = ">=3.10"
# dependencies = ["visual-kinematics", "numpy", "scipy", "matplotlib", "pillow"]
# ///
"""Повтор записанной траектории BridgeData V2 на кинематической модели WidowX 250 6DOF.

Позы захвата из датасета (x, y, z, roll, pitch, yaw) → обратная кинематика по DH-параметрам
(visual-kinematics) → углы суставов → анимация: слева кадр камеры из датасета, справа робот.
Заодно проверяется, достижимы ли записанные позы в системе координат основания Interbotix:
это подтверждает (или опровергает) наши предположения о системе координат и порядке углов.

    uv run scripts/bridge_replay_animation.py --episode-dir examples/bridge_val000_ep34 \
        --out chapter1/images/bridge_replay.gif
"""

import argparse
import csv
import io
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from visual_kinematics.Frame import Frame
from visual_kinematics.RobotSerial import RobotSerial

# DH WidowX 250 6DOF (docs/robot_widowx250s.md): | d | a | alpha | offset |
ELBOW = np.arctan2(0.25, 0.04975)
DH = np.array([
    [0.11065, 0.0, -np.pi / 2, 0.0],
    [0.0, np.hypot(0.04975, 0.25), 0.0, -ELBOW],
    [0.0, 0.0, -np.pi / 2, ELBOW - np.pi / 2],
    [0.25, 0.0, np.pi / 2, 0.0],
    [0.0, 0.0, -np.pi / 2, 0.0],
    [0.158575, 0.0, 0.0, 0.0],
])
# 6-й DH-кадр → ee_gripper_link (только поворот)
R_TOOL = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)
# BridgeData хранит углы не самого захвата, а R_ee · DEFAULT_ROTATIONᵀ: нулевые углы = пальцы вниз.
# Источник: rail-berkeley/bridge_data_robot, widowx_controller.py (DEFAULT_ROTATION)
# и transformation_utils.py (transform2state / state2transform, R = Rz·Ry·Rx).
DEFAULT_ROTATION = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
LIMITS = np.radians([[-180, 180], [-108, 114], [-123, 92], [-180, 180], [-100, 123], [-180, 180]])


def load_episode(ep: Path):
    with open(ep / "trajectory.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pos = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in rows])
    rpy = np.array([[float(r[k]) for k in ("roll", "pitch", "yaw")] for r in rows])
    grip = np.array([float(r["gripper"]) for r in rows])
    gif = Image.open(ep / "frames.gif")
    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        frames.append(np.asarray(gif.convert("RGB")))
    return pos, rpy, grip, frames


def solve_ik(robot, pos, rpy):
    """IK по каждой позе с тёплым стартом от предыдущего решения."""
    logging.disable(logging.ERROR)
    robot.forward(np.radians([0, -20, 20, 0, 40, 0]))
    qs, ok = [], []
    for p, e in zip(pos, rpy):
        r_ee = Rotation.from_euler("xyz", e).as_matrix() @ DEFAULT_ROTATION
        target = Frame.from_r_3_3(r_ee @ R_TOOL.T, p.reshape(3, 1))
        robot.inverse(target)
        q = robot.axis_values.copy()
        reached = bool(robot.is_reachable_inverse) and np.all((q >= LIMITS[:, 0]) & (q <= LIMITS[:, 1]))
        qs.append(q)
        ok.append(reached)
    logging.disable(logging.NOTSET)
    return np.array(qs), np.array(ok)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    pos, rpy, grip, frames = load_episode(args.episode_dir)
    robot = RobotSerial(DH, plot_xlim=[-0.1, 0.45], plot_ylim=[-0.25, 0.25], plot_zlim=[0.0, 0.5],
                        max_iter=500, final_loss=1e-5)

    qs, ok = solve_ik(robot, pos, rpy)
    jump = np.degrees(np.abs(np.diff(np.unwrap(qs, axis=0), axis=0)).max(axis=0))
    fingers_z = (Rotation.from_euler("xyz", rpy).as_matrix() @ DEFAULT_ROTATION)[:, 2, 0]
    print(f"достижимо {ok.sum()}/{len(ok)} поз в пределах суставов")
    print("макс. изменение сустава за шаг, °:", np.round(jump, 1))
    print(f"ось пальцев вниз (z < -0.7) в {np.mean(fingers_z < -0.7) * 100:.0f}% поз")

    out_frames = []
    fig = robot.figure
    fig.set_size_inches(3.2, 3.2)
    for t, q in enumerate(qs):
        robot.forward(q)
        robot.draw()
        ax = robot.ax
        ax.plot(pos[: t + 1, 0], pos[: t + 1, 1], pos[: t + 1, 2], color="tab:orange", lw=1.5)
        ax.set_title(f"шаг {t:2d}  захват {'открыт' if grip[t] > 0.6 else 'закрыт'}"
                     + ("" if ok[t] else "  (IK!)"), fontsize=8)
        ax.view_init(elev=25, azim=-60 + 0.6 * t)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        kin = Image.open(buf).convert("RGB").resize((256, 256))
        cam = Image.fromarray(frames[t]).resize((256, 256))
        canvas = Image.new("RGB", (512, 256), "white")
        canvas.paste(cam, (0, 0))
        canvas.paste(kin, (256, 0))
        out_frames.append(canvas)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_frames[0].save(args.out, save_all=True, append_images=out_frames[1:], duration=200, loop=0)
    plt.close(fig)
    print(f"сохранено {args.out}: {len(out_frames)} кадров")


if __name__ == "__main__":
    main()
