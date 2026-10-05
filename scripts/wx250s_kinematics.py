# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Кинематика WidowX 250 6DOF (wx250s) — робота, на котором записан BridgeData V2.

Источник: открытый код Trossen Robotics (Interbotix):
  - винтовые оси (product of exponentials):
    interbotix_ros_toolboxes / interbotix_xs_modules / xs_robot / mr_descriptions.py, class wx250s
  - пределы суставов: interbotix_ros_manipulators / interbotix_xsarm_descriptions / urdf / wx250s.urdf.xacro

Здесь эти данные пересчитаны в классические параметры Денавита–Хартенберга (DH)
и проверено, что обе модели дают одинаковую прямую кинематику.

Нулевая поза: рука вытянута горизонтально вперёд вдоль оси X базы.
Система координат базы: X — вперёд, Z — вверх, начало — в основании робота.
Конечная точка — ee_gripper_link (центр между пальцами захвата).

Запуск проверки:
    uv run scripts/wx250s_kinematics.py
"""

import numpy as np
from scipy.linalg import expm

# --- Модель производителя: винтовые оси в системе базы (ω, v), метры -----------
SLIST = np.array([
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],             # 1 waist (поворот основания)
    [0.0, 1.0, 0.0, -0.11065, 0.0, 0.0],        # 2 shoulder (плечо)
    [0.0, 1.0, 0.0, -0.36065, 0.0, 0.04975],    # 3 elbow (локоть)
    [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0],         # 4 forearm_roll (вращение предплечья)
    [0.0, 1.0, 0.0, -0.36065, 0.0, 0.29975],    # 5 wrist_angle (наклон кисти)
    [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0],         # 6 wrist_rotate (вращение кисти)
]).T
M_HOME = np.array([
    [1.0, 0.0, 0.0, 0.458325],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.36065],
    [0.0, 0.0, 0.0, 1.0],
])

# --- Пределы суставов из URDF, градусы ----------------------------------------
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
JOINT_LIMITS_DEG = np.array([
    [-180, 180],
    [-108, 114],
    [-123, 92],
    [-180, 180],
    [-100, 123],
    [-180, 180],
])

# --- Классические DH-параметры (выведены из модели выше) ----------------------
# θ_i = q_i + offset_i;  T_i = Rz(θ_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)
UPPER_ARM = np.hypot(0.04975, 0.25)               # 0.254901 м: плечо → локоть
ELBOW_OFFSET = np.arctan2(0.25, 0.04975)          # 78.746°: излом плечевого звена
DH = np.array([
    # a [м]       alpha [рад]    d [м]       offset [рад]
    [0.0,         -np.pi / 2,    0.11065,    0.0],
    [UPPER_ARM,    0.0,          0.0,       -ELBOW_OFFSET],
    [0.0,         -np.pi / 2,    0.0,        ELBOW_OFFSET - np.pi / 2],
    [0.0,          np.pi / 2,    0.25,       0.0],
    [0.0,         -np.pi / 2,    0.0,        0.0],
    [0.0,          0.0,          0.158575,   0.0],
])
# Постоянный поворот от 6-го DH-кадра к ee_gripper_link (X захвата — вперёд, Z — вверх)
T_TOOL = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def _se3(screw: np.ndarray, theta: float) -> np.ndarray:
    w, v = screw[:3], screw[3:]
    m = np.zeros((4, 4))
    m[:3, :3] = [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]
    m[:3, 3] = v
    return expm(m * theta)


def fk_poe(q: np.ndarray) -> np.ndarray:
    """Прямая кинематика по модели производителя."""
    t = np.eye(4)
    for i in range(6):
        t = t @ _se3(SLIST[:, i], q[i])
    return t @ M_HOME


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk_dh(q: np.ndarray) -> np.ndarray:
    """Прямая кинематика по DH-параметрам."""
    t = np.eye(4)
    for (a, alpha, d, offset), qi in zip(DH, q):
        t = t @ dh_transform(a, alpha, d, qi + offset)
    return t @ T_TOOL


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    print("DH-параметры WidowX 250 6DOF (классические, θ = q + offset):")
    print(f"{'звено':>5} {'a, м':>10} {'alpha, °':>10} {'d, м':>10} {'offset, °':>10}   сустав, пределы")
    for i, (a, alpha, d, off) in enumerate(DH):
        lo, hi = JOINT_LIMITS_DEG[i]
        print(f"{i + 1:>5} {a:>10.6f} {np.degrees(alpha):>10.2f} {d:>10.6f} {np.degrees(off):>10.3f}"
              f"   {JOINT_NAMES[i]} [{lo}°, {hi}°]")

    print("\nНулевая поза, ee_gripper_link:\n", fk_dh(np.zeros(6)))

    rng = np.random.default_rng(0)
    lo, hi = np.radians(JOINT_LIMITS_DEG).T
    err = max(np.abs(fk_dh(q) - fk_poe(q)).max() for q in rng.uniform(lo, hi, size=(10_000, 6)))
    print(f"\nПроверка на 10 000 случайных конфигурациях: max |FK_DH − FK_производителя| = {err:.2e}")
    assert err < 1e-9, "DH-модель не совпадает с моделью производителя"
    print("DH-параметры совпадают с моделью производителя.")


if __name__ == "__main__":
    main()
