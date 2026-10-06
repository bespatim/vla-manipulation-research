"""Прогон OpenVLA в симуляторе LIBERO в замкнутом цикле с записью физики.

Повторяет логику официального скрипта OpenVLA
(experiments/robot/libero/run_libero_eval.py) без зависимости от всего пакета OpenVLA
и дополнительно на каждом шаге записывает то, чего модель НЕ видит:
  - силу контакта пальцев захвата с каждым целевым предметом задачи (из физики MuJoCo);
  - положение целевых предметов;
  - положение суставов и захвата;
  - кадр с камеры на запястье (в видео, рядом с кадром, который видит модель).

Используется из ноутбука notebooks/libero_openvla_colab.ipynb:
    import libero_openvla_eval as E
    E.setup_libero("/content/LIBERO")
    policy = E.OpenVLAPolicy("libero_spatial")
    E.evaluate(policy, "libero_spatial", task_ids=[0], trials_per_task=1, out_dir="results/libero")

Локальная проверка симулятора без модели (скриптовый робот):
    python scripts/libero_openvla_eval.py --libero-root <путь к LIBERO> --policy scripted --suite libero_object
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Шаги эпизода — как в официальном скрипте (самая длинная демонстрация + запас)
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
NUM_STEPS_WAIT = 10                       # первые шаги симулятор «роняет» предметы на стол
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
CHECKPOINTS = {s: f"openvla/openvla-7b-finetuned-{s.replace('_', '-')}"
               for s in ("libero_spatial", "libero_object", "libero_goal")}
CHECKPOINTS["libero_10"] = "openvla/openvla-7b-finetuned-libero-10"
# Результаты авторов: Приложение E, Таблица 12 (500 попыток × 3 зерна)
PAPER_SUCCESS = {"libero_spatial": 84.7, "libero_object": 88.4, "libero_goal": 79.2, "libero_10": 53.7}


# --------------------------------------------------------------------------------------
# LIBERO
# --------------------------------------------------------------------------------------
def setup_libero(libero_root: str) -> None:
    """Создаёт конфиг LIBERO без интерактивного вопроса и добавляет пакет в sys.path."""
    root = Path(libero_root).resolve()
    bench = root / "libero" / "libero"
    cfg_dir = root / ".libero_config"
    cfg_dir.mkdir(exist_ok=True)
    paths = {"benchmark_root": bench, "bddl_files": bench / "bddl_files", "init_states": bench / "init_files",
             "datasets": root / "datasets", "assets": bench / "assets"}
    (cfg_dir / "config.yaml").write_text("".join(f"{k}: {v}\n" for k, v in paths.items()))
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def make_env(suite_name: str, task_id: int, resolution: int = 256):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)  # как в официальном скрипте: влияет на положение предметов
    return env, task.language, suite.get_task_init_states(task_id), suite.n_tasks


class ContactProbe:
    """Сила контакта пальцев захвата с предметами сцены — то, чего модель не получает."""

    def __init__(self, env, objects):
        import mujoco

        self._mujoco = mujoco
        self.m, self.d = env.sim.model._model, env.sim.data._data
        name = lambda kind, i: mujoco.mj_id2name(self.m, kind, i) or ""
        self.gripper = {g for g in range(self.m.ngeom)
                        if name(mujoco.mjtObj.mjOBJ_GEOM, g).startswith("gripper0_")}
        self.objects = list(objects)
        # геом → имя предмета по корневому телу («alphabet_soup_1_main» → «alphabet_soup_1»)
        self.owner = {}
        for g in range(self.m.ngeom):
            root = name(mujoco.mjtObj.mjOBJ_BODY, self.m.body_rootid[self.m.geom_bodyid[g]])
            self.owner[g] = next((o for o in self.objects if root.startswith(o)), None)
        self._f6 = np.zeros(6)

    def measure(self) -> dict:
        forces = {o: 0.0 for o in self.objects}
        other, n = 0.0, 0
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            g1, g2 = c.geom1 in self.gripper, c.geom2 in self.gripper
            if g1 == g2:
                continue
            self._mujoco.mj_contactForce(self.m, self.d, k, self._f6)
            partner = c.geom2 if g1 else c.geom1
            obj = self.owner.get(partner)
            if obj is None:
                other += abs(self._f6[0])
            else:
                forces[obj] += abs(self._f6[0])
            n += 1
        return {**{f"force_{o}": f for o, f in forces.items()}, "force_other": other, "n_contacts": n}


# --------------------------------------------------------------------------------------
# Предобработка кадра — как при обучении OpenVLA
# --------------------------------------------------------------------------------------
def _tf():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # иначе TF займёт всю видеопамять
    return tf


def preprocess_image(agentview: np.ndarray, size: int = 224) -> np.ndarray:
    """Поворот на 180°, JPEG-кодирование и lanczos3-ресайз, как в загрузчике данных OpenVLA."""
    img = np.ascontiguousarray(agentview[::-1, ::-1])
    try:
        tf = _tf()
        x = tf.io.decode_image(tf.image.encode_jpeg(img), expand_animations=False, dtype=tf.uint8)
        x = tf.image.resize(x, (size, size), method="lanczos3", antialias=True)
        return tf.cast(tf.clip_by_value(tf.round(x), 0, 255), tf.uint8).numpy()
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((size, size), Image.LANCZOS))


def center_crop(img: np.ndarray, crop_scale: float = 0.9) -> np.ndarray:
    """Центральный кроп 90% площади и ресайз обратно (модель дообучали со случайным кропом)."""
    try:
        tf = _tf()
        x = tf.image.convert_image_dtype(tf.convert_to_tensor(img), tf.float32)[None]
        s = float(np.sqrt(crop_scale))
        o = (1 - s) / 2
        x = tf.image.crop_and_resize(x, [[o, o, o + s, o + s]], [0], img.shape[:2])[0]
        return tf.image.convert_image_dtype(tf.clip_by_value(x, 0, 1), tf.uint8, saturate=True).numpy()
    except ImportError:
        from PIL import Image

        h, w = img.shape[:2]
        s = np.sqrt(crop_scale)
        top, left = int(round(h * (1 - s) / 2)), int(round(w * (1 - s) / 2))
        crop = img[top:top + int(round(h * s)), left:left + int(round(w * s))]
        return np.asarray(Image.fromarray(crop).resize((w, h), Image.BILINEAR))


# --------------------------------------------------------------------------------------
# Политики
# --------------------------------------------------------------------------------------
class OpenVLAPolicy:
    """OpenVLA, дообученная авторами на наборе LIBERO. На T4 грузится в 4 битах."""

    def __init__(self, suite_name: str, checkpoint: str | None = None, load_in_4bit: bool | None = None):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        self.torch = torch
        self.checkpoint = checkpoint or CHECKPOINTS[suite_name]
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.load_in_4bit = gpu_mem < 20 if load_in_4bit is None else load_in_4bit
        self.dtype = torch.float16 if self.load_in_4bit else torch.bfloat16
        kwargs = dict(torch_dtype=self.dtype, low_cpu_mem_usage=True, trust_remote_code=True)
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(self.checkpoint, **kwargs)
        if not self.load_in_4bit:
            self.model = self.model.to("cuda")
        stats = self.model.norm_stats
        self.unnorm_key = suite_name if suite_name in stats else f"{suite_name}_no_noops"
        assert self.unnorm_key in stats, f"нет ключа {suite_name} в norm_stats: {list(stats)}"
        print(f"OpenVLA: {self.checkpoint} | {'4-bit' if self.load_in_4bit else 'bf16'} | "
              f"видеопамять {torch.cuda.memory_allocated() / 1e9:.1f} ГБ из {gpu_mem:.0f}")

    def __call__(self, image: np.ndarray, instruction: str, obs: dict) -> np.ndarray:
        from PIL import Image

        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(center_crop(image))).to("cuda", dtype=self.dtype)
        with self.torch.inference_mode():
            return self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)


class ScriptedPolicy:
    """Простой скриптовый робот для проверки симулятора и записи без нейросети:
    подлетает к первому целевому предмету, опускается, сжимает и поднимает."""

    def __init__(self, target: str):
        self.target, self.t = target, 0

    def __call__(self, image, instruction, obs):
        self.t += 1
        eef, obj = obs["robot0_eef_pos"], obs[f"{self.target}_pos"]
        if self.t < 40:
            goal, is_open = obj + [0, 0, 0.10], True      # над предметом
        elif self.t < 70:
            goal, is_open = obj + [0, 0, 0.01], True      # опуститься
        elif self.t < 85:
            goal, is_open = eef, False                    # сжать
        else:
            goal, is_open = obj + [0, 0, 0.25], False     # поднять
        a = np.zeros(7)
        a[:3] = np.clip((goal - eef) / 0.02, -1, 1)
        a[6] = 1.0 if is_open else 0.0                    # формат OpenVLA: 1 — открыть, 0 — закрыть
        return a


def to_env_action(action: np.ndarray) -> np.ndarray:
    """Захват: [0,1] → [-1,+1], бинаризация и смена знака (в LIBERO −1 — открыть, +1 — закрыть)."""
    a = np.asarray(action, dtype=np.float64).copy()
    a[-1] = -np.sign(2 * a[-1] - 1)
    return a


# --------------------------------------------------------------------------------------
# Эпизод и оценка
# --------------------------------------------------------------------------------------
def make_goal_check(env, goal_on):
    """Своя цель «A лежит на B» через предикат LIBERO On(A, B) = B.check_ontop(A)."""
    if goal_on is None:
        return None
    obj, support = goal_on
    states = env.env.object_states_dict
    missing = [o for o in goal_on if o not in states]
    assert not missing, f"нет предметов {missing} в сцене; есть: {sorted(states)}"
    return lambda: bool(states[support].check_ontop(states[obj]))


def run_episode(policy, env, instruction, init_state, max_steps, objects, video_path=None, log_path=None,
                goal_on=None, use_env_goal=True):
    """goal_on=(A, B): успех, когда A лежит на B. use_env_goal=False: не засчитывать исходную цель сцены
    (нужно, когда команда своя, а сцена от другой задачи)."""
    import imageio

    env.reset()
    obs = env.set_init_state(init_state)
    probe = ContactProbe(env, objects)
    goal_check = make_goal_check(env, goal_on)
    rows, frames, success, t, step = [], [], False, 0, 0
    infer_time = 0.0
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(DUMMY_ACTION)
            t += 1
            continue
        image = preprocess_image(obs["agentview_image"])
        t0 = time.time()
        raw = np.asarray(policy(image, instruction, obs), dtype=np.float64)
        dt = time.time() - t0
        infer_time += dt
        action = to_env_action(raw)

        row = {"step": step, "infer_s": round(dt, 3)}
        row.update({f"eef_{a}": v for a, v in zip("xyz", obs["robot0_eef_pos"])})
        row.update({f"eef_q{a}": v for a, v in zip("xyzw", obs["robot0_eef_quat"])})
        row.update({f"gripper_q{i}": v for i, v in enumerate(obs["robot0_gripper_qpos"])})
        row.update({f"joint_{i}": v for i, v in enumerate(obs["robot0_joint_pos"])})
        row.update({f"act_raw_{i}": v for i, v in enumerate(raw)})
        row.update({f"act_env_{i}": v for i, v in enumerate(action)})
        for o in objects:
            if f"{o}_pos" in obs:
                row.update({f"{o}_{a}": v for a, v in zip("xyz", obs[f"{o}_pos"])})
        row.update(probe.measure())
        rows.append(row)
        if video_path is not None:
            wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
            frames.append(np.concatenate([obs["agentview_image"][::-1, ::-1], wrist], axis=1))

        obs, _, done, _ = env.step(action.tolist())
        t += 1
        step += 1
        if (goal_check is not None and goal_check()) or (use_env_goal and done):
            success = True
            break

    if log_path is not None and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(log_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    if video_path is not None and frames:
        imageio.mimsave(video_path, frames, fps=20)  # 20 Гц — частота управления LIBERO: видео в реальном времени
    return {"success": success, "steps": step, "infer_s_per_step": infer_time / max(step, 1)}


def evaluate(policy, suite_name, task_ids=None, trials_per_task=1, out_dir="results/libero",
             save_video=True, seed=7, instruction=None, goal_on=None, max_steps=None):
    """instruction — своя команда вместо задачи сцены; goal_on=(A, B) — своя проверка успеха «A на B».
    Со своей командой исходная цель сцены не засчитывается."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.csv"
    results = []
    if task_ids is None:
        from libero.libero import benchmark

        task_ids = range(benchmark.get_benchmark_dict()[suite_name]().n_tasks)
    for task_id in task_ids:
        env, task_instruction, init_states, _ = make_env(suite_name, task_id)
        command = instruction or task_instruction
        objects = list(goal_on) if goal_on else []
        objects += [o for o in env.obj_of_interest if o not in objects]
        if isinstance(policy, type) and policy is ScriptedPolicy:
            make_policy = lambda: ScriptedPolicy(objects[0])
        else:
            make_policy = lambda: policy
        for trial in range(trials_per_task):
            name = f"{suite_name}_task{task_id:02d}_trial{trial:02d}"
            t0 = time.time()
            r = run_episode(make_policy(), env, command, init_states[trial], max_steps or MAX_STEPS[suite_name],
                            objects, video_path=out / f"{name}.mp4" if save_video else None,
                            log_path=out / f"{name}.csv", goal_on=goal_on, use_env_goal=instruction is None)
            r.update({"suite": suite_name, "task_id": task_id, "trial": trial, "instruction": command,
                      "objects": " ".join(objects), "wall_s": round(time.time() - t0, 1)})
            results.append(r)
            done = sum(x["success"] for x in results)
            print(f"[{len(results)}] {instruction} | успех={r['success']} | шагов={r['steps']} | "
                  f"{r['infer_s_per_step']:.2f} с/шаг | итого {done}/{len(results)} ({100 * done / len(results):.0f}%)")
            with open(summary_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(results[0]))
                w.writeheader()
                w.writerows(results)
        env.close()
    rate = 100 * sum(x["success"] for x in results) / len(results)
    if instruction is None:
        print(f"\nУспех: {rate:.1f}% на {len(results)} эпизодах | у авторов: {PAPER_SUCCESS.get(suite_name)}%")
    else:
        goal = f"{goal_on[0]} на {goal_on[1]}" if goal_on else "не задана — смотрите видео"
        print(f"\nСвоя команда «{instruction}» | цель: {goal} | успех: {rate:.0f}% на {len(results)} эпизодах")
    (out / "meta.json").write_text(json.dumps({
        "suite": suite_name, "episodes": len(results), "success_rate": rate,
        "paper_success_rate": PAPER_SUCCESS.get(suite_name) if instruction is None else None,
        "instruction": instruction, "goal_on": goal_on,
        "checkpoint": getattr(policy, "checkpoint", "scripted"),
        "load_in_4bit": getattr(policy, "load_in_4bit", None), "seed": seed}, ensure_ascii=False, indent=2))
    return results


def save_preview(suite_name: str, task_id: int, path: str) -> None:
    """Три кадра без модели: сырая внешняя камера | что получает модель | камера на запястье."""
    from PIL import Image

    env, instruction, init_states, _ = make_env(suite_name, task_id)
    env.reset()
    obs = env.set_init_state(init_states[0])
    raw = obs["agentview_image"][::-1, ::-1]
    model_view = np.asarray(Image.fromarray(center_crop(preprocess_image(obs["agentview_image"])))
                            .resize(raw.shape[1::-1]))
    wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
    Image.fromarray(np.concatenate([raw, model_view, wrist], axis=1)).save(path)
    print(f"{suite_name}, задача {task_id}: «{instruction}» | целевые предметы: {list(env.obj_of_interest)}")
    print("все предметы сцены:", sorted(env.env.object_states_dict))
    env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--libero-root", required=True)
    p.add_argument("--suite", default="libero_spatial", choices=list(MAX_STEPS))
    p.add_argument("--tasks", type=int, nargs="*", default=None)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--policy", choices=["openvla", "scripted"], default="openvla")
    p.add_argument("--out", default="results/libero")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--preview", default=None, help="только сохранить превью сцены (без модели) и выйти")
    p.add_argument("--instruction", default=None, help="своя команда роботу (по-английски) вместо задачи сцены")
    p.add_argument("--goal-on", nargs=2, metavar=("A", "B"), default=None,
                   help="своя проверка успеха: предмет A лежит на предмете B, например cookies_1 plate_1")
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    # Colab передаёт дочерним процессам свой inline-бэкенд matplotlib, которого нет в venv
    if "inline" in os.environ.get("MPLBACKEND", ""):
        os.environ["MPLBACKEND"] = "Agg"
    setup_libero(args.libero_root)
    if args.preview:
        save_preview(args.suite, (args.tasks or [0])[0], args.preview)
        return
    policy = ScriptedPolicy if args.policy == "scripted" else OpenVLAPolicy(args.suite)
    evaluate(policy, args.suite, args.tasks, args.trials, args.out, save_video=not args.no_video,
             instruction=args.instruction, goal_on=tuple(args.goal_on) if args.goal_on else None,
             max_steps=args.max_steps)


if __name__ == "__main__":
    main()
