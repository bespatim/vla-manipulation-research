"""Octo в LIBERO — те же задачи, расстановки и лимиты шагов, что у OpenVLA (глава 1) и OpenVLA-OFT (глава 2).

Модели (--model):
  octo_libero — Octo-Small 1.5, дообученная на демонстрациях всех четырёх наборов LIBERO
                (`cyrusneary/octo-finetuned-libero`, папка octo_small, шаг 60000). Входы: внешняя камера
                и команда — как у OpenVLA в главе 1; камеры на запястье и датчиков нет.
  local       — своя дообученная модель (--checkpoint, --step): для шага 2 главы 3.

Подготовка кадров и действий — как в коде OpenVLA для LIBERO (moojink/openvla-oft,
experiments/robot/libero): кадр поворачивается на 180°, захват в действиях модели 1 = открыт, 0 = закрыт
и переводится в −1/+1 симулятора. Дополнительно на каждом шаге записываются сила контакта пальцев
с предметами, положение предметов, время модели и симулятора.

Пример:
    python scripts/libero_octo_eval.py --libero-root /content/LIBERO --suite libero_spatial \
        --tasks 0 1 2 --trials 3 --out results/octo_libero_spatial
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
NUM_STEPS_WAIT = 10
# Ключ статистики действий в чекпойнте (у libero_10 автор чекпойнта назвал набор «liber_o10»)
STATS_KEY = {"libero_spatial": "libero_spatial", "libero_object": "libero_object",
             "libero_goal": "libero_goal", "libero_10": "liber_o10"}
MODELS = {"octo_libero": ("cyrusneary/octo-finetuned-libero",
                          "2025-06-20_octo_small_1p5_libero_finetune/octo_finetune/experiment_20250620_175739",
                          60000)}
# Табл. 12 статьи OpenVLA: Octo, дообученная авторами OpenVLA на LIBERO (500 попыток на набор)
PAPER_SUCCESS = {"libero_spatial": 78.9}


# --------------------------------------------------------------------------------------
# Окружение
# --------------------------------------------------------------------------------------
def setup(libero_root: str) -> None:
    """Конфиг LIBERO без интерактивного вопроса, путь к LIBERO, видеопамять для JAX и симулятора."""
    os.environ["MPLBACKEND"] = "Agg"  # Colab передаёт дочерним процессам свой inline-бэкенд matplotlib
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX не забирает 75% видеопамяти сразу
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
    # TensorFlow нужен Octo только для обработки данных; иначе он займёт всю видеопамять
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")


def get_libero_env(task, resolution=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)
    return env, task.language


def agentview(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # как в демонстрациях: поворот на 180°


def wrist(obs):
    return np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])


def resize_like_octo(image, size):
    """Сжатие кадра, как в загрузчике данных Octo (tf.image.resize, lanczos3, со сглаживанием)."""
    import tensorflow as tf

    out = tf.image.resize(image, (size, size), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(out), 0, 255), tf.uint8).numpy()


def robot_state(obs):
    """Датчики, как в демонстрациях LIBERO (`state`): положение захвата, его поворот (ось-угол), пальцы."""
    q = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)  # x, y, z, w
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    axis_angle = np.zeros(3) if np.isclose(den, 0.0) else q[:3] * 2.0 * np.arccos(w) / den
    return np.concatenate([obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"]])


_FONT = None


def caption_desc(frame, desc, truth, frozen):
    """Подпись под кадром: что модель «думает» о захвате (зелёным — верно, красным — ошибается) и правда."""
    global _FONT
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont

    from octo_aux_head import GRIP_TEXT, HEIGHT_TEXT

    if _FONT is None:
        _FONT = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 13)
    img = Image.fromarray(frame)
    out = Image.new("RGB", (img.width, img.height + 44), (0, 0, 0))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    ok = desc == truth
    d.text((6, img.height + 4), f"модель думает: {GRIP_TEXT[desc[0]]}, {HEIGHT_TEXT[desc[1]]}",
           fill=(80, 220, 80) if ok else (255, 90, 90), font=_FONT)
    d.text((6, img.height + 23), f"датчики на самом деле: {GRIP_TEXT[truth[0]]}, {HEIGHT_TEXT[truth[1]]}"
           + ("   (модели подаются замороженные)" if frozen else ""), fill=(230, 230, 230), font=_FONT)
    return np.asarray(out)


def to_env_action(a):
    """Захват: модель 1 = открыт, 0 = закрыт → симулятор −1 = открыт, +1 = закрыт."""
    a = np.asarray(a, dtype=np.float64).copy()
    a[-1] = -np.sign(2 * a[-1] - 1) or -1.0
    return a


class ContactProbe:
    """Сила контакта пальцев захвата с предметами сцены — сигнал, которого модель не получает."""

    def __init__(self, env, objects):
        import mujoco

        self._mujoco = mujoco
        self.m, self.d = env.sim.model._model, env.sim.data._data
        name = lambda kind, i: mujoco.mj_id2name(self.m, kind, i) or ""
        self.gripper = {g for g in range(self.m.ngeom) if name(mujoco.mjtObj.mjOBJ_GEOM, g).startswith("gripper0_")}
        self.objects = list(objects)
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
            obj = self.owner.get(c.geom2 if g1 else c.geom1)
            if obj is None:
                other += abs(self._f6[0])
            else:
                forces[obj] += abs(self._f6[0])
            n += 1
        return {**{f"force_{o}": f for o, f in forces.items()}, "force_other": other, "n_contacts": n}


def make_goal_check(env, goal_on):
    """Своя цель «A лежит на B» через предикат LIBERO On(A, B) = B.check_ontop(A)."""
    if goal_on is None:
        return None
    obj, support = goal_on
    states = env.env.object_states_dict
    missing = [o for o in goal_on if o not in states]
    assert not missing, f"нет предметов {missing} в сцене; есть: {sorted(states)}"
    return lambda: bool(states[support].check_ontop(states[obj]))


# --------------------------------------------------------------------------------------
# Модель
# --------------------------------------------------------------------------------------
class OctoPolicy:
    def __init__(self, suite_name, model="octo_libero", checkpoint=None, step=None, exec_steps=None, seed=0):
        import jax
        from octo.model.octo_model import OctoModel

        if model == "local":
            path, self.checkpoint = checkpoint, checkpoint
        else:
            from huggingface_hub import snapshot_download

            repo, sub, step = MODELS[model]
            local = snapshot_download(repo_id=repo, allow_patterns=[f"{sub}/{step}/*", f"{sub}/*.json",
                                                                    f"{sub}/*.msgpack"])
            path, self.checkpoint = os.path.join(local, sub), f"{repo}/{sub}@{step}"
        self.model = OctoModel.load_pretrained(path, step=step)
        self.model_name = model
        stats = self.model.dataset_statistics
        if "action" not in stats:  # чекпойнт обучен на нескольких наборах — статистика по каждому
            stats = stats[STATS_KEY[suite_name]]
        self.action_stats = stats["action"]
        self.proprio_stats = stats.get("proprio")
        head = self.model.config["model"]["heads"]["action"]["kwargs"]
        self.horizon = head.get("action_horizon", 1)
        self.exec_steps = exec_steps or self.horizon
        self.inputs = sorted(self.model.config["model"]["observation_tokenizers"])
        # У octo_libero токенизатор запястья остался от Octo-Small 1.5, но на запястье её не обучали
        self.used = [k for k in self.inputs if not (model != "local" and k == "wrist")]
        self._jax = jax
        self._rng = jax.random.PRNGKey(seed)
        self._task_cache = {}
        # Упрощённый FuSe (scripts/octo_aux_head.py): если у модели есть голова описаний, на каждом запросе
        # записываем, что модель «думает» о захвате и его высоте. На действия это не влияет.
        self.has_aux = "aux" in self.model.config["model"]["heads"]
        self.last_desc = None

    def __call__(self, image, instruction, wrist_image=None, proprio=None):
        if instruction not in self._task_cache:
            self._task_cache[instruction] = self.model.create_tasks(texts=[instruction])
        obs = {"image_primary": image[None, None], "timestep_pad_mask": np.array([[True]])}
        if wrist_image is not None:  # при обучении кадры запястья сжимались до 128×128
            obs["image_wrist"] = resize_like_octo(wrist_image, 128)[None, None]
        if proprio is not None:  # при обучении датчики нормализовались по статистике демонстраций
            s = self.proprio_stats
            obs["proprio"] = ((proprio - np.asarray(s["mean"])) / (np.asarray(s["std"]) + 1e-8))[None, None]
            obs["proprio"] = obs["proprio"].astype(np.float32)
        if self.has_aux:
            emb = self.model.run_transformer(obs, self._task_cache[instruction], obs["timestep_pad_mask"],
                                             train=False)
            logits = self.model.module.apply({"params": self.model.params}, emb,
                                             method=lambda m, e: m.heads["aux"](e, train=False))
            lg = np.asarray(logits)[0, -1]
            self.last_desc = (int(lg[:4].argmax()), int(lg[4:].argmax()))
        self._rng, key = self._jax.random.split(self._rng)
        actions = self.model.sample_actions(obs, self._task_cache[instruction],
                                            unnormalization_statistics=self.action_stats, rng=key)
        return list(np.asarray(actions[0])[: self.exec_steps])


# --------------------------------------------------------------------------------------
# Эпизод
# --------------------------------------------------------------------------------------
def run_episode(policy, env, instruction, init_state, max_steps, objects,
                video_path=None, log_path=None, goal_on=None, use_env_goal=True):
    import imageio

    env.reset()
    obs = env.set_init_state(init_state)
    probe = ContactProbe(env, objects)
    goal_check = make_goal_check(env, goal_on)
    queue = deque()
    frozen_state = None  # для --freeze-proprio: показания датчиков с первого запроса («датчик завис»)
    rows, frames, success, t, step = [], [], False, 0, 0
    infer_total, sim_total, n_queries = 0.0, 0.0, 0
    desc = None  # (захват, высота) по мнению модели — обновляется на каждом запросе
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:  # ждём, пока предметы упадут на стол
            obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            t += 1
            continue
        query = len(queue) == 0
        infer_s = 0.0
        if query:
            state = robot_state(obs)
            if frozen_state is None:
                frozen_state = state.copy()
            if getattr(policy, "freeze_proprio", False):
                state = frozen_state.copy()
            t0 = time.time()
            queue.extend(policy(agentview(obs), instruction,
                                wrist_image=wrist(obs) if "wrist" in policy.used else None,
                                proprio=state if "proprio" in policy.used else None))
            infer_s = time.time() - t0
            infer_total += infer_s
            n_queries += 1
            desc = policy.last_desc if getattr(policy, "has_aux", False) else None
        raw = np.asarray(queue.popleft(), dtype=np.float64)
        action = to_env_action(raw)

        row = {"step": step, "query": int(query), "infer_s": round(infer_s, 4)}
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
        truth = None
        if desc is not None:
            from octo_aux_head import describe

            g, h = describe(robot_state(obs))  # правда — по живым датчикам, даже если модели подаются замороженные
            truth = (int(g), int(h))
            row.update({"model_grip": desc[0], "model_height": desc[1], "true_grip": truth[0], "true_height": truth[1]})
        if video_path is not None:
            frame = np.concatenate([agentview(obs), wrist(obs)], axis=1)
            if desc is not None:
                frame = caption_desc(frame, desc, truth, getattr(policy, "freeze_proprio", False))
            frames.append(frame)

        t0 = time.time()
        obs, _, done, _ = env.step(action.tolist())
        row["sim_s"] = round(time.time() - t0, 4)
        sim_total += row["sim_s"]
        rows.append(row)
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
        imageio.mimsave(video_path, frames, fps=20)  # 20 Гц — частота управления LIBERO
    described = [r for r in rows if "model_grip" in r]
    extra = {}
    if described:
        extra = {"desc_acc_grip": float(np.mean([r["model_grip"] == r["true_grip"] for r in described])),
                 "desc_acc_height": float(np.mean([r["model_height"] == r["true_height"] for r in described]))}
    return {**extra, "success": success, "steps": step, "queries": n_queries,
            "infer_s_per_query": infer_total / max(n_queries, 1),
            "infer_s_per_step": infer_total / max(step, 1), "sim_s_per_step": sim_total / max(step, 1)}


def evaluate(policy, suite_name, task_ids, trials_per_task, out_dir, save_video=True, instruction=None,
             goal_on=None):
    from libero.libero import benchmark

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    results = []
    for task_id in task_ids if task_ids is not None else range(suite.n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, task_instruction = get_libero_env(task)
        command = instruction or task_instruction
        objects = list(goal_on) if goal_on else []
        objects += [o for o in env.obj_of_interest if o not in objects]
        for trial in range(trials_per_task):
            name = f"{suite_name}_task{task_id:02d}_trial{trial:02d}"
            t0 = time.time()
            r = run_episode(policy, env, command, init_states[trial], MAX_STEPS[suite_name], objects,
                            video_path=out / f"{name}.mp4" if save_video else None, log_path=out / f"{name}.csv",
                            goal_on=goal_on, use_env_goal=instruction is None)
            r.update({"suite": suite_name, "model": policy.model_name, "task_id": task_id, "trial": trial,
                      "instruction": command, "objects": " ".join(objects), "wall_s": round(time.time() - t0, 1)})
            results.append(r)
            ok = sum(x["success"] for x in results)
            print(f"[{len(results)}] {command} | успех={r['success']} | шагов={r['steps']} | "
                  f"запросов к модели={r['queries']} ({r['infer_s_per_query']:.2f} с каждый) | "
                  f"{r['wall_s']} с | итого {ok}/{len(results)} ({100 * ok / len(results):.0f}%)", flush=True)
            with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(results[0]))
                w.writeheader()
                w.writerows(results)
        env.close()
    rate = 100 * sum(x["success"] for x in results) / len(results)
    paper = PAPER_SUCCESS.get(suite_name)
    print(f"\nуспех {rate:.1f}% на {len(results)} эпизодах"
          + (f" | Octo у авторов OpenVLA: {paper}%" if paper and instruction is None else ""))
    (out / "meta.json").write_text(json.dumps({
        "suite": suite_name, "model": policy.model_name, "checkpoint": policy.checkpoint,
        "inputs": policy.used, "action_horizon": policy.horizon, "exec_steps": policy.exec_steps,
        "episodes": len(results), "success_rate": rate, "paper_octo_success_rate": paper,
        "instruction": instruction, "goal_on": goal_on,
        "freeze_proprio": getattr(policy, "freeze_proprio", False),
        "mean_infer_s_per_query": float(np.mean([x["infer_s_per_query"] for x in results])),
        "mean_infer_s_per_step": float(np.mean([x["infer_s_per_step"] for x in results])),
        "mean_sim_s_per_step": float(np.mean([x["sim_s_per_step"] for x in results]))},
        ensure_ascii=False, indent=2))
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--libero-root", required=True)
    p.add_argument("--suite", default="libero_spatial", choices=list(MAX_STEPS))
    p.add_argument("--model", choices=[*MODELS, "local"], default="octo_libero")
    p.add_argument("--checkpoint", default=None, help="папка своей дообученной модели (для --model local)")
    p.add_argument("--step", type=int, default=None, help="шаг чекпойнта своей модели")
    p.add_argument("--exec-steps", type=int, default=None,
                   help="сколько действий из пачки выполнять до следующего запроса (по умолчанию — всю пачку)")
    p.add_argument("--tasks", type=int, nargs="*", default=None)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--instruction", default=None, help="своя команда роботу (по-английски)")
    p.add_argument("--goal-on", nargs=2, metavar=("A", "B"), default=None)
    p.add_argument("--freeze-proprio", action="store_true",
                   help="подавать датчики с первого шага попытки (проверка, пользуется ли ими модель)")
    args = p.parse_args()

    out = Path(args.out).resolve()
    setup(args.libero_root)
    policy = OctoPolicy(args.suite, args.model, args.checkpoint, args.step, args.exec_steps)
    policy.freeze_proprio = args.freeze_proprio
    print(f"Модель: {policy.checkpoint} | входы: {policy.used} | пачка {policy.horizon}, "
          f"выполняем {policy.exec_steps}", flush=True)
    evaluate(policy, args.suite, args.tasks, args.trials, out, save_video=not args.no_video,
             instruction=args.instruction, goal_on=tuple(args.goal_on) if args.goal_on else None)


if __name__ == "__main__":
    main()
