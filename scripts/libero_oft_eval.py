"""OpenVLA-OFT в LIBERO: какие входы модель на самом деле использует.

Модель авторов (`moojink/openvla-7b-oft-finetuned-libero-*`) обучена сразу с двумя камерами
(внешняя + запястье) и состоянием робота (позиция и ориентация захвата + положение двух пальцев).
Других версий авторы не выкладывали, поэтому вклад каждого входа проверяется «заморозкой» на тесте:
вход подаётся, но всё время один и тот же — тот, что был на первом шаге эпизода. Картинка/числа
остаются правдоподобными, а информации о происходящем в них нет.

Условия:
  full        — 2 камеры + состояние робота (как у авторов)
  no_proprio  — 2 камеры, состояние заморожено          («2 фото без датчиков»)
  no_wrist    — камера запястья заморожена + состояние  («1 фото + датчики»)
  image_only  — заморожены и запястье, и состояние      («1 фото», как вход OpenVLA из главы 1)

Цикл эпизода, подготовка кадров и обработка действий — функции из репозитория авторов
(moojink/openvla-oft, experiments/robot/libero/run_libero_eval.py). Дополнительно на каждом шаге
записываются сила контакта пальцев с предметами, положение предметов, время модели и симулятора.

Используется из notebooks/libero_oft_colab.ipynb:
    python scripts/libero_oft_eval.py --libero-root /content/LIBERO --oft-root /content/openvla-oft \
        --suite libero_spatial --condition full --tasks 0 1 2 --trials 3 --out results/oft/full
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
CHECKPOINTS = {
    "libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "libero_object": "moojink/openvla-7b-oft-finetuned-libero-object",
    "libero_goal": "moojink/openvla-7b-oft-finetuned-libero-goal",
    "libero_10": "moojink/openvla-7b-oft-finetuned-libero-10",
}
# Табл. I статьи OpenVLA-OFT (500 попыток на набор, 2 камеры + состояние робота)
PAPER_SUCCESS = {"libero_spatial": 97.6, "libero_object": 98.4, "libero_goal": 97.9, "libero_10": 94.5}
CONDITIONS = {
    #             заморозить запястье, заморозить состояние
    "full": (False, False),
    "no_proprio": (False, True),
    "no_wrist": (True, False),
    "image_only": (True, True),
}


# --------------------------------------------------------------------------------------
# Окружение
# --------------------------------------------------------------------------------------
def setup(libero_root: str, oft_root: str) -> None:
    """Конфиг LIBERO без интерактивного вопроса, пути к LIBERO и коду OFT, TF без видеокарты."""
    # Colab передаёт дочерним процессам свой inline-бэкенд matplotlib, которого нет в venv
    if "inline" in os.environ.get("MPLBACKEND", ""):
        os.environ["MPLBACKEND"] = "Agg"
    root = Path(libero_root).resolve()
    bench = root / "libero" / "libero"
    cfg_dir = root / ".libero_config"
    cfg_dir.mkdir(exist_ok=True)
    paths = {"benchmark_root": bench, "bddl_files": bench / "bddl_files", "init_states": bench / "init_files",
             "datasets": root / "datasets", "assets": bench / "assets"}
    (cfg_dir / "config.yaml").write_text("".join(f"{k}: {v}\n" for k, v in paths.items()))
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)
    for p in (str(root), str(Path(oft_root).resolve())):
        if p not in sys.path:
            sys.path.insert(0, p)
    # TensorFlow нужен коду авторов только для обработки картинок; иначе он займёт всю видеопамять
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")


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
class OFTPolicy:
    """OpenVLA-OFT, загруженная так же, как в run_libero_eval.initialize_model авторов."""

    def __init__(self, suite_name: str):
        import torch
        from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key
        from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector
        from experiments.robot.robot_utils import get_image_resize_size, get_model, set_seed_everywhere

        self.cfg = GenerateConfig(pretrained_checkpoint=CHECKPOINTS[suite_name], task_suite_name=suite_name,
                                  use_l1_regression=True, use_diffusion=False, use_film=False,
                                  num_images_in_input=2, use_proprio=True, center_crop=True,
                                  num_open_loop_steps=8, lora_rank=32)
        set_seed_everywhere(self.cfg.seed)
        self.model = get_model(self.cfg)
        self.proprio_projector = get_proprio_projector(self.cfg, self.model.llm_dim, proprio_dim=8)
        self.action_head = get_action_head(self.cfg, self.model.llm_dim)
        self.processor = get_processor(self.cfg)
        check_unnorm_key(self.cfg, self.model)
        self.resize_size = get_image_resize_size(self.cfg)
        self.checkpoint = self.cfg.pretrained_checkpoint
        print(f"OpenVLA-OFT: {self.checkpoint} | unnorm_key={self.cfg.unnorm_key} | "
              f"видеопамять {torch.cuda.memory_allocated() / 1e9:.1f} ГБ")

    def __call__(self, observation: dict, instruction: str) -> list:
        from experiments.robot.robot_utils import get_action

        return get_action(self.cfg, self.model, observation, instruction, processor=self.processor,
                          action_head=self.action_head, proprio_projector=self.proprio_projector,
                          noisy_action_projector=None, use_film=False)


# --------------------------------------------------------------------------------------
# Эпизод и оценка
# --------------------------------------------------------------------------------------
def run_episode(policy, env, instruction, init_state, max_steps, objects, condition,
                video_path=None, log_path=None, goal_on=None, use_env_goal=True):
    import imageio
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_image, \
        get_libero_wrist_image, quat2axisangle
    from experiments.robot.libero.run_libero_eval import process_action
    from experiments.robot.openvla_utils import resize_image_for_policy

    freeze_wrist, freeze_proprio = CONDITIONS[condition]
    env.reset()
    obs = env.set_init_state(init_state)
    probe = ContactProbe(env, objects)
    goal_check = make_goal_check(env, goal_on)
    queue = deque(maxlen=policy.cfg.num_open_loop_steps)
    frozen_wrist = frozen_state = None
    rows, frames, success, t, step = [], [], False, 0, 0
    infer_total, sim_total, n_queries = 0.0, 0.0, 0
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue
        query = len(queue) == 0
        infer_s = 0.0
        if query:
            wrist = resize_image_for_policy(get_libero_wrist_image(obs), policy.resize_size)
            state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"]))
            if frozen_wrist is None:
                frozen_wrist, frozen_state = wrist.copy(), state.copy()
            observation = {
                "full_image": resize_image_for_policy(get_libero_image(obs), policy.resize_size),
                "wrist_image": frozen_wrist.copy() if freeze_wrist else wrist,
                "state": frozen_state.copy() if freeze_proprio else state,
            }
            t0 = time.time()
            queue.extend(policy(observation, instruction))
            infer_s = time.time() - t0
            infer_total += infer_s
            n_queries += 1
        raw = np.asarray(queue.popleft(), dtype=np.float64)
        action = process_action(raw.copy(), "openvla")

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
        if video_path is not None:
            frames.append(np.concatenate([get_libero_image(obs), get_libero_wrist_image(obs)], axis=1))

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
    return {"success": success, "steps": step, "queries": n_queries,
            "infer_s_per_query": infer_total / max(n_queries, 1),
            "infer_s_per_step": infer_total / max(step, 1), "sim_s_per_step": sim_total / max(step, 1)}


def evaluate(policy, suite_name, condition, task_ids, trials_per_task, out_dir, save_video=True,
             instruction=None, goal_on=None):
    from experiments.robot.libero.libero_utils import get_libero_env
    from libero.libero import benchmark

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    results = []
    for task_id in task_ids if task_ids is not None else range(suite.n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, task_instruction = get_libero_env(task, "openvla", resolution=256)
        command = instruction or task_instruction
        objects = list(goal_on) if goal_on else []
        objects += [o for o in env.obj_of_interest if o not in objects]
        for trial in range(trials_per_task):
            name = f"{suite_name}_task{task_id:02d}_trial{trial:02d}"
            t0 = time.time()
            r = run_episode(policy, env, command, init_states[trial], MAX_STEPS[suite_name], objects, condition,
                            video_path=out / f"{name}.mp4" if save_video else None, log_path=out / f"{name}.csv",
                            goal_on=goal_on, use_env_goal=instruction is None)
            r.update({"suite": suite_name, "condition": condition, "task_id": task_id, "trial": trial,
                      "instruction": command, "objects": " ".join(objects), "wall_s": round(time.time() - t0, 1)})
            results.append(r)
            ok = sum(x["success"] for x in results)
            print(f"[{condition} {len(results)}] {command} | успех={r['success']} | шагов={r['steps']} | "
                  f"запросов к модели={r['queries']} ({r['infer_s_per_query']:.2f} с каждый) | "
                  f"{r['wall_s']} с | итого {ok}/{len(results)} ({100 * ok / len(results):.0f}%)", flush=True)
            with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(results[0]))
                w.writeheader()
                w.writerows(results)
        env.close()
    rate = 100 * sum(x["success"] for x in results) / len(results)
    print(f"\n{condition}: успех {rate:.1f}% на {len(results)} эпизодах"
          + (f" | у авторов (full): {PAPER_SUCCESS[suite_name]}%" if instruction is None else ""))
    (out / "meta.json").write_text(json.dumps({
        "suite": suite_name, "condition": condition, "freeze_wrist": CONDITIONS[condition][0],
        "freeze_proprio": CONDITIONS[condition][1], "episodes": len(results), "success_rate": rate,
        "paper_success_rate_full": PAPER_SUCCESS.get(suite_name), "instruction": instruction, "goal_on": goal_on,
        "checkpoint": policy.checkpoint, "seed": policy.cfg.seed,
        "mean_infer_s_per_query": float(np.mean([x["infer_s_per_query"] for x in results])),
        "mean_infer_s_per_step": float(np.mean([x["infer_s_per_step"] for x in results])),
        "mean_sim_s_per_step": float(np.mean([x["sim_s_per_step"] for x in results]))},
        ensure_ascii=False, indent=2))
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--libero-root", required=True)
    p.add_argument("--oft-root", required=True)
    p.add_argument("--suite", default="libero_spatial", choices=list(CHECKPOINTS))
    p.add_argument("--conditions", nargs="+", default=["full"], choices=list(CONDITIONS))
    p.add_argument("--tasks", type=int, nargs="*", default=None)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--out", required=True, help="папка; внутри по подпапке на условие")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--instruction", default=None, help="своя команда роботу (по-английски)")
    p.add_argument("--goal-on", nargs=2, metavar=("A", "B"), default=None)
    args = p.parse_args()

    setup(args.libero_root, args.oft_root)
    policy = OFTPolicy(args.suite)  # модель грузится один раз на все условия
    for condition in args.conditions:
        evaluate(policy, args.suite, condition, args.tasks, args.trials, Path(args.out) / condition,
                 save_video=not args.no_video, instruction=args.instruction,
                 goal_on=tuple(args.goal_on) if args.goal_on else None)


if __name__ == "__main__":
    main()
