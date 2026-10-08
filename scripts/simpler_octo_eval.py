"""Готовая Octo (без дообучения) в симуляторе SimplerEnv: задачи робота WidowX из BridgeData V2.

Модель получает только картинку внешней камеры и команду — как OpenVLA в главе 1. В SimplerEnv для
каждой задачи заданы 24 фиксированные расстановки предметов (episode_id 0–23), поэтому попытки
можно повторить один в один.

Пример:
    python simpler_octo_eval.py --model octo-small --tasks widowx_carrot_on_plate --episodes 0 1 2 --out results/octo_simpler
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

TASKS = ["widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]


def scalar_info(info):
    """Числа и флаги из info симулятора (успех, захвачен ли предмет, сдвинут ли не тот предмет и т.п.)."""
    out = {}
    for k, v in info.items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v)
    return out


def run_episode(env, policy, episode_id, instruction=None, video_path=None):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    obs, _ = env.reset(options={"obj_init_options": {"episode_id": episode_id}})
    task_instruction = env.get_language_instruction()
    instruction = instruction or task_instruction
    policy.reset(instruction)

    frames, steps = [], []
    truncated, info, ever_success = False, {}, False
    t_start = time.time()
    while not truncated:
        image = get_image_from_maniskill2_obs_dict(env, obs)
        frames.append(image)
        t0 = time.time()
        _, action = policy.step(image, instruction)
        infer_s = time.time() - t0
        a = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        obs, _, done, truncated, info = env.step(a)
        ever_success |= bool(info.get("success", False))
        eef = obs.get("agent", {}).get("eef_pos")
        steps.append({"step": len(steps), "infer_s": infer_s,
                      **{f"a{i}": float(x) for i, x in enumerate(a)},
                      **({f"eef{i}": float(x) for i, x in enumerate(np.ravel(eef))} if eef is not None else {}),
                      **scalar_info(info)})
    frames.append(get_image_from_maniskill2_obs_dict(env, obs))

    if video_path:
        import imageio
        imageio.mimsave(video_path, frames, fps=5)
    # Как в SimplerEnv: успех — состояние сцены на последнем шаге
    return {"success": bool(info.get("success", False)), "ever_success": ever_success,
            "steps": len(steps), "episode_s": time.time() - t_start,
            "infer_s": float(np.mean([s["infer_s"] for s in steps])),
            "instruction": instruction, "task_instruction": task_instruction,
            **{f"final_{k}": v for k, v in scalar_info(info).items()}}, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="octo-small", choices=["octo-small", "octo-base"])
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    p.add_argument("--episodes", nargs="+", type=int, default=list(range(24)))
    p.add_argument("--instruction", default=None, help="своя команда вместо команды задачи")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.environ["MPLBACKEND"] = "Agg"
    # JAX по умолчанию забирает 75% памяти видеокарты, TensorFlow — всю; симулятору тоже нужна видеокарта
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import simpler_env
    from simpler_env.policies.octo.octo_model import OctoInference

    policy = OctoInference(model_type=args.model, policy_setup="widowx_bridge", init_rng=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=1), encoding="utf-8")

    summary_path = out / "summary.csv"
    for task in args.tasks:
        env = simpler_env.make(task)
        for ep in args.episodes:
            name = f"{task}_ep{ep:02d}"
            res, steps = run_episode(env, policy, ep, args.instruction, out / f"{name}.mp4")
            with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for s in steps for k in s)))
                w.writeheader()
                w.writerows(steps)
            row = {"model": args.model, "task": task, "episode": ep, **res}
            new = not summary_path.exists()
            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"{name}: {'УСПЕХ' if res['success'] else 'неудача'} | шагов {res['steps']} | "
                  f"модель {res['infer_s']:.2f} с/шаг | «{res['instruction']}»", flush=True)
        env.close()


if __name__ == "__main__":
    main()
