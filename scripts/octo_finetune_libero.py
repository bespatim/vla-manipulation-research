"""Дообучение Octo-Small 1.5 на демонстрациях LIBERO с выбранным набором входов (глава 3, шаг 2).

Основа — пример авторов Octo examples/02_finetune_new_observation_action.py: берём готовую модель,
меняем набор входов в её настройках, переносим все подходящие веса, а новые блоки (например, токенизатор
датчиков) учатся с нуля. Голова действий остаётся прежней: Octo-Small 1.5 уже выдаёт пачку из 4 действий
по 7 чисел — ровно как нужно для LIBERO.

Входы:
  внешняя камера и команда — всегда;
  --wrist   — камера на запястье (у Octo-Small 1.5 для неё уже есть свой токенизатор, 128×128);
  --proprio — датчики: положение и ориентация захвата + положение пальцев (8 чисел, `state` в данных).

Данные — 432 демонстрации LIBERO-Spatial в формате RLDS, те же, на которых обучались OpenVLA и
OpenVLA-OFT (`openvla/modified_libero_rlds`, папка libero_spatial_no_noops).

Пример:
    python scripts/octo_finetune_libero.py --data-dir /content/libero_rlds --wrist --proprio \
        --steps 5000 --batch-size 32 --save-dir /content/ckpt/octo_wrist_proprio
"""

import argparse
import json
import os
import time
from pathlib import Path


def libero_transform(trajectory):
    """Захват в данных LIBERO: −1 = открыт, +1 = закрыт → 1 = открыт, 0 = закрыт (как у OpenVLA и Octo)."""
    import tensorflow as tf

    gripper = 1 - tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
    trajectory["action"] = tf.concat([trajectory["action"][:, :6], gripper], axis=1)
    return trajectory


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="папка, внутри которой лежит libero_spatial_no_noops/1.0.0")
    p.add_argument("--dataset", default="libero_spatial_no_noops")
    p.add_argument("--pretrained", default="hf://rail-berkeley/octo-small-1.5",
                   help="откуда начинать: hf://... или octo_libero — Octo-Small, уже дообученная на LIBERO (шаг 1)")
    p.add_argument("--wrist", action="store_true", help="добавить камеру на запястье")
    p.add_argument("--proprio", action="store_true", help="добавить датчики (state)")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--shuffle", type=int, default=5000, help="буфер перемешивания кадров (ест оперативную память)")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--freeze-transformer", action="store_true", help="учить только новые блоки и голову")
    args = p.parse_args()

    os.environ["MPLBACKEND"] = "Agg"
    # Симулятор при обучении не нужен: JAX сразу занимает 75% видеопамяти одним куском (как по умолчанию).
    # При выделении по требованию пачка 32 со всеми входами не помещалась в T4 (нужно 7,6 ГБ одним куском),
    # а при 90% не хватало места cuDNN (CUDNN_STATUS_EXECUTION_FAILED)
    os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    import jax
    import numpy as np
    import optax
    import tensorflow as tf
    from octo.data.dataset import make_single_dataset
    from octo.model.components.tokenizers import LowdimObsTokenizer
    from octo.model.octo_model import OctoModel
    from octo.utils.jax_utils import initialize_compilation_cache
    from octo.utils.spec import ModuleSpec
    from octo.utils.train_utils import TrainState, freeze_weights, merge_params, process_text

    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")  # TensorFlow только читает данные

    if args.pretrained == "octo_libero":  # та же модель, что в шаге 1 (60 000 шагов на LIBERO, только камера)
        from huggingface_hub import snapshot_download

        repo = "cyrusneary/octo-finetuned-libero"
        sub = "2025-06-20_octo_small_1p5_libero_finetune/octo_finetune/experiment_20250620_175739"
        local = snapshot_download(repo_id=repo, allow_patterns=[f"{sub}/60000/*", f"{sub}/*.json", f"{sub}/*.msgpack"])
        pretrained = OctoModel.load_pretrained(os.path.join(local, sub), step=60000)
    else:
        pretrained = OctoModel.load_pretrained(args.pretrained)
    text_processor = pretrained.text_processor

    image_keys = {"primary": "image", **({"wrist": "wrist_image"} if args.wrist else {})}
    resize = {"primary": (256, 256), **({"wrist": (128, 128)} if args.wrist else {})}
    # Искажения картинок — как в настройках Octo для дообучения: обрезка и цвет для внешней камеры,
    # только цвет для камеры на запястье
    color = dict(random_brightness=[0.1], random_contrast=[0.9, 1.1], random_saturation=[0.9, 1.1],
                 random_hue=[0.05])
    augment = {"primary": dict(random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]), **color,
                               augment_order=["random_resized_crop", "random_brightness", "random_contrast",
                                              "random_saturation", "random_hue"]),
               "wrist": dict(**color, augment_order=["random_brightness", "random_contrast",
                                                     "random_saturation", "random_hue"])}
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name=args.dataset,
            data_dir=args.data_dir,
            image_obs_keys=image_keys,
            proprio_obs_key="state" if args.proprio else None,
            language_key="language_instruction",
            action_normalization_mask=[True] * 6 + [False],  # захват (0/1) не нормализуем
            standardize_fn=ModuleSpec.create(libero_transform),
        ),
        traj_transform_kwargs=dict(window_size=1, action_horizon=4, goal_relabeling_strategy=None,
                                   task_augment_strategy="delete_task_conditioning",
                                   task_augment_kwargs=dict(keep_image_prob=0.0)),  # задача — только текст
        frame_transform_kwargs=dict(resize_size=resize,
                                    image_augment_kwargs={k: v for k, v in augment.items() if k in resize}),
        train=True,
    )
    it = dataset.repeat().unbatch().shuffle(args.shuffle).batch(args.batch_size).iterator()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    it = map(process_batch, it)
    example_batch = next(it)

    # Набор входов: внешняя камера есть всегда; запястье — у Octo-Small 1.5 уже есть, убираем при ненадобности;
    # датчики — новый токенизатор (как в примере авторов для ALOHA)
    config = pretrained.config
    if not args.wrist:
        del config["model"]["observation_tokenizers"]["wrist"]
    if args.proprio:
        config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
            LowdimObsTokenizer, n_bins=256, bin_type="normal", low=-2.0, high=2.0, obs_keys=["proprio"])

    model = OctoModel.from_config(config, example_batch, text_processor, verbose=True,
                                  dataset_statistics=dataset.dataset_statistics)
    model = model.replace(params=merge_params(model.params, pretrained.params))
    del pretrained

    lr = optax.warmup_cosine_decay_schedule(0.0, args.lr, args.warmup, args.steps, end_value=0.0)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=0.01))
    frozen = list(model.config["optimizer"]["frozen_keys"])  # языковая модель T5 не дообучается
    if args.freeze_transformer:
        frozen.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen)
    state = TrainState.create(rng=jax.random.PRNGKey(1234), model=model, tx=tx)

    def loss_fn(params, batch, rng, train=True):
        bound = model.module.bind({"params": params}, rngs={"dropout": rng})
        emb = bound.octo_transformer(batch["observation"], batch["task"],
                                     batch["observation"]["timestep_pad_mask"], train=train)
        return bound.heads["action"].loss(emb, batch["action"], batch["observation"]["timestep_pad_mask"],
                                          batch["action_pad_mask"], train=train)

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, dropout_rng)
        return state.apply_gradients(grads=grads, rng=rng), info

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(model.config["model"]["observation_tokenizers"])
    (save_dir / "run.json").write_text(json.dumps({**vars(args), "inputs": inputs}, ensure_ascii=False, indent=1))
    print(f"Входы модели: {inputs} | шагов {args.steps}, пачка {args.batch_size}", flush=True)

    log, t0, losses = [], time.time(), []
    for i in range(1, args.steps + 1):
        state, info = train_step(state, next(it))
        losses.append(float(jax.device_get(info["loss"])) if i % 10 == 0 else None)
        if i % 100 == 0:
            recent = [x for x in losses[-100:] if x is not None]
            rate = i / (time.time() - t0)
            row = {"step": i, "loss": float(np.mean(recent)), "steps_per_s": rate,
                   "eta_min": (args.steps - i) / rate / 60}
            log.append(row)
            print(f"шаг {i}/{args.steps} | ошибка {row['loss']:.4f} | {rate:.2f} шаг/с | "
                  f"осталось ~{row['eta_min']:.0f} мин", flush=True)
            (save_dir / "train_log.json").write_text(json.dumps(log, indent=1))
        if i % args.save_every == 0 or i == args.steps:
            state.model.save_pretrained(step=i, checkpoint_path=str(save_dir))
    print(f"Готово: {save_dir}, шаг {args.steps}", flush=True)


if __name__ == "__main__":
    main()
