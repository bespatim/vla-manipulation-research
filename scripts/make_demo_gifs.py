"""GIF демонстраций LIBERO-Spatial — так выглядят примеры, на которых учится модель (глава 3).

На каждом кадре: слева внешняя камера, справа камера на запястье; сверху команда; снизу показания
датчиков (`state`: положение захвата x, y, z, его поворот, раскрытие пальцев) и действие человека
на этом шаге (сдвиг и поворот захвата, открыть/закрыть пальцы) — ровно то, что видит модель при обучении.

Пример (в Colab, после блока 4 блокнота libero_octo_finetune_colab.ipynb):
    python scripts/make_demo_gifs.py --data-dir /content/libero_rlds --count 2 --out /content/demo_gifs
"""
import argparse
import os
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--dataset", default="libero_spatial_no_noops")
    p.add_argument("--count", type=int, default=2, help="сколько демонстраций (с разными командами)")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.environ["MPLBACKEND"] = "Agg"  # Colab передаёт дочерним процессам свой inline-бэкенд
    import matplotlib
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from PIL import Image, ImageDraw, ImageFont

    tf.config.set_visible_devices([], "GPU")
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 11)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    builder = tfds.builder_from_directory(str(Path(args.data_dir) / args.dataset / "1.0.0"))
    seen = set()
    for ep in tfds.as_numpy(builder.as_dataset(split="train")):
        steps = list(ep["steps"])
        command = steps[0]["language_instruction"].decode()
        if command in seen:
            continue
        seen.add(command)
        frames = []
        for k, s in enumerate(steps[:: args.stride]):
            o = s["observation"]
            img = Image.fromarray(np.concatenate([o["image"], o["wrist_image"]], axis=1))
            canvas = Image.new("RGB", (img.width, img.height + 52), (0, 0, 0))
            canvas.paste(img, (0, 18))
            d = ImageDraw.Draw(canvas)
            st, a = o["state"], s["action"]
            d.text((4, 3), command, fill=(255, 255, 0), font=font)
            d.text((4, img.height + 21),
                   f"датчики: захват x={st[0]:+.2f} y={st[1]:+.2f} z={st[2]:+.2f}  "
                   f"пальцы {st[6]:.3f} {st[7]:.3f}", fill=(255, 255, 255), font=font)
            d.text((4, img.height + 36),
                   f"действие: dx={a[0]:+.2f} dy={a[1]:+.2f} dz={a[2]:+.2f}  "
                   f"захват {'закрыть' if a[6] > 0 else 'открыть'}   шаг {k * args.stride}",
                   fill=(255, 255, 255), font=font)
            frames.append(canvas)
        pal = frames[len(frames) // 2].quantize(colors=128)
        q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        path = out / f"demo_{len(seen) - 1:02d}.gif"
        q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 * args.stride / 20), loop=0,
                  optimize=True)
        print(path, len(steps), "шагов |", command, flush=True)
        if len(seen) >= args.count:
            break


if __name__ == "__main__":
    main()
