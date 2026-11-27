from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

OUT_DIR = Path("eval_lfw")


def main():
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} already exists. Use it for evaluation or rename it before exporting again.")

    lfw = fetch_lfw_people(
        min_faces_per_person=15, color=True, slice_=None, resize=1.0
    )
    print(f"Images: {len(lfw.images)} | People: {len(lfw.target_names)}")

    counts = Counter()
    with TemporaryDirectory(prefix=".lfw-", dir=OUT_DIR.parent) as temp:
        staging = Path(temp) / OUT_DIR.name
        staging.mkdir()

        for img, label in zip(lfw.images, lfw.target):
            name = lfw.target_names[label].replace(" ", "_")
            folder = staging / name
            folder.mkdir(exist_ok=True)
            counts[name] += 1
            path = folder / f"{counts[name]}.jpg"

            if img.max() <= 1.0:
                img = img * 255.0
            img_u8 = np.clip(img, 0, 255).astype("uint8")
            image_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), image_bgr):
                raise OSError(f"Could not write {path}")

        if OUT_DIR.exists():
            raise SystemExit(f"{OUT_DIR} already exists. Export cancelled.")
        staging.rename(OUT_DIR)

    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
