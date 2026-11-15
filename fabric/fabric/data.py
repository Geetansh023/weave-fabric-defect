"""Deterministic synthetic fixtures. These are NOT real fabric photographs."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["defect", "normal"]

def make_image(seed: int, defect: bool, size: int = 128) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    period = rng.uniform(3.5, 6.5)
    weave = 132 + 23*np.sin(x*2*np.pi/period) + 18*np.sin(y*2*np.pi/period)
    weave += rng.normal(0, 5, (size, size))
    tint = rng.uniform(.88, 1.12, 3)
    rgb = np.clip(weave[..., None]*tint, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    if defect:
        draw = ImageDraw.Draw(image)
        cx, cy = rng.integers(size//4, 3*size//4, 2)
        if seed % 2:
            draw.line([(int(cx-18), int(cy-26)), (int(cx+12), int(cy+26))], fill=(34, 30, 32), width=7)
        else:
            draw.ellipse((int(cx-14), int(cy-10), int(cx+14), int(cy+10)), fill=(36, 32, 35))
    return image

def generate(root: str, count: int = 64, seed: int = 42):
    """Independent samples per split. Existing output is refused, not mixed."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"{root} is not empty. Choose a new directory.")
    if count < 4:
        raise ValueError("count must be at least 4 per class")
    rng = np.random.default_rng(seed)
    for split, n in [("train", count), ("val", max(4, count//4)), ("test", max(4, count//4))]:
        for label in CLASSES:
            dest = root / split / label
            dest.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                sample_seed = int(rng.integers(0, 2**32-1))
                make_image(sample_seed, label == "defect").save(dest / f"{i:04d}.png")
    (root / "SYNTHETIC.txt").write_text("Procedural illustrations only. Not evidence of real-world performance.\n")
