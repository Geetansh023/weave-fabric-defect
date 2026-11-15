"""Train only on train; select checkpoints on val; evaluate test separately."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from .data import CLASSES, generate
from .model import build_model, transform, load_checkpoint, open_image, GradCAM, overlay

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

def dataset(root, size):
    ds = ImageFolder(root, transform=transform(size))
    if ds.classes != CLASSES:
        raise ValueError(f"Expected exactly {CLASSES}, got {ds.classes}")
    return ds

@torch.no_grad()
def score(model, loader):
    model.eval()
    matrix = torch.zeros((2, 2), dtype=torch.int64)
    loss = 0.
    for x, y in loader:
        logits = model(x)
        loss += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        for truth, pred in zip(y, logits.argmax(1)):
            matrix[truth, pred] += 1
    n = int(matrix.sum())
    tp, fn = int(matrix[0, 0]), int(matrix[0, 1])
    fp = int(matrix[1, 0])
    return {"samples": n, "loss": loss/n, "accuracy": float(matrix.diag().sum()/n),
            "defect_precision": tp/(tp+fp) if tp+fp else 0.,
            "defect_recall": tp/(tp+fn) if tp+fn else 0.,
            "confusion_matrix": matrix.tolist(), "matrix_order": CLASSES}

def train(args):
    seed_all(args.seed)
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    root = Path(args.data)
    out = Path(args.output)
    if out.exists():
        raise ValueError("Output checkpoint exists. Choose another path to avoid overwriting work.")
    train_ds, val_ds = dataset(root/"train", args.size), dataset(root/"val", args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    model = build_model(args.pretrained)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss, history = float("inf"), []
    provenance = "synthetic" if (root/"SYNTHETIC.txt").exists() else "user-supplied (not independently verified)"
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(y)
        val = score(model, val_loader)
        row = {"epoch": epoch+1, "train_loss": total_loss/len(train_ds), "val": val}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save({"format_version": 1, "state_dict": model.state_dict(), "classes": CLASSES,
                        "image_size": args.size, "data_kind": provenance, "seed": args.seed,
                        "initialization": "ImageNet" if args.pretrained else "random",
                        "selected_epoch": epoch+1, "val_metrics": val}, out)
    report = {"data_kind": provenance, "selection": "lowest validation loss", "train_samples": len(train_ds),
              "val_samples": len(val_ds), "history": history}
    out.with_suffix(".training.json").write_text(json.dumps(report, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Create synthetic demonstration fixtures")
    gen.add_argument("--output", default="data/synthetic")
    gen.add_argument("--count", type=int, default=64)
    gen.add_argument("--seed", type=int, default=42)
    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True)
    tr.add_argument("--output", default="artifacts/demo.pt")
    tr.add_argument("--epochs", type=int, default=4)
    tr.add_argument("--size", type=int, default=128)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=.001)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--pretrained", action="store_true", help="Download ImageNet weights; requires network")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--data", required=True, help="Held-out split, e.g. data/real/test")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--output", default="artifacts/evaluation.json")
    pr = sub.add_parser("predict")
    pr.add_argument("--image", required=True)
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--output", default="artifacts/gradcam.png")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.output, args.count, args.seed)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        seed_all(42)
        model, meta = load_checkpoint(args.checkpoint)
        result = score(model, DataLoader(dataset(args.data, meta["image_size"]), batch_size=16))
        result.update({"checkpoint_data_kind": meta["data_kind"], "evaluated_split": str(args.data),
                       "synthetic_split": (Path(args.data).parent/"SYNTHETIC.txt").exists()})
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    else:
        seed_all(42)
        model, meta = load_checkpoint(args.checkpoint)
        image = open_image(args.image)
        cam = GradCAM(model)
        try:
            probabilities, heat, target = cam(transform(meta["image_size"])(image).unsqueeze(0))
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            overlay(image, heat).save(args.output)
            print(json.dumps({"predicted_class": CLASSES[target], "softmax_scores": probabilities.tolist(),
                              "data_kind": meta["data_kind"], "note": "Scores are uncalibrated; CAM is not a defect mask."}))
        finally:
            cam.close()

if __name__ == "__main__":
    main()
