# ◈ Weave
### Fabric inspection, with the evidence in view.

A fresh prototype implementation of a fabric-defect classifier with a ResNet18 model, Grad-CAM visual explanations, and a Streamlit interface.

![Weave synthetic demonstration](docs/demo-desktop.png)

> **Status: synthetic demonstration, not a validated fabric inspection system.**
> This project was written from scratch in November 2026. It does not reconstruct an earlier project, recover prior experiments, or substantiate historical CV claims. No real fabric dataset was supplied. The screenshot shows a model trained on generated patterns. The displayed scores are uncalibrated model outputs, not real-world accuracy.

## What works

- Two-class ResNet18 training: `defect` and `normal`.
- Separate train, validation and test splits. Lowest validation loss selects the checkpoint; the test split is evaluated separately.
- Optional ImageNet initialization for real-data experiments. The included demonstration used random initialization and no downloaded weights.
- Grad-CAM on the final residual block, with selectable explanation class and overlay strength.
- Image upload, generated samples, PNG overlay download and prediction JSON export.
- CLI generation, training, evaluation and prediction.
- CPU tests and a GitHub Actions test workflow, ready to use if a repository is later approved.

## Start locally

Tested on Linux with Python 3.10 and CPU PyTorch. Python 3.10+ is recommended. Windows and macOS have not been tested. Allow roughly 1 GB or more for the Python environment and model dependencies.

Create an environment in the unzipped project directory:

```sh
python -m venv .venv
```

Activate it:

```sh
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell (instead of the line above)
.venv\Scripts\Activate.ps1
```

For CPU-only PyTorch, install it first:

```sh
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

Generate the synthetic data, train the demonstration model, and open the app:

```sh
python -m fabric.cli generate --output data/synthetic --count 32
python -m fabric.cli train --data data/synthetic --epochs 4 --output artifacts/demo.pt
python -m streamlit run app.py
```

Streamlit prints a local browser address. Open that address on the same computer. Training time depends on your hardware; the review run took only a short CPU run, but this is not a timing guarantee.

**The ZIP intentionally contains no trained checkpoint or real dataset.** Run the two generation/training commands above to reproduce the demo. Without a checkpoint, the app still opens and shows an explicit "Not loaded" state rather than inventing a prediction. Generated data and weights are excluded from Git by default.

The generator refuses an existing nonempty output folder, and training refuses to overwrite an existing checkpoint. To rerun, choose new paths or deliberately remove your own generated files first. Then update the checkpoint path in the app sidebar if you used a different path.

## Evaluate and explain

```sh
python -m fabric.cli evaluate --data data/synthetic/test --checkpoint artifacts/demo.pt --output artifacts/evaluation.json
python -m fabric.cli predict --image data/synthetic/test/defect/0000.png --checkpoint artifacts/demo.pt --output artifacts/gradcam.png
python -m pytest -q
```

Evaluation reports accuracy, defect precision, defect recall and a confusion matrix. Its rows are true classes; columns are predicted classes, in the order `[defect, normal]`. A zero precision or recall denominator is reported as zero. The test data is never used by the training command to select a checkpoint.

## Use a real dataset

Obtain a dataset you are allowed to use and keep its license and source records. Arrange it as:

```text
data/real/
  train/
    defect/
    normal/
  val/
    defect/
    normal/
  test/
    defect/
    normal/
```

```sh
python -m fabric.cli train --data data/real --output artifacts/real.pt --epochs 10 --size 224 --pretrained
python -m fabric.cli evaluate --data data/real/test --checkpoint artifacts/real.pt --output artifacts/real-test.json
```

`--pretrained` downloads torchvision's ImageNet ResNet18 weights. It requires internet access. No pretrained weights are redistributed in this package. Without that flag, weights start randomly.

Before training:

1. Split by fabric roll, acquisition session, source photograph or production batch, not by near-identical image crops. The code cannot infer these groups for you.
2. Keep near-duplicates out of different splits. The fixture tests check exact uniqueness of generated samples; this does not prove real-data independence.
3. Keep a genuinely held-out test set and do not tune against it. Include realistic lighting, weave types, scales and hard negative samples.
4. Check class balance, defect labels and the cost of missed defects. Choose any operating threshold on validation data. This prototype currently uses the largest of two softmax scores, not a tuned industrial threshold.

Images are resized to a square and normalized with ImageNet channel statistics. This can distort shape and lose tiny defects. Resizing, full-frame classification and the coarse feature map are intentional prototype limits. Patch-based inspection or segmentation may be better for small flaws, but neither is implemented here.

## Read the heatmap correctly

Grad-CAM visualizes regions that affect a selected class. It is **not** a defect segmentation mask, a causal explanation or a guarantee that the model found the physical flaw. A map can be blank when positive class-sensitive activations are absent. Texture, lighting and background can influence predictions. Compare maps against annotated real data before drawing conclusions.

Scores are softmax outputs. They are not calibrated probabilities, test accuracy, defect severity or quality-control certification. A synthetic-only model should not be used to inspect real fabric.

## Review evidence

- `docs/test-results.txt`: actual unit-test output.
- `docs/synthetic-training-run.json`: the recorded four-epoch synthetic run, seed 42.
- `docs/synthetic-test-run.json`: held-out generated-fixture evaluation from that run. Strong results on these easy procedural patterns say nothing about performance on real fabric.
- `docs/tested-environment.txt`: the exact environment used for review, including CPU package versions. It is an audit record, not a cross-platform lockfile.
- `docs/demo-desktop.png`, `docs/demo-intact.png`, `docs/demo-mobile.png`: screenshots of the running app.
- `docs/verification.md`: interactions and checks performed.

No real-world metrics, dataset benchmark claims or historical training results are included. The next useful step is to obtain a licensed real fabric dataset, train with an independent split, and review failure cases.

## Layout

```text
app.py                  Streamlit interface
fabric/data.py          Synthetic sample generator
fabric/model.py         ResNet18, preprocessing, checkpoints, Grad-CAM
fabric/cli.py           Train, evaluate and predict commands
tests/test_core.py      Unit tests
.streamlit/config.toml  Theme and upload limit
.github/workflows/      CPU test workflow
docs/                   Review evidence and screenshots
```

## Privacy and checkpoint safety

Inference happens in the Python process running the app. This code does not call an external image-prediction API. If you deploy it on a server, uploads will reach that server, so review hosting access, retention and privacy separately. The screenshot is a local demonstration, not a hosted service.

Checkpoints load with PyTorch `weights_only=True` and a fixed expected architecture/class mapping. Only load checkpoints you trust anyway. Model files can consume significant memory and should not be accepted from arbitrary visitors. The app accepts a local checkpoint path, not an uploaded checkpoint.

The upload limit is 15 MB and images larger than 25 megapixels are rejected by the app. Very large, corrupt or unusual images may still fail. This is a prototype, not a hardened multi-user service.

## Technical references

- [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [PyTorch: saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Streamlit image display](https://docs.streamlit.io/develop/api-reference/media/st.image)

## Release status

Published as a prototype with the owner's approval on 15 November 2026. Synthetic demonstration only; real-data testing is pending. A project-wide open-source license has not yet been selected; see `LICENSE-NOTICE.md`. Review dataset rights, claims and licensing before reuse or deployment.
