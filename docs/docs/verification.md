# Verification record

Review date: 15 November 2026.

## Executed, not inferred

- Python 3.10, Linux, CPU PyTorch. Exact package versions are in `tested-environment.txt`.
- `python -m pytest -q`: **7 passed**.
- Generated 64 training images, 16 validation images and 16 test images, balanced between `defect` and `normal`. These are procedural illustrations, not photographs.
- Trained ResNet18 from random initialization for four epochs, seed 42, image size 128, batch size 16, Adam learning rate 0.001.
- Selected the lowest-validation-loss checkpoint, which was epoch 4.
- Evaluated that checkpoint once on the generated test split. All 16 generated test fixtures were classified correctly in this small run. This does not estimate real-world accuracy and is not a fabric benchmark.
- Ran CLI prediction and saved a Grad-CAM overlay.

## Unit-test coverage

1. Reproducible synthetic images and different defect/intact fixtures.
2. Balanced generated splits, exact-file uniqueness across splits and refusal to overwrite data.
3. ResNet18 input shape, probability normalization, finite normalized CAM and overlay dimensions.
4. Checkpoint round trip and rejection of reversed class mappings.
5. Rejection of unexpected dataset class folders.
6. Grayscale image conversion to RGB.
7. Evaluation sample count and confusion-matrix accounting.

## Running-app checks

Performed in Chromium against the running Streamlit app:

- Desktop synthetic-defect view renders with no Streamlit exception.
- Switching to the synthetic intact sample produces the `normal` model response in this demonstration.
- Switching the explanation target to `normal` updates the Grad-CAM target caption.
- Overlay download produces a valid 128 × 128 PNG.
- A missing checkpoint produces `Not loaded` and no prediction-download control.
- Restoring the local checkpoint enables inference again.
- The upload branch accepts a generated normal PNG file and runs without a Streamlit exception. No real fabric photo was tested.
- At a 390-pixel mobile viewport, there is no horizontal overflow; the three desktop columns stack vertically.

The desktop screenshot and mobile screenshots were visually inspected. The dark background, mint accents, readable warnings, image panels and model response are visible without overlap in the inspected views. Mobile screenshots capture the top and middle of the page, not every lower control. The lower controls were tested on desktop.

## Not verified

- Real fabric performance, calibration, robustness, defect localization accuracy or production suitability.
- Windows or macOS installation; GPU training.
- A hosted deployment or public GitHub repository.
- Dataset rights or an open-source license choice for a future release.
- The absence of near-duplicate real samples across splits. Real-data grouping remains the user's responsibility.

## Package limits

The review ZIP includes source code, docs, synthetic-run logs and screenshots. It does not include the generated dataset, the roughly 45 MB demonstration checkpoint, or an environment directory. The README gives commands to regenerate the synthetic data and train the demonstration checkpoint locally.
