import hashlib
import numpy as np
import pytest
import torch
from PIL import Image
from fabric.data import generate, make_image
from fabric.model import build_model, transform, GradCAM, overlay, load_checkpoint, open_image
from fabric.cli import dataset, score
from torch.utils.data import DataLoader

torch.set_num_threads(2)

def test_fixture_reproducible():
    assert np.array_equal(make_image(4, True), make_image(4, True))
    assert not np.array_equal(make_image(4, True), make_image(4, False))

def test_generated_splits_unique_and_balanced(tmp_path):
    generate(tmp_path, 4)
    hashes = []
    for split in ['train', 'val', 'test']:
        for label in ['defect', 'normal']:
            files = list((tmp_path/split/label).glob('*.png'))
            assert len(files) == 4
            hashes += [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    assert len(set(hashes)) == len(hashes)
    with pytest.raises(ValueError):
        generate(tmp_path, 4)

def test_model_transform_and_gradcam():
    torch.manual_seed(42)
    model = build_model().eval()
    image = make_image(43, True)
    x = transform()(image).unsqueeze(0)
    assert x.shape == (1, 3, 128, 128)
    cam = GradCAM(model)
    try:
        probs, heat, target = cam(x, 0)
        assert probs.shape == (2,)
        assert torch.isclose(probs.sum(), torch.tensor(1.))
        assert heat.shape == (128, 128)
        assert np.isfinite(heat).all() and heat.min() >= 0 and heat.max() <= 1
        assert target == 0
        assert overlay(image, heat).size == image.size
    finally:
        cam.close()

def test_checkpoint_roundtrip_and_class_guard(tmp_path):
    model = build_model().eval()
    path = tmp_path/'test.pt'
    saved = {'format_version': 1, 'state_dict': model.state_dict(), 'classes': ['defect', 'normal'], 'image_size': 128}
    torch.save(saved, path)
    restored, _ = load_checkpoint(path)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))
    saved['classes'] = ['normal', 'defect']
    torch.save(saved, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)

def test_wrong_dataset_classes_rejected(tmp_path):
    for label in ['a', 'b']:
        (tmp_path/label).mkdir()
        make_image(1, False).save(tmp_path/label/'image.png')
    with pytest.raises(ValueError):
        dataset(tmp_path, 128)

def test_grayscale_image_conversion(tmp_path):
    path = tmp_path/'gray.png'
    Image.new('L', (20, 20), 127).save(path)
    assert open_image(path).mode == 'RGB'

def test_metrics_account_for_all_samples(tmp_path):
    generate(tmp_path, 4)
    ds = dataset(tmp_path/'test', 128)
    result = score(build_model().eval(), DataLoader(ds, batch_size=4))
    assert result['samples'] == 8
    assert sum(map(sum, result['confusion_matrix'])) == 8
    assert 0 <= result['defect_recall'] <= 1
