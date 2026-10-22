import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from trueocr import load_model
from trueocr.cli import main
from trueocr.preprocess import prepare_image

SAMPLE = Path(__file__).parents[1] / "examples" / "sample.png"
MODEL_DIR = os.environ.get("TRUEOCR_MODEL_DIR")
needs_model = pytest.mark.skipif(not MODEL_DIR, reason="đặt TRUEOCR_MODEL_DIR tới thư mục model")


def test_inputs_give_same_crop():
    bgr = cv2.imread(str(SAMPLE))
    expected = prepare_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    for value in (bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA), SAMPLE.read_bytes(), SAMPLE):
        assert np.array_equal(prepare_image(value), expected)


def test_bad_input_rejected():
    for value in (b"not an image", np.zeros((0, 5), np.uint8), np.zeros((4, 4), np.float32)):
        with pytest.raises(ValueError):
            prepare_image(value)


def test_missing_model_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="model_dir"):
        load_model(model_dir=tmp_path)


@pytest.fixture(scope="module")
def model():
    return load_model(model_dir=MODEL_DIR, threads=1)


@needs_model
def test_page_upright_and_turned(model):
    result = model.predict(SAMPLE)
    assert len(result.lines) == 6
    assert result.lines[3]["text"] == "Ngày sinh: 01/01/1990"
    turned = model.predict(Image.open(SAMPLE).rotate(90, expand=True))
    assert turned.data["rotation_clockwise_degrees"] == 90
    assert turned.text == result.text
    blank = model.predict(Image.new("RGB", (640, 480), "white"))
    assert blank.text == "" and blank.lines == []


@needs_model
def test_line_mode_and_batching(model):
    page = model.predict(SAMPLE)
    gray = np.asarray(page.image.convert("L"))
    crops = [gray[y0:y1, x0:x1] for x0, y0, x1, y1 in (line["bbox"] for line in page.lines)]
    one_by_one = model.recognizer.predict(crops, batch_size=1)
    assert [p.text for p in one_by_one] == [line["text"] for line in page.lines]
    line = model.predict(cv2.imencode(".png", crops[3])[1].tobytes(), mode="line")
    assert line.text == page.lines[3]["text"]


@needs_model
def test_cli_json(capsys, tmp_path):
    assert main([str(SAMPLE), "--model-dir", MODEL_DIR, "--json", "--out", str(tmp_path)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["lines"]) == 6
    assert len(list(tmp_path.iterdir())) == 3  # .txt, .json, _boxes.png
