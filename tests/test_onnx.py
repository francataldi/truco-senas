import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

import onnx
import onnxruntime as ort

from agents.ppo.net import NetConfig, build_net, save_checkpoint
from agents.ppo.onnx_export import export_onnx, sample_inputs, verify_onnx
from truco.rules import Rules

PAREJAS = Rules(num_players=4, signals=True)
SIN_SENAS = Rules(num_players=4, signals=False)
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("rules", [PAREJAS, SIN_SENAS], ids=["con_senas", "sin_senas"])
def test_onnx_logits_match_torch(tmp_path, rules):
    torch.manual_seed(0)
    net = build_net(rules, NetConfig(hidden=48, depth=2))
    path = export_onnx(net, tmp_path / "politica.onnx", rules=rules)
    report = verify_onnx(net, path, rules, samples=600, seed=1)
    assert report["ok"], report
    assert report["max_abs_logits"] < 1e-4
    assert report["masked_logits_equal"]
    assert report["argmax_agreement"] == 1.0


def test_onnx_has_dynamic_batch_and_metadata(tmp_path):
    net = build_net(PAREJAS, NetConfig(hidden=32, depth=1))
    path = export_onnx(net, tmp_path / "politica.onnx", rules=PAREJAS, extra={"iteration": 9})
    props = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    assert props["obs_dim"] == str(net.obs_dim)
    assert props["n_actions"] == str(net.n_actions)
    assert props["iteration"] == "9"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["obs", "mask"]
    for batch in (1, 7):
        obs, mask = sample_inputs(PAREJAS, batch, seed=batch)
        logits, value, aux = session.run(None, {"obs": obs, "mask": mask})
        assert logits.shape == (batch, net.n_actions)
        assert value.shape == (batch,)
        assert aux.shape == (batch, 8)
        assert np.all(logits[~mask] <= -1e8)


def test_export_script_end_to_end(tmp_path):
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(checkpoint, build_net(PAREJAS, NetConfig(hidden=32, depth=1)), PAREJAS, {"iteration": 3})
    out = tmp_path / "exportado.onnx"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_onnx.py"), "--checkpoint", str(checkpoint), "--out", str(out), "--samples", "300", "--bench", "20"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert "ok:" in result.stdout
