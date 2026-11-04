from __future__ import annotations

import json
import logging
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from truco import actions as A
from truco.encoding import action_mask, encode
from truco.engine import TrucoHand
from truco.features import FEATURE_NAMES
from truco.rules import Rules

from ..heuristic import HeuristicAgent
from .net import TrucoNet

INPUTS = ("obs", "mask")
OUTPUTS = ("logits", "value", "aux")


def export_onnx(net: TrucoNet, path: str | Path, rules: Rules | None = None, opset: int = 18, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net = net.eval()
    obs = torch.zeros(2, net.obs_dim)
    mask = torch.ones(2, net.n_actions, dtype=torch.bool)
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            batch = torch.export.Dim("batch")
            torch.onnx.export(
                net,
                (obs, mask),
                str(path),
                output_names=list(OUTPUTS),
                dynamic_shapes={"obs": {0: batch}, "mask": {0: batch}},
                opset_version=opset,
                dynamo=True,
                verbose=False,
            )
        except TypeError:
            torch.onnx.export(
                net,
                (obs, mask),
                str(path),
                input_names=list(INPUTS),
                output_names=list(OUTPUTS),
                dynamic_axes={name: {0: "batch"} for name in (*INPUTS, *OUTPUTS)},
                opset_version=opset,
            )
    _add_metadata(path, net, rules, extra or {})
    return path


def _add_metadata(path: Path, net: TrucoNet, rules: Rules | None, extra: dict) -> None:
    try:
        import onnx
    except ImportError:
        return
    model = onnx.load(str(path))
    props = {
        "obs_dim": str(net.obs_dim),
        "n_actions": str(net.n_actions),
        "hidden": str(net.cfg.hidden),
        "depth": str(net.cfg.depth),
        "outputs": "logits enmascarados, valor de la mano y creencias sobre el compañero",
        "aux_features": json.dumps(list(FEATURE_NAMES)),
        "signal_base": str(A.SIGNAL_BASE),
    }
    if rules is not None:
        props["rules"] = json.dumps(rules.to_dict())
    props.update({k: json.dumps(v) if not isinstance(v, str) else v for k, v in extra.items()})
    onnx.helper.set_model_props(model, props)
    onnx.save(model, str(path))


def sample_inputs(rules: Rules, samples: int, seed: int = 0, n_actions: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    agents = [HeuristicAgent(seed=seed + i) for i in range(rules.num_players)]
    n_actions = n_actions or A.num_actions(rules.signal_vocab)
    obs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    while len(obs) < samples:
        hand = TrucoHand(
            rules,
            rng=random.Random(rng.random()),
            scores=(rng.randrange(rules.target), rng.randrange(rules.target)),
            mano=rng.randrange(rules.num_players),
        )
        while not hand.done and len(obs) < samples:
            seat = hand.current_player
            obs.append(encode(hand, seat))
            masks.append(action_mask(hand, n_actions))
            action = rng.choice(hand.legal_actions()) if rng.random() < 0.3 else agents[seat].act(hand, seat)
            hand.step(action)
    return np.stack(obs).astype(np.float32), np.stack(masks)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def verify_onnx(net: TrucoNet, path: str | Path, rules: Rules, samples: int = 2048, seed: int = 0, atol: float = 1e-4, bench: int = 0) -> dict:
    import onnxruntime as ort

    obs, mask = sample_inputs(rules, samples, seed, net.n_actions)
    net = net.eval()
    with torch.no_grad():
        t_logits, t_value, t_aux = (x.numpy() for x in net(torch.from_numpy(obs), torch.from_numpy(mask)))
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    o_logits, o_value, o_aux = session.run(list(OUTPUTS), dict(zip(names, (obs, mask))))
    legal = mask
    diff_logits = float(np.abs(o_logits - t_logits)[legal].max())
    masked_ok = bool(np.array_equal(o_logits[~legal], t_logits[~legal]))
    diff_probs = float(np.abs(_softmax(o_logits) - _softmax(t_logits)).max())
    agreement = float((o_logits.argmax(-1) == t_logits.argmax(-1)).mean())
    report = {
        "samples": int(samples),
        "max_abs_logits": diff_logits,
        "max_abs_probs": diff_probs,
        "max_abs_value": float(np.abs(o_value - t_value).max()),
        "max_abs_aux": float(np.abs(o_aux - t_aux).max()),
        "masked_logits_equal": masked_ok,
        "argmax_agreement": agreement,
        "atol": atol,
    }
    report["ok"] = bool(
        diff_logits <= atol
        and report["max_abs_value"] <= atol
        and report["max_abs_aux"] <= atol
        and masked_ok
        and agreement == 1.0
    )
    if bench:
        report["latency_ms"] = _benchmark(net, session, names, obs[:1], mask[:1], bench)
    return report


def _benchmark(net: TrucoNet, session, names: list[str], obs: np.ndarray, mask: np.ndarray, repeats: int) -> dict:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    feeds = dict(zip(names, (obs, mask)))
    x, m = torch.from_numpy(obs), torch.from_numpy(mask)
    timings = {}
    try:
        with torch.no_grad():
            for name, fn in (("torch", lambda: net(x, m)), ("onnxruntime", lambda: session.run(None, feeds))):
                for _ in range(min(50, repeats)):
                    fn()
                start = time.perf_counter()
                for _ in range(repeats):
                    fn()
                timings[name] = round((time.perf_counter() - start) / repeats * 1e3, 4)
    finally:
        torch.set_num_threads(threads)
    return timings
