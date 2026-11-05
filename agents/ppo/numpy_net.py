from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from truco.rules import Rules

from .policy import PolicyAgent

NEG_INF = -1e9
LAYER_NORM_EPS = 1e-5
ERF_P = 0.3275911
ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def erf(x: np.ndarray) -> np.ndarray:
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + ERF_P * ax)
    poly = t * (ERF_A[0] + t * (ERF_A[1] + t * (ERF_A[2] + t * (ERF_A[3] + t * ERF_A[4]))))
    return sign * (1.0 - poly * np.exp(-ax * ax))


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + LAYER_NORM_EPS) * weight + bias


@dataclass(frozen=True)
class NumpyNetConfig:
    hidden: int
    depth: int


class NumpyNet:
    def __init__(self, weights: dict[str, np.ndarray], cfg: NumpyNetConfig) -> None:
        self.cfg = cfg
        self.layers = []
        for i in range(cfg.depth):
            linear = 3 * i
            norm = linear + 1
            self.layers.append(
                (
                    weights[f"body.{linear}.weight"].T.astype(np.float32),
                    weights[f"body.{linear}.bias"].astype(np.float32),
                    weights[f"body.{norm}.weight"].astype(np.float32),
                    weights[f"body.{norm}.bias"].astype(np.float32),
                )
            )
        self.heads = {name: (weights[f"{name}.weight"].T.astype(np.float32), weights[f"{name}.bias"].astype(np.float32)) for name in ("policy", "value", "aux")}
        self.obs_dim = self.layers[0][0].shape[0]
        self.n_actions = self.heads["policy"][0].shape[1]

    def eval(self) -> NumpyNet:
        return self

    def __call__(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = np.asarray(obs, dtype=np.float32)
        for w, b, g, beta in self.layers:
            h = gelu(layer_norm(h @ w + b, g, beta))
        logits = h @ self.heads["policy"][0] + self.heads["policy"][1]
        logits = np.where(np.asarray(mask, dtype=bool), logits, NEG_INF)
        value = (h @ self.heads["value"][0] + self.heads["value"][1])[..., 0]
        aux = h @ self.heads["aux"][0] + self.heads["aux"][1]
        return logits, value, aux


class NumpyPPOAgent(PolicyAgent):
    name = "ppo"

    def forward(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.net(obs, mask)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> NumpyPPOAgent:
        net, rules, _ = load_numpy_policy(path)
        return cls(net, rules=rules, **kwargs)


def save_numpy_policy(path: str | Path, weights: dict[str, np.ndarray], rules: dict, net: dict, extra: dict | None = None, dtype=np.float32) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({"rules": rules, "net": net, "extra": extra or {}}, ensure_ascii=False)
    arrays = {k: np.asarray(v, dtype=dtype) for k, v in weights.items()}
    np.savez_compressed(path, __meta__=np.array(meta), **arrays)
    return path


def load_numpy_policy(path: str | Path) -> tuple[NumpyNet, Rules, dict]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["__meta__"]))
        weights = {k: data[k] for k in data.files if k != "__meta__"}
    rules = Rules(**meta["rules"])
    net = NumpyNet(weights, NumpyNetConfig(int(meta["net"]["hidden"]), int(meta["net"]["depth"])))
    return net, rules, meta.get("extra", {})


def export_checkpoint(checkpoint: str | Path, out: str | Path, dtype=np.float32) -> Path:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = {k: v.detach().cpu().float().numpy() for k, v in payload["model"].items()}
    return save_numpy_policy(out, weights, payload["rules"], payload["net"], payload.get("extra", {}), dtype)
