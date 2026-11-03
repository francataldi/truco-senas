from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from truco import actions as A
from truco.encoding import obs_size
from truco.features import N_FEATURES
from truco.rules import Rules

NEG_INF = -1e9


@dataclass(frozen=True)
class NetConfig:
    hidden: int = 512
    depth: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


class TrucoNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, cfg: NetConfig = NetConfig(), aux_dim: int = N_FEATURES) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = obs_dim
        for _ in range(cfg.depth):
            layers += [nn.Linear(width, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU()]
            width = cfg.hidden
        self.body = nn.Sequential(*layers)
        self.policy = nn.Linear(cfg.hidden, n_actions)
        self.value = nn.Linear(cfg.hidden, 1)
        self.aux = nn.Linear(cfg.hidden, aux_dim)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        logits = self.policy(h).float().masked_fill(~mask, NEG_INF)
        return logits, self.value(h).float().squeeze(-1), self.aux(h).float()


def build_net(rules: Rules, cfg: NetConfig) -> TrucoNet:
    return TrucoNet(obs_size(rules), A.num_actions(rules.signal_vocab), cfg)


def save_checkpoint(path, net: TrucoNet, rules: Rules, extra: dict | None = None) -> None:
    payload = {
        "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "rules": rules.to_dict(),
        "net": net.cfg.to_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path, device: str = "cpu") -> tuple[TrucoNet, Rules, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    rules = Rules(**payload["rules"])
    net = build_net(rules, NetConfig(**payload["net"]))
    net.load_state_dict(payload["model"])
    net.to(device)
    net.eval()
    return net, rules, payload.get("extra", {})
