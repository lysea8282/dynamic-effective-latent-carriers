from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(192, 384), nn.GELU(), nn.Linear(384, 192), nn.GELU(), nn.Linear(192, 8)
        )

    def forward(self, carrier: torch.Tensor) -> torch.Tensor:
        return self.network(carrier)


class DeterministicBeliefModel(nn.Module):
    """Deterministic recurrent model used for the frozen experiments."""

    carrier_dim = 192
    cap = 3.0

    def __init__(self) -> None:
        super().__init__()
        self.observation_encoder = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, 64))
        self.action_encoder = nn.Sequential(nn.Linear(4, 128), nn.GELU(), nn.Linear(128, 64))
        self.message_network = nn.Sequential(
            nn.Linear(192, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 64)
        )
        self.deterministic_update = nn.GRUCell(144, 48)
        self.global_update = nn.GRUCell(192, 64)
        self.decoder = Decoder()
        self.prior_network = nn.Sequential(
            nn.Linear(176, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 16)
        )
        self.posterior_network = nn.Sequential(
            nn.Linear(176, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 16)
        )

    @staticmethod
    def unpack(carrier: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return carrier[:, :48], carrier[:, 48:64], carrier[:, 64:112], carrier[:, 112:128], carrier[:, 128:]

    @staticmethod
    def pack(
        h1: torch.Tensor,
        middle1: torch.Tensor,
        h2: torch.Tensor,
        middle2: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((h1, middle1, h2, middle2, global_state), dim=-1)

    def zero_carrier(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.carrier_dim, device=device, dtype=dtype)

    def _deterministic(self, carrier: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h1, middle1, h2, middle2, global_state = self.unpack(carrier)
        message1 = self.message_network(torch.cat((h1, middle1, h2, middle2, global_state), dim=-1))
        message2 = self.message_network(torch.cat((h2, middle2, h1, middle1, global_state), dim=-1))
        magnitude = torch.linalg.norm(action[:, 2:], dim=-1, keepdim=True)[:, None].expand(-1, 2, -1)
        per_object = torch.cat((action[:, :2, None], action[:, None, 2:].expand(-1, 2, -1), magnitude), dim=-1)
        action_embedding = self.action_encoder(per_object)
        new_h1 = self.deterministic_update(torch.cat((middle1, message1, action_embedding[:, 0]), dim=-1), h1)
        new_h2 = self.deterministic_update(torch.cat((middle2, message2, action_embedding[:, 1]), dim=-1), h2)
        new_global = self.global_update(
            torch.cat((new_h1, middle1, new_h2, middle2, self.action_encoder(action)), dim=-1), global_state
        )
        return new_h1, new_h2, new_global, action_embedding

    def _step(self, carrier: torch.Tensor, action: torch.Tensor, observation: torch.Tensor | None) -> torch.Tensor:
        h1, h2, global_state, action_embedding = self._deterministic(carrier, action)
        prior1 = self.cap * torch.tanh(
            self.prior_network(torch.cat((h1, global_state, action_embedding[:, 0]), dim=-1)) / self.cap
        )
        prior2 = self.cap * torch.tanh(
            self.prior_network(torch.cat((h2, global_state, action_embedding[:, 1]), dim=-1)) / self.cap
        )
        prior = torch.stack((prior1, prior2), dim=1)
        if observation is None:
            selected = prior
        else:
            visibility = observation[..., 4:5]
            encoded = self.observation_encoder(observation)
            post1 = self.cap * torch.tanh(
                self.posterior_network(torch.cat((h1, global_state, encoded[:, 0]), dim=-1)) / self.cap
            )
            post2 = self.cap * torch.tanh(
                self.posterior_network(torch.cat((h2, global_state, encoded[:, 1]), dim=-1)) / self.cap
            )
            posterior = torch.stack((post1, post2), dim=1)
            selected = visibility * posterior + (1.0 - visibility) * prior
        return self.pack(h1, selected[:, 0], h2, selected[:, 1], global_state)

    def encode_history(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        carrier = self.zero_carrier(observations.shape[0], observations.device, observations.dtype)
        saved = []
        for time_index in range(observations.shape[1]):
            carrier = self._step(carrier, actions[:, time_index], observations[:, time_index])
            saved.append(carrier)
        return carrier, {"saved_carriers": saved, "carrier_complete": True}

    def transition(self, carrier: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._step(carrier, action, None)

    def decode(self, carrier: torch.Tensor) -> torch.Tensor:
        return self.decoder(carrier)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def autonomous_sequence(
    model: DeterministicBeliefModel,
    carrier: torch.Tensor,
    actions: torch.Tensor,
    anchor: int,
    horizon: int = 12,
) -> torch.Tensor:
    states = [model.decode(carrier).reshape(-1, 2, 4)]
    for source_frame in range(anchor, anchor + horizon):
        carrier = model.transition(carrier, actions[:, source_frame])
        states.append(model.decode(carrier).reshape(-1, 2, 4))
    return torch.stack(states, dim=1)
