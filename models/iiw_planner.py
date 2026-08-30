"""Point-aligned IIW planners, including a backward-compatible residual MoE.

``IIWPlanner`` is the original dense planner.  Its module names and state-dict
keys are intentionally unchanged so every existing dense checkpoint remains
strict-loadable.

``MoEIIWPlanner`` adds a sample-global soft router and small low-rank residual
experts *after* the dense logits.  A dense checkpoint can therefore be
migrated with exact logit parity: every residual expert starts at exactly zero
while the inherited dense parameter keys keep their original names.

Both planners accept scene points ``[B,N,6]``, CLIP text features ``[B,512]``
and planar start state ``[B,4]``.  Their public ``forward`` and
``forward_logits`` methods retain the original tensor-only API and produce
``[B,Q,N,6]``.  MoE routing is opt-in through ``forward_with_routing``.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Union

import torch
import torch.nn as nn


NATIVE_BODY_PART_NAMES: Tuple[str, ...] = (
    "base",
    "spine",
    "right_hand",
    "left_hand",
    "right_foot",
    "left_foot",
)

_DENSE_CONFIG_KEYS = (
    "scene_dim",
    "text_dim",
    "state_dim",
    "num_phases",
    "num_bodies",
    "hidden_dim",
    "point_dim",
    "context_dim",
)
_MOE_CONFIG_KEYS = (
    "num_experts",
    "residual_rank",
    "router_hidden_dim",
    "routing_temperature",
)


class IIWPlanner(nn.Module):
    """Original permutation-equivariant dense IIW planner.

    Do not rename its submodules: those names are the deployed dense
    checkpoint contract and are also the inherited trunk of the MoE planner.
    """

    geometry_dim = 5

    def __init__(
        self,
        scene_dim: int = 6,
        text_dim: int = 512,
        state_dim: int = 4,
        num_phases: int = 8,
        num_bodies: int = 6,
        hidden_dim: int = 128,
        point_dim: int = 96,
        context_dim: int = 128,
    ) -> None:
        super().__init__()
        for name, value in (
            ("scene_dim", scene_dim),
            ("text_dim", text_dim),
            ("state_dim", state_dim),
            ("num_phases", num_phases),
            ("num_bodies", num_bodies),
            ("hidden_dim", hidden_dim),
            ("point_dim", point_dim),
            ("context_dim", context_dim),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if scene_dim < 3:
            raise ValueError("scene_dim must contain at least xyz")
        if state_dim != 4:
            raise ValueError(
                "state_dim must be 4: start_x, start_y, direction_x, direction_y"
            )
        if num_bodies != len(NATIVE_BODY_PART_NAMES):
            raise ValueError(
                "num_bodies must be {} for native IIW order, got {}".format(
                    len(NATIVE_BODY_PART_NAMES), num_bodies
                )
            )

        self.scene_dim = scene_dim
        self.text_dim = text_dim
        self.state_dim = state_dim
        self.num_phases = num_phases
        self.num_bodies = num_bodies
        self.hidden_dim = hidden_dim
        self.point_dim = point_dim
        self.context_dim = context_dim

        self.point_encoder = nn.Sequential(
            nn.Linear(scene_dim + self.geometry_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, point_dim),
            nn.GELU(),
        )
        self.scene_encoder = nn.Sequential(
            nn.Linear(2 * point_dim, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        self.text_encoder = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, context_dim),
            nn.GELU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(3 * context_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, context_dim),
            nn.GELU(),
        )

        self.point_film = nn.Linear(context_dim, 2 * point_dim)
        self.point_key = nn.Sequential(
            nn.LayerNorm(point_dim),
            nn.Linear(point_dim, point_dim),
        )
        self.point_body_bias = nn.Linear(point_dim, num_bodies)

        self.phase_embedding = nn.Embedding(num_phases, context_dim)
        self.body_embedding = nn.Embedding(num_bodies, context_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(3 * context_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, point_dim),
        )
        self.query_bias = nn.Sequential(nn.Linear(point_dim, 1))
        nn.init.constant_(self.query_bias[0].bias, -3.0)

    @property
    def config(self) -> Dict[str, int]:
        """Return the exact legacy constructor manifest."""
        return {
            "scene_dim": self.scene_dim,
            "text_dim": self.text_dim,
            "state_dim": self.state_dim,
            "num_phases": self.num_phases,
            "num_bodies": self.num_bodies,
            "hidden_dim": self.hidden_dim,
            "point_dim": self.point_dim,
            "context_dim": self.context_dim,
        }

    def _validate_inputs(
        self,
        scene_points: torch.Tensor,
        text_features: torch.Tensor,
        state: torch.Tensor,
    ) -> None:
        for name, value in (
            ("scene_points", scene_points),
            ("text_features", text_features),
            ("state", state),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError("{} must be a torch.Tensor".format(name))
            if not torch.is_floating_point(value):
                raise TypeError("{} must be floating point".format(name))
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("{} contains NaN or Inf".format(name))

        if text_features.ndim != 2 or text_features.shape[1] != self.text_dim:
            raise ValueError(
                "text_features must have shape [B,{}], got {}".format(
                    self.text_dim, tuple(text_features.shape)
                )
            )
        if scene_points.ndim != 3 or scene_points.shape[2] != self.scene_dim:
            raise ValueError(
                "scene_points must have shape [B,N,{}], got {}".format(
                    self.scene_dim, tuple(scene_points.shape)
                )
            )
        if scene_points.shape[1] <= 0:
            raise ValueError("scene point cloud must not be empty")
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                "state must have shape [B,{}], got {}".format(
                    self.state_dim, tuple(state.shape)
                )
            )
        batch_size = scene_points.shape[0]
        if text_features.shape[0] != batch_size or state.shape[0] != batch_size:
            raise ValueError("planner input batch sizes differ")

        device = scene_points.device
        if text_features.device != device or state.device != device:
            raise ValueError("all planner inputs must share one device")
        parameter = next(self.parameters())
        if parameter.device != device:
            raise ValueError(
                "input device {} differs from planner device {}".format(
                    device, parameter.device
                )
            )
        if text_features.dtype != scene_points.dtype or state.dtype != scene_points.dtype:
            raise TypeError("all planner inputs must share one dtype")

    @staticmethod
    def _planar_geometry(
        scene_points: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return start-relative ``[dx,dy,r,forward,lateral]`` features."""
        delta = scene_points[..., :2] - state[:, None, :2]
        radius = torch.sqrt(delta.square().sum(dim=-1, keepdim=True) + 1e-8)
        direction = state[:, None, 2:4]
        direction = direction / torch.sqrt(
            direction.square().sum(dim=-1, keepdim=True) + 1e-8
        )
        forward = (delta * direction).sum(dim=-1, keepdim=True)
        lateral = (
            -delta[..., 0:1] * direction[..., 1:2]
            + delta[..., 1:2] * direction[..., 0:1]
        )
        return torch.cat((delta, radius, forward, lateral), dim=-1)

    def _forward_features(
        self,
        scene_points: torch.Tensor,
        text_features: torch.Tensor,
        state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build the shared dense features once for dense or MoE inference."""
        self._validate_inputs(scene_points, text_features, state)
        batch_size = scene_points.shape[0]
        geometry = self._planar_geometry(scene_points, state)
        point_feature = self.point_encoder(torch.cat((scene_points, geometry), dim=-1))

        scene_mean = point_feature.mean(dim=1)
        scene_max = point_feature.max(dim=1).values
        scene_context = self.scene_encoder(torch.cat((scene_mean, scene_max), dim=-1))
        text_context = self.text_encoder(text_features)
        state_context = self.state_encoder(state)
        context = self.context_fusion(
            torch.cat((scene_context, text_context, state_context), dim=-1)
        )

        film_scale, film_shift = self.point_film(context).chunk(2, dim=-1)
        conditioned_points = point_feature * (
            1.0 + torch.tanh(film_scale)[:, None, :]
        ) + film_shift[:, None, :]
        point_key = self.point_key(conditioned_points)
        point_bias = self.point_body_bias(conditioned_points)

        phase = self.phase_embedding.weight.view(
            1, self.num_phases, 1, self.context_dim
        ).expand(batch_size, -1, self.num_bodies, -1)
        body = self.body_embedding.weight.view(
            1, 1, self.num_bodies, self.context_dim
        ).expand(batch_size, self.num_phases, -1, -1)
        global_query = context[:, None, None, :].expand(
            -1, self.num_phases, self.num_bodies, -1
        )
        query = self.query_mlp(torch.cat((global_query, phase, body), dim=-1))
        query_bias = self.query_bias(query).squeeze(-1)
        return {
            "point_key": point_key,
            "point_bias": point_bias,
            "query": query,
            "query_bias": query_bias,
            "scene_context": scene_context,
            "text_context": text_context,
            "state_context": state_context,
            "context": context,
        }

    def _dense_logits(
        self,
        features: Mapping[str, torch.Tensor],
        num_points: int,
    ) -> torch.Tensor:
        point_key = features["point_key"]
        query = features["query"]
        logits = torch.einsum("bnd,bqcd->bqnc", point_key, query)
        logits = logits / math.sqrt(float(self.point_dim))
        logits = (
            logits
            + features["point_bias"][:, None, :, :]
            + features["query_bias"][:, :, None, :]
        )
        expected = (
            point_key.shape[0], self.num_phases, num_points, self.num_bodies
        )
        if logits.shape != expected:
            raise RuntimeError(
                "internal IIW shape mismatch: expected {}, got {}".format(
                    expected, tuple(logits.shape)
                )
            )
        return logits.contiguous()

    def forward_logits(
        self,
        scene_points: torch.Tensor,
        text_features: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return dense unconstrained logits ``[B,Q,N,6]``."""
        features = self._forward_features(scene_points, text_features, state)
        return self._dense_logits(features, int(scene_points.shape[1]))

    def forward(
        self,
        scene_points: torch.Tensor,
        text_features: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return IIW probabilities ``[B,Q,N,6]``."""
        return torch.sigmoid(self.forward_logits(scene_points, text_features, state))


class _DenseParameterView:
    """Non-registering parameter view used by strict freeze scripts."""

    def __init__(self, owner: "MoEIIWPlanner") -> None:
        self._owner = owner

    def parameters(self, recurse: bool = True) -> Iterable[nn.Parameter]:
        del recurse
        return self._owner.dense_parameters()


class MoEIIWPlanner(IIWPlanner):
    """Two-or-more expert residual MoE around a migrated dense planner.

    The gate is *sample global*: one probability vector ``[B,K]`` applies to
    every phase, point and body channel of a sample.  Experts are low-rank
    residuals over the already-computed dense point/query features, so the
    expensive trunk is evaluated only once.
    """

    planner_type = "moe_iiw_v1"
    format_version = 2

    def __init__(
        self,
        scene_dim: int = 6,
        text_dim: int = 512,
        state_dim: int = 4,
        num_phases: int = 8,
        num_bodies: int = 6,
        hidden_dim: int = 128,
        point_dim: int = 96,
        context_dim: int = 128,
        num_experts: int = 2,
        residual_rank: int = 16,
        router_hidden_dim: int = 64,
        routing_temperature: float = 1.0,
        state_mean: Optional[Union[torch.Tensor, Iterable[float]]] = None,
        state_std: Optional[Union[torch.Tensor, Iterable[float]]] = None,
    ) -> None:
        super().__init__(
            scene_dim=scene_dim,
            text_dim=text_dim,
            state_dim=state_dim,
            num_phases=num_phases,
            num_bodies=num_bodies,
            hidden_dim=hidden_dim,
            point_dim=point_dim,
            context_dim=context_dim,
        )
        if not isinstance(num_experts, int) or num_experts < 2:
            raise ValueError("num_experts must be an integer >= 2")
        if not isinstance(residual_rank, int) or residual_rank <= 0:
            raise ValueError("residual_rank must be a positive integer")
        if not isinstance(router_hidden_dim, int) or router_hidden_dim <= 0:
            raise ValueError("router_hidden_dim must be a positive integer")
        if not isinstance(routing_temperature, (float, int)) or not math.isfinite(
            float(routing_temperature)
        ) or float(routing_temperature) <= 0.0:
            raise ValueError("routing_temperature must be finite and positive")

        # Capture exactly the legacy keys before registering any MoE state.
        self._dense_state_key_names = tuple(self.state_dict().keys())
        self.num_experts = num_experts
        self.residual_rank = residual_rank
        self.router_hidden_dim = router_hidden_dim
        self.routing_temperature = float(routing_temperature)

        mean = self._coerce_state_stat("state_mean", state_mean, 0.0)
        std = self._coerce_state_stat("state_std", state_std, 1.0)
        if bool((std <= 0.0).any().item()):
            raise ValueError("state_std must be strictly positive")
        self.register_buffer("state_mean", mean)
        self.register_buffer("state_std", std.clamp_min(1e-4))

        router_input_dim = 2 * context_dim + state_dim
        self.router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_experts),
        )
        # Uniform routing is the least-assumptive migration start.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        # Leading K is deliberate: diagnostics and checkpoints can compare
        # experts without relying on fragile ModuleList naming conventions.
        self.expert_point_projection = nn.Parameter(
            torch.empty(num_experts, point_dim, residual_rank)
        )
        self.expert_query_projection = nn.Parameter(
            torch.zeros(num_experts, point_dim, residual_rank)
        )
        self.expert_residual_scale = nn.Parameter(torch.ones(num_experts))
        self.expert_residual_bias = nn.Parameter(
            torch.zeros(num_experts, num_bodies)
        )
        for expert in range(num_experts):
            nn.init.kaiming_uniform_(
                self.expert_point_projection[expert], a=math.sqrt(5.0)
            )

    def _coerce_state_stat(
        self,
        name: str,
        value: Optional[Union[torch.Tensor, Iterable[float]]],
        default: float,
    ) -> torch.Tensor:
        if value is None:
            result = torch.full((self.state_dim,), float(default), dtype=torch.float32)
        else:
            result = torch.as_tensor(value, dtype=torch.float32).detach().clone()
        if result.shape != (self.state_dim,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name, self.state_dim, tuple(result.shape)
                )
            )
        if not bool(torch.isfinite(result).all().item()):
            raise ValueError(name + " contains NaN/Inf")
        return result

    @property
    def config(self) -> Dict[str, Union[int, float]]:
        result: Dict[str, Union[int, float]] = dict(super().config)
        result.update({
            "num_experts": self.num_experts,
            "residual_rank": self.residual_rank,
            "router_hidden_dim": self.router_hidden_dim,
            "routing_temperature": self.routing_temperature,
        })
        return result

    @property
    def dense_planner(self) -> _DenseParameterView:
        """Compatibility view; it does not add a ``dense_planner.`` key prefix."""
        return _DenseParameterView(self)

    @property
    def base_planner(self) -> _DenseParameterView:
        return self.dense_planner

    def dense_parameters(self) -> Iterable[nn.Parameter]:
        """Yield inherited dense parameters only, preserving old key names."""
        for name, parameter in self.named_parameters():
            if not name.startswith("router.") and not name.startswith("expert_"):
                yield parameter

    def moe_parameters(self) -> Iterable[nn.Parameter]:
        """Yield router and residual-expert parameters only."""
        for name, parameter in self.named_parameters():
            if name.startswith("router.") or name.startswith("expert_"):
                yield parameter

    def load_dense_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """Migrate a legacy dense state dict without touching zero experts."""
        source = _strip_module_prefix(state_dict)
        expected = set(self._dense_state_key_names)
        supplied = set(source)
        missing = sorted(expected.difference(supplied))
        unexpected = sorted(supplied.difference(expected))
        if strict and (missing or unexpected):
            raise ValueError(
                "dense state-dict mismatch: missing={} unexpected={}".format(
                    missing, unexpected
                )
            )
        own = self.state_dict()
        with torch.no_grad():
            for name in sorted(expected.intersection(supplied)):
                value = source[name]
                if not isinstance(value, torch.Tensor):
                    raise TypeError("dense state value is not a tensor: " + name)
                if own[name].shape != value.shape:
                    raise ValueError(
                        "dense state shape mismatch for {}: {} vs {}".format(
                            name, tuple(value.shape), tuple(own[name].shape)
                        )
                    )
                own[name].copy_(value.to(device=own[name].device, dtype=own[name].dtype))

    def _normalized_state(self, state: torch.Tensor) -> torch.Tensor:
        mean = self.state_mean.to(device=state.device, dtype=state.dtype)
        std = self.state_std.to(device=state.device, dtype=state.dtype)
        return (state - mean[None, :]) / std[None, :]

    def _route(
        self,
        features: Mapping[str, torch.Tensor],
        state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        router_input = torch.cat(
            (
                features["scene_context"],
                features["text_context"],
                self._normalized_state(state),
            ),
            dim=-1,
        )
        raw_logits = self.router(router_input)
        routing_logits = raw_logits / self.routing_temperature
        probabilities = torch.softmax(routing_logits, dim=-1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        return {
            "raw_routing_logits": raw_logits,
            "routing_logits": routing_logits,
            "routing_probabilities": probabilities,
            "selected_expert": probabilities.argmax(dim=-1),
            "routing_entropy": entropy,
            "expert_load": probabilities.mean(dim=0),
            "routing_temperature": routing_logits.new_tensor(
                self.routing_temperature
            ),
        }

    def _expert_latents(
        self,
        features: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        point_latent = torch.einsum(
            "bnd,kdr->bknr",
            features["point_key"],
            self.expert_point_projection,
        )
        query_latent = torch.einsum(
            "bqcd,kdr->bkqcr",
            features["query"],
            self.expert_query_projection,
        )
        return point_latent, query_latent

    def _single_expert_residual(
        self,
        point_latent: torch.Tensor,
        query_latent: torch.Tensor,
        expert: int,
    ) -> torch.Tensor:
        residual = torch.einsum(
            "bnr,bqcr->bqnc",
            point_latent[:, expert],
            query_latent[:, expert],
        ) / math.sqrt(float(self.residual_rank))
        residual = residual * self.expert_residual_scale[expert]
        residual = residual + self.expert_residual_bias[expert][None, None, None, :]
        return residual

    def _forward_moe(
        self,
        scene_points: torch.Tensor,
        text_features: torch.Tensor,
        state: torch.Tensor,
        return_expert_logits: bool,
    ) -> Dict[str, torch.Tensor]:
        features = self._forward_features(scene_points, text_features, state)
        dense_logits = self._dense_logits(features, int(scene_points.shape[1]))
        routing = self._route(features, state)
        point_latent, query_latent = self._expert_latents(features)
        probabilities = routing["routing_probabilities"]

        # Accumulate directly into one [B,Q,N,C] tensor during deployment.
        # The [B,K,Q,N,C] diagnostic tensor is created only when requested.
        mixed_residual = torch.zeros_like(dense_logits)
        residual_rows = []
        for expert in range(self.num_experts):
            residual = self._single_expert_residual(
                point_latent, query_latent, expert
            )
            mixed_residual = mixed_residual + (
                probabilities[:, expert, None, None, None] * residual
            )
            if return_expert_logits:
                residual_rows.append(residual)
        logits = (dense_logits + mixed_residual).contiguous()
        result = dict(routing)
        result.update({
            "logits": logits,
            "prediction": torch.sigmoid(logits),
            "dense_logits": dense_logits,
            "mixed_residual_logits": mixed_residual,
        })
        if return_expert_logits:
            expert_residuals = torch.stack(residual_rows, dim=1).contiguous()
            result["expert_residual_logits"] = expert_residuals
            result["expert_logits"] = (
                dense_logits[:, None, :, :, :] + expert_residuals
            ).contiguous()
        return result

    def routing_diagnostics(
        self,
        scene_points: torch.Tensor,
        text_features: torch.Tensor,
        state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return only sample-global gate diagnostics."""
        features = self._forward_features(scene_points, text_features, state)
        return self._route(features, state)

    def forward_with_routing(
        self,
        scene_points: torch.Tensor,
        text_features: torch.Tensor,
        state: torch.Tensor,
        return_expert_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return IIW prediction plus explicit routing diagnostics."""
        return self._forward_moe(
            scene_points, text_features, state, return_expert_logits
        )

    def forward_logits_and_routing(
        self,
        scene_points: torch.Tensor,
        text_features: torch.Tensor,
        state: torch.Tensor,
        return_expert_logits: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Tuple API used by migration and training contracts."""
        result = self.forward_with_routing(
            scene_points,
            text_features,
            state,
            return_expert_logits=return_expert_logits,
        )
        diagnostics = {key: value for key, value in result.items() if key not in (
            "prediction", "logits"
        )}
        return result["logits"], diagnostics

    def forward_logits(
        self,
        scene_points: torch.Tensor,
        text_features: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_moe(
            scene_points, text_features, state, False
        )["logits"]

    def forward(
        self,
        scene_points: torch.Tensor,
        text_features: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_moe(
            scene_points, text_features, state, False
        )["prediction"]


# Factories must remain stable even when the production evaluator temporarily
# monkey-patches the public ``IIWPlanner`` symbol with a compatibility shim.
_DENSE_PLANNER_CLASS = IIWPlanner
_MOE_PLANNER_CLASS = MoEIIWPlanner


def _strip_module_prefix(
    state_dict: Mapping[str, torch.Tensor]
) -> "OrderedDict[str, torch.Tensor]":
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TypeError("checkpoint state dict must be a non-empty mapping")
    keys = [str(key) for key in state_dict]
    all_prefixed = all(key.startswith("module.") for key in keys)
    any_prefixed = any(key.startswith("module.") for key in keys)
    if any_prefixed and not all_prefixed:
        raise ValueError("checkpoint has an inconsistent module. prefix")
    return OrderedDict(
        (
            (str(key)[7:] if all_prefixed else str(key)),
            value,
        )
        for key, value in state_dict.items()
    )


def _extract_checkpoint_state(
    checkpoint: Mapping[str, object]
) -> Mapping[str, torch.Tensor]:
    for name in ("model", "iiw_planner_state_dict", "model_state_dict"):
        value = checkpoint.get(name)
        if isinstance(value, Mapping) and value:
            return value  # type: ignore[return-value]
    raise KeyError(
        "checkpoint has none of model, iiw_planner_state_dict, model_state_dict"
    )


def _coerce_config(
    config: Mapping[str, object],
    keys: Iterable[str],
) -> Dict[str, object]:
    result = {}
    for key in keys:
        if key in config:
            result[key] = config[key]
    return result


def _checkpoint_is_moe(
    checkpoint: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
) -> bool:
    markers = (
        checkpoint.get("planner_type"),
        checkpoint.get("model_type"),
        checkpoint.get("planner_class"),
    )
    config = checkpoint.get("model_config")
    if isinstance(config, Mapping):
        markers += (config.get("planner_type"),)
        if "num_experts" in config:
            return True
    if any("moe" in str(value).lower() for value in markers if value is not None):
        return True
    return any(
        str(key).startswith("router.") or str(key).startswith("expert_")
        for key in state_dict
    )


def build_iiw_planner(
    model_config: Mapping[str, object],
    planner_type: str = "dense",
    state_mean: Optional[Union[torch.Tensor, Iterable[float]]] = None,
    state_std: Optional[Union[torch.Tensor, Iterable[float]]] = None,
) -> Union[IIWPlanner, MoEIIWPlanner]:
    """Build a dense or MoE planner from a checkpoint-style config."""
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping")
    dense = _coerce_config(model_config, _DENSE_CONFIG_KEYS)
    kind = str(planner_type).lower()
    if "moe" not in kind:
        return _DENSE_PLANNER_CLASS(**dense)  # type: ignore[arg-type]
    moe = _coerce_config(model_config, _MOE_CONFIG_KEYS)
    return _MOE_PLANNER_CLASS(
        **dense,
        **moe,
        state_mean=state_mean,
        state_std=state_std,
    )  # type: ignore[arg-type]


def build_iiw_planner_from_checkpoint(
    checkpoint: Mapping[str, object],
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    migrate_dense_to_moe: bool = False,
    moe_overrides: Optional[Mapping[str, object]] = None,
) -> Union[IIWPlanner, MoEIIWPlanner]:
    """Strictly construct/load legacy dense or version-2 MoE checkpoints.

    ``migrate_dense_to_moe=True`` is the only path that changes architecture.
    It copies the dense trunk and leaves all residual expert outputs exactly
    zero.  Normal loading never silently converts a checkpoint.
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    config_value = checkpoint.get("model_config")
    if not isinstance(config_value, Mapping):
        raise TypeError("checkpoint has no model_config mapping")
    state = _strip_module_prefix(_extract_checkpoint_state(checkpoint))
    is_moe = _checkpoint_is_moe(checkpoint, state)
    config: MutableMapping[str, object] = dict(config_value)
    if moe_overrides is not None:
        if not isinstance(moe_overrides, Mapping):
            raise TypeError("moe_overrides must be a mapping")
        config.update(moe_overrides)

    target_device = torch.device(device) if device is not None else None
    if is_moe:
        planner_type = str(checkpoint.get("planner_type", "moe_iiw_v1"))
        mean = checkpoint.get("state_mean", state.get("state_mean"))
        std = checkpoint.get("state_std", state.get("state_std"))
        model = build_iiw_planner(config, planner_type, mean, std)
        if not isinstance(model, _MOE_PLANNER_CLASS):
            raise AssertionError("MoE checkpoint factory built a dense planner")
        if target_device is not None:
            model.to(target_device)
        model.load_state_dict(state, strict=strict)
        return model

    if migrate_dense_to_moe:
        mean = checkpoint.get("state_mean")
        std = checkpoint.get("state_std")
        if mean is None:
            mean = [0.0] * int(config.get("state_dim", 4))
        if std is None:
            std = [1.0] * int(config.get("state_dim", 4))
        model = build_iiw_planner(config, "moe_iiw_v1", mean, std)
        if not isinstance(model, _MOE_PLANNER_CLASS):
            raise AssertionError("dense migration factory built a dense planner")
        if target_device is not None:
            model.to(target_device)
        model.load_dense_state_dict(state, strict=strict)
        return model

    model = build_iiw_planner(config, "dense")
    if not isinstance(model, _DENSE_PLANNER_CLASS) or isinstance(
        model, _MOE_PLANNER_CLASS
    ):
        raise AssertionError("dense checkpoint factory built the wrong class")
    if target_device is not None:
        model.to(target_device)
    model.load_state_dict(state, strict=strict)
    return model


__all__ = (
    "NATIVE_BODY_PART_NAMES",
    "IIWPlanner",
    "MoEIIWPlanner",
    "build_iiw_planner",
    "build_iiw_planner_from_checkpoint",
)
