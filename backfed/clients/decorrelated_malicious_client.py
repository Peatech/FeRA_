"""
Decorrelated Backdoor Attack — Gradient-Decorrelation Evasion

Addresses Review B, Question 2: "If attackers add a regularization term to
minimize cosine similarity between their updates, can they bypass the
Consistency Filter entirely?"

ATTACK IDEA
-----------
FeRA's Consistency Filter (default_filter) flags clients whose updates are
BOTH:
  (a) poorly aligned with the global-model direction  (low TDA), AND
  (b) highly similar to other sampled clients         (high mutual_sim)

FeRA's Collusion Filter flags clients whose updates are BOTH:
  (a) well aligned with the global-model direction    (high TDA), AND
  (b) highly similar to other sampled clients         (high mutual_sim)

The common condition across both filters is HIGH mutual_sim.  An attacker who
can suppress their pairwise cosine similarity with every other client could
evade both filters simultaneously.

EVASION STRATEGY
----------------
This client adds two mechanisms on top of standard backdoor training:

1. **Gradient decorrelation** (training time):
   After every backward pass, the accumulated gradients are projected to be
   orthogonal to the global model's parameter vector.  This drives the local
   update delta away from the global direction, lowering TDA and making the
   malicious update direction less predictable / less correlated.

2. **Random-subspace post-processing** (post-training):
   After local training the final update delta is further randomised by adding
   a small perturbation in a random direction.  Because each malicious client
   draws an independent random direction each round, their pairwise cosine
   similarities are reduced without coordination.  The perturbation magnitude
   is controlled by `random_scale`.

PARAMETERS
----------
decor_lambda : float  [0, ∞)
    Strength of gradient decorrelation.  At each backward step the gradient is
    modified as:
        g ← g − decor_lambda * proj_g(global_direction)
    0.0 → no modification (vanilla backdoor)
    1.0 → full projection (gradient made orthogonal to global direction)
    Values > 1.0 over-project (gradient component in global direction negated).

random_scale : float  [0, 1]
    Magnitude of the random perturbation added post-training, expressed as a
    fraction of the update delta norm.  0.0 disables random noise.

keep_norm : bool
    If True, the final update delta is rescaled to match the original delta
    norm after decorrelation (preserves attack magnitude).

norm_clip_percentile / target_norm_scale
    Inherited from AdaptiveBadNetClient for optional norm clipping.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple
from logging import INFO
from backfed.clients.adaptive_badnet_client import AdaptiveBadNetClient
from backfed.utils import log
from backfed.const import Metrics


class DecorrelatedBadNetClient(AdaptiveBadNetClient):
    """
    Backdoor client that suppresses cosine similarity between its update and
    the global model direction, and randomises its direction to break
    pairwise mutual similarity across malicious clients.
    """

    def __init__(
        self,
        client_id,
        dataset,
        model,
        client_config,
        atk_config,
        poison_module,
        context_actor,
        client_type: str = "decorrelated_badnet",
        verbose: bool = True,
        # Decorrelation parameters
        decor_lambda: float = 1.0,
        random_scale: float = 0.05,
        keep_norm: bool = True,
        # Inherited norm-clipping from AdaptiveBadNetClient (disabled by default)
        norm_clip_percentile: float = 100.0,
        alignment_weight: float = 0.0,
        gradual_ramp_rounds: int = 0,
        target_norm_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            model=model,
            client_config=client_config,
            atk_config=atk_config,
            poison_module=poison_module,
            context_actor=context_actor,
            client_type=client_type,
            verbose=verbose,
            norm_clip_percentile=norm_clip_percentile,
            alignment_weight=alignment_weight,
            gradual_ramp_rounds=gradual_ramp_rounds,
            target_norm_scale=target_norm_scale,
            **kwargs
        )
        self.decor_lambda = decor_lambda
        self.random_scale = random_scale
        self.keep_norm = keep_norm

        # Pre-compute (lazily) the flattened global direction for the round
        self._global_flat: torch.Tensor | None = None

        log(INFO, f"Client [{self.client_id}] DecorrelatedBadNet | "
                  f"decor_lambda={decor_lambda:.2f}  random_scale={random_scale:.3f}  "
                  f"keep_norm={keep_norm}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, train_package: Dict[str, Any]) -> Tuple[int, Dict[str, torch.Tensor], Metrics]:
        """Backdoor train with gradient decorrelation, then post-process."""
        # Cache the flattened global direction so _hook can use it
        self._global_flat = self._flatten_params(train_package["global_model_params"])

        # Register gradient hooks on all parameters if lambda > 0
        hooks = []
        if self.decor_lambda > 0.0:
            hooks = self._register_decor_hooks()

        # Standard backdoor training (via MaliciousClient)
        num_examples, state_dict, metrics = super(AdaptiveBadNetClient, self).train(train_package)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Post-processing: random-subspace perturbation + optional norm clip
        server_round = train_package.get("server_round", 0)
        state_dict = self._post_process(
            state_dict,
            train_package["global_model_params"],
            server_round,
        )

        return num_examples, state_dict, metrics

    # ------------------------------------------------------------------
    # Gradient decorrelation hooks
    # ------------------------------------------------------------------

    def _register_decor_hooks(self):
        """
        Register backward hooks that project each parameter gradient to be
        orthogonal to the corresponding slice of the global model direction.
        """
        global_flat = self._global_flat  # full vector, read-only
        hooks = []
        offset = 0
        for name, param in self.model.named_parameters():
            numel = param.numel()
            g_slice = global_flat[offset: offset + numel].reshape(param.shape).detach()
            g_norm_sq = (g_slice ** 2).sum().clamp(min=1e-12)
            offset += numel

            def make_hook(gs, gns):
                def hook(grad):
                    if grad is None:
                        return None
                    proj = (grad * gs).sum() / gns
                    return grad - self.decor_lambda * proj * gs
                return hook

            h = param.register_hook(make_hook(g_slice, g_norm_sq))
            hooks.append(h)
        return hooks

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process(
        self,
        state_dict: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor],
        server_round: int,
    ) -> Dict[str, torch.Tensor]:
        """
        1. Compute delta = state_dict − global_params.
        2. Add random perturbation (random_scale * ||delta||) to break
           pairwise mutual_sim across independently-acting malicious clients.
        3. Optionally rescale to original delta norm (keep_norm=True).
        4. Reconstruct state dict.
        """
        delta = self._compute_delta(state_dict, global_params)
        original_norm = torch.linalg.norm(delta).item()

        if self.random_scale > 0.0 and original_norm > 1e-9:
            noise = torch.randn_like(delta)
            noise = noise / torch.linalg.norm(noise).clamp(min=1e-12)
            delta = delta + self.random_scale * original_norm * noise

        if self.keep_norm and original_norm > 1e-9:
            current_norm = torch.linalg.norm(delta).clamp(min=1e-12)
            delta = delta * (original_norm / current_norm)

        if self.verbose:
            final_norm = torch.linalg.norm(delta).item()
            # Cosine similarity with global direction
            g = self._global_flat
            cos = (delta @ g / (torch.linalg.norm(delta) * torch.linalg.norm(g) + 1e-12)).item()
            log(INFO, f"Client [{self.client_id}] round {server_round} post-process | "
                      f"‖δ‖={final_norm:.4f}  cos(δ,global)={cos:.4f}")

        return self._apply_delta(global_params, delta)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_params(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.to(self.device).flatten() for k, v in sorted(params.items())])
