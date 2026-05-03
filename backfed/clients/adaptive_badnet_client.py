"""
Adaptive BadNet attack with norm clipping and direction-alignment evasion.

Extends MaliciousClient with post-training update manipulation to reduce
detectability by norm-based and representation-alignment defenses.

Evasion pipeline (applied after standard backdoor training):
  1. Compute update delta = client_params − global_params.
  2. Apply gradual ramping: scale delta by min(1, round / gradual_ramp_rounds).
  3. Clip delta norm to target_norm_scale × estimated benign norm.
  4. Blend delta direction with global model direction (alignment_weight controls
     the fraction blended toward the global direction).
  5. Reconstruct and return the modified state dict.

Configuration (via atk_config or Hydra override):
  norm_clip_percentile  – percentile of past delta norms used as clip target (default: 75)
  alignment_weight      – fraction of direction blended toward global model (default: 0.3)
  gradual_ramp_rounds   – rounds to ramp from zero to full strength (default: 10)
  target_norm_scale     – multiplier applied to the clip-target norm (default: 0.5)
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple
from logging import INFO, WARNING
from backfed.clients.base_malicious_client import MaliciousClient
from backfed.utils import log
from backfed.const import Metrics


class AdaptiveBadNetClient(MaliciousClient):
    """
    Adaptive BadNet attack with norm clipping and direction alignment evasion.
    
    Extends MaliciousClient with post-training evasion techniques to bypass
    norm-based and alignment-based defenses.
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
        client_type: str = "adaptive_badnet",
        verbose: bool = True,
        norm_clip_percentile: float = 75.0,
        alignment_weight: float = 0.3,
        gradual_ramp_rounds: int = 10,
        target_norm_scale: float = 0.5,
        **kwargs
    ):
        """
        Initialize Adaptive BadNet client.
        
        Args:
            client_id: Unique client identifier
            dataset: Client dataset
            model: Training model
            client_config: Client configuration
            atk_config: Attack configuration
            poison_module: Poison module (typically BadNets)
            context_actor: Context actor for synchronization
            client_type: Type identifier (default: "adaptive_badnet")
            verbose: Verbose logging flag
            norm_clip_percentile: Target percentile for norm clipping (0-100)
            alignment_weight: Weight for direction alignment (0-1)
            gradual_ramp_rounds: Rounds to reach full attack strength
            target_norm_scale: Base norm scale for clipping
            **kwargs: Additional arguments
        """
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
            **kwargs
        )

        self.norm_clip_percentile = norm_clip_percentile
        self.alignment_weight = alignment_weight
        self.gradual_ramp_rounds = gradual_ramp_rounds
        self.target_norm_scale = target_norm_scale
        self.norm_history = []
    
    def train(self, train_package: Dict[str, Any]) -> Tuple[int, Dict[str, torch.Tensor], Metrics]:
        num_examples, state_dict, metrics = super().train(train_package)
        server_round = train_package.get("server_round", 0)
        evaded_state_dict = self._apply_evasion(
            state_dict=state_dict,
            global_params=train_package["global_model_params"],
            server_round=server_round
        )
        return num_examples, evaded_state_dict, metrics
    
    def _apply_evasion(
        self,
        state_dict: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor],
        server_round: int
    ) -> Dict[str, torch.Tensor]:
        delta = self._compute_delta(state_dict, global_params)
        ramp_factor = min(1.0, server_round / max(1, self.gradual_ramp_rounds))
        delta_ramped = delta * ramp_factor
        delta_clipped = self._clip_norm(delta_ramped, server_round)
        delta_aligned = self._align_direction(delta_clipped, global_params, weight=self.alignment_weight)
        evaded_state_dict = self._apply_delta(global_params, delta_aligned)

        if self.verbose:
            original_norm = torch.linalg.norm(delta).item()
            final_norm = torch.linalg.norm(delta_aligned).item()
            log(INFO, f"Client [{self.client_id}] Round {server_round} evasion: "
                      f"ramp={ramp_factor:.2f}, norm {original_norm:.4f}→{final_norm:.4f}")
        return evaded_state_dict
    
    def _compute_delta(
        self,
        state_dict: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        delta_list = []
        for key in sorted(state_dict.keys()):
            if key in global_params:
                client_param = state_dict[key].to(self.device)
                global_param = global_params[key].to(self.device)
                delta_list.append((client_param - global_param).flatten())
        
        if not delta_list:
            return torch.tensor([], device=self.device)
        
        return torch.cat(delta_list)
    
    def _clip_norm(self, delta: torch.Tensor, server_round: int) -> torch.Tensor:
        current_norm = torch.linalg.norm(delta).item()
        if self.norm_history:
            target_norm = np.percentile(self.norm_history, self.norm_clip_percentile) * self.target_norm_scale
        else:
            target_norm = current_norm * self.target_norm_scale

        if current_norm > target_norm > 1e-9:
            delta_clipped = delta * (target_norm / current_norm)
        else:
            delta_clipped = delta

        self.norm_history.append(current_norm)
        if len(self.norm_history) > 50:
            self.norm_history.pop(0)
        return delta_clipped
    
    def _align_direction(
        self,
        delta: torch.Tensor,
        global_params: Dict[str, torch.Tensor],
        weight: float
    ) -> torch.Tensor:
        """Blend delta direction with global model direction to increase DAS score."""
        if weight <= 0.0 or weight >= 1.0:
            if weight >= 1.0:
                log(WARNING, f"Client [{self.client_id}] alignment_weight=1.0 disables the attack.")
            return delta

        global_flat = torch.cat([global_params[k].to(self.device).flatten()
                                  for k in sorted(global_params.keys())])
        global_dir = global_flat / torch.linalg.norm(global_flat).clamp(min=1e-9)

        delta_norm = torch.linalg.norm(delta).clamp(min=1e-9)
        blended = (1 - weight) * (delta / delta_norm) + weight * global_dir
        blended_norm = torch.linalg.norm(blended).clamp(min=1e-9)
        return (blended / blended_norm) * delta_norm
    
    def _apply_delta(
        self,
        global_params: Dict[str, torch.Tensor],
        delta: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct state dict as global_params + delta (param-aligned)."""
        state_dict = {}
        offset = 0
        for key in sorted(global_params.keys()):
            g = global_params[key].to(self.device)
            numel = g.numel()
            if offset + numel <= delta.numel():
                state_dict[key] = g + delta[offset:offset + numel].reshape(g.shape)
            else:
                state_dict[key] = g
            offset += numel
        return state_dict

