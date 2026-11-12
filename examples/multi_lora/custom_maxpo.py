"""Centered max@K advantages via conditional and leave-one-out expected maxima."""

import math

import numpy as np
import torch


def compute_mean_maxpo_advantages(rewards: np.ndarray, k: int, mean_weight: float) -> tuple[np.ndarray, dict[str, float]]:
    """Mix mean and max@K policy gradients with coefficients for a batch-mean loss."""
    n = len(rewards)
    if not 1 <= k < n:
        raise ValueError("MaxPO requires 1 <= k < group size")
    if not np.isfinite(rewards).all() or not 0 <= mean_weight <= 1:
        raise ValueError("MaxPO requires finite rewards and a mean weight in [0, 1]")
    order = np.argsort(rewards, kind="stable")
    ranked = rewards[order]
    included = np.zeros(n)
    excluded = np.zeros(n)
    include_denominator = math.comb(n - 1, k - 1)
    exclude_denominator = math.comb(n - 1, k)
    for i in range(n):
        included[i] = ranked[i] * (math.comb(i, k - 1) / include_denominator)
        for j in range(n):
            if j == i:
                continue
            lower = j - int(i < j)
            excluded[i] += ranked[j] * (math.comb(lower, k - 1) / exclude_denominator)
            if j > i and k > 1:
                included[i] += ranked[j] * (math.comb(j - 1, k - 2) / include_denominator)
    max_advantages = k * (included - excluded)
    mean_advantages = (ranked - ranked.mean()) * n / (n - 1)
    advantages = np.empty(n)
    advantages[order] = mean_weight * mean_advantages + (1 - mean_weight) * max_advantages
    expected_max = float(included.mean())
    return advantages, {
        "maxpo/k": float(k),
        "maxpo/group_size": float(n),
        "maxpo/mean_weight": mean_weight,
        "maxpo/expected_max_reward": expected_max,
        "maxpo/joint_reward": mean_weight * float(rewards.mean()) + (1 - mean_weight) * expected_max,
        "maxpo/mean_advantage_std": float(mean_advantages.std()),
        "maxpo/max_advantage_std": float(max_advantages.std()),
        "maxpo/advantage_mean": float(advantages.mean()),
        "maxpo/advantage_std": float(advantages.std()),
    }


def maxpo_loss_inputs(rewards, old_logprobs, response_masks, *, k=64, mean_weight=0.5):
    """Build inputs for Miles' native importance_sampling loss.

    Use one complete prompt group per call. Concatenate the resulting inputs
    across groups and divide advantages by the number of groups if needed.
    Each selected token gets A_i / group_size, so the on-policy gradient uses
    the full sequence score (a sum, not a length-normalized mean).
    """
    advantages, metrics = compute_mean_maxpo_advantages(np.asarray(rewards, dtype=float), k, mean_weight)
    if len(old_logprobs) != len(advantages) or len(response_masks) != len(advantages):
        raise ValueError("Supply one logprob vector and response mask per reward")
    inputs = []
    for old, mask, advantage in zip(old_logprobs, response_masks, advantages, strict=True):
        old = torch.as_tensor(old).detach().clone()
        mask = torch.as_tensor(mask, dtype=torch.bool, device=old.device)
        if old.ndim != 1 or old.shape != mask.shape or not mask.any() or not torch.isfinite(old[mask]).all():
            raise ValueError("Response masks must select finite logprobs from matching 1-D vectors")
        inputs.append(
            {
                "logprobs": old,
                "advantages": mask.to(old.dtype) * float(advantage) / len(advantages),
            }
        )
    return inputs, metrics


def make_maxpo_loss(rewards, old_logprobs, response_masks, *, k=64, mean_weight=0.5):
    """Tinker callback matching the native token importance_sampling surrogate.

    Native: forward_backward(datums, loss_fn="importance_sampling").
    Custom: forward_backward_custom(datums, callback).
    This uses token importance sampling, without PPO/GSPO clipping or KL loss.
    It is an on-policy MaxPO policy-gradient estimator, not an exact
    off-policy sequence importance estimator.
    """
    inputs, metrics = maxpo_loss_inputs(rewards, old_logprobs, response_masks, k=k, mean_weight=mean_weight)

    def loss_fn(data, logprobs):
        if len(data) != len(inputs) or len(logprobs) != len(inputs):
            raise ValueError("The returned logprobs must match the captured prompt group")
        terms = []
        for current, captured in zip(logprobs, inputs, strict=True):
            old = captured["logprobs"].to(current)
            weights = captured["advantages"].to(current)
            if current.shape != old.shape:
                raise ValueError("The model must return matching logprob vectors")
            selected = weights != 0
            terms.append(-((current[selected] - old[selected]).exp() * weights[selected]).sum())
        loss = torch.stack(terms).sum()
        return loss, {**metrics, "maxpo/loss": loss.detach().item()}

    return loss_fn
