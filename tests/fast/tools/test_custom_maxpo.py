import itertools

import numpy as np
import pytest

from examples.multi_lora.custom_maxpo import compute_mean_maxpo_advantages


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_advantages_match_explicit_subset_maxima(k: int, weight: float) -> None:
    rewards = np.array([0.9, 0.0, 0.5, 0.5])
    actual, metrics = compute_mean_maxpo_advantages(rewards, k, weight)
    subsets = list(itertools.combinations(range(len(rewards)), k))
    expected = []
    for i in range(len(rewards)):
        included = np.mean([max(rewards[list(s)]) for s in subsets if i in s])
        excluded = np.mean([max(rewards[list(s)]) for s in subsets if i not in s])
        mean = rewards[i] - np.delete(rewards, i).mean()
        expected.append(weight * mean + (1 - weight) * k * (included - excluded))
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert actual.mean() == pytest.approx(0.0, abs=1e-14)
    assert metrics["maxpo/expected_max_reward"] == pytest.approx(np.mean([max(rewards[list(s)]) for s in subsets]))


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_advantages_recover_the_objective_gradient(weight: float) -> None:
    probabilities = np.array([0.2, 0.3, 0.5])
    reward_values = np.array([0.0, 0.3, 1.0])
    estimated = np.zeros(3)
    exact = np.zeros(3)
    for actions in itertools.product(range(3), repeat=3):
        probability = np.prod(probabilities[list(actions)])
        rewards = reward_values[list(actions)]
        scores = np.eye(3)[list(actions)] - probabilities
        advantages, _ = compute_mean_maxpo_advantages(rewards, 2, weight)
        objective = weight * rewards.mean() + (1 - weight) * np.mean([max(rewards[list(s)]) for s in itertools.combinations(range(3), 2)])
        estimated += probability * (advantages[:, None] * scores).mean(axis=0)
        exact += probability * objective * scores.sum(axis=0)
    np.testing.assert_allclose(estimated, exact, atol=1e-14)


@pytest.mark.parametrize("k", [4, 64, 127])
def test_large_groups_preserve_centering_and_equal_reward_ties(k: int) -> None:
    rewards = np.linspace(0, 1, 128)
    actual, _ = compute_mean_maxpo_advantages(rewards, k, 0.5)
    assert actual.mean() == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(actual).all()
    tied, _ = compute_mean_maxpo_advantages(np.full(128, 0.5), k, 0.5)
    np.testing.assert_allclose(tied, 0.0, atol=1e-12)


@pytest.mark.parametrize("k", [0, 3, 4])
def test_invalid_comparison_size_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="group size"):
        compute_mean_maxpo_advantages(np.array([0.0, 0.5, 1.0]), k, 0.5)


def test_callback_matches_importance_sampling_surrogate_and_gradients():
    import torch
    from examples.multi_lora.custom_maxpo import make_maxpo_loss, maxpo_loss_inputs

    rewards = [0.0, 0.5, 1.0, 0.5]
    old = [torch.tensor([-1.0, -2.0, -3.0]) for _ in rewards]
    masks = [torch.tensor([False, True, True]) for _ in rewards]
    inputs, _ = maxpo_loss_inputs(rewards, old, masks, k=2)
    callback = make_maxpo_loss(rewards, old, masks, k=2)
    current = [x.clone().add(0.05).requires_grad_() for x in old]
    actual, _ = callback([None] * 4, current)
    expected = -sum(((lp - x["logprobs"]).exp() * x["advantages"]).sum() for lp, x in zip(current, inputs, strict=True))
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, current, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, current)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b)
        assert a[0] == 0


def test_callback_detaches_rollout_and_handles_zero_advantages():
    import torch
    from examples.multi_lora.custom_maxpo import make_maxpo_loss

    old = [torch.tensor([-1.0], requires_grad=True) for _ in range(3)]
    loss_fn = make_maxpo_loss([1.0] * 3, old, [torch.ones(1)] * 3, k=2)
    current = [torch.tensor([-0.5], requires_grad=True) for _ in range(3)]
    loss, _ = loss_fn([None] * 3, current)
    loss.backward()
    assert loss.item() == 0
    assert all(x.grad is None for x in old)
    assert all(x.grad.item() == 0 for x in current)
