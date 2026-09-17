"""Policy-Fisher EWC on top of the unmodified SB3 PPO update.

EWC gradient hooks run during backward(), BEFORE SB3's global gradient clipping.
This implements the gradient of L_SB3 + lambda/2 * sum(F*(theta-anchor)^2)
without copying or replacing SB3's PPO training loop. They are installed only
inside train() and removed even if training raises an exception.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from stable_baselines3 import PPO


@contextmanager
def preserve_rng():
    """Diagnostics/Fisher/evaluation must not consume the training RNG stream."""
    import random
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def policy_fisher(policy, observations, n_samples, seed):
    """Average per-example squared score gradients, NOT squared mean gradients.

    States come from the final CURRENT task rollout (already counted in budget).
    Actions are newly sampled from the consolidated policy, detached, and NOT
    clipped before log_prob. Thus this estimates that policy's diagonal Fisher
    on the final rollout's empirical state distribution. Value-only parameters
    have no likelihood gradient and are excluded.
    """
    if not observations:
        raise ValueError("Empty observation dictionary")
    length = len(next(iter(observations.values())))
    if not 0 < n_samples <= length:
        raise ValueError("Invalid Fisher sample count")
    parameters = dict(policy.named_parameters())
    sums = {}
    prior_mode = policy.training
    with preserve_rng():
        torch.manual_seed(seed)
        indices = np.random.default_rng(seed).choice(length, n_samples, replace=False)
        policy.set_training_mode(False)
        try:
            for i in indices:
                sample = {k: v[i:i + 1] for k, v in observations.items()}
                obs, _ = policy.obs_to_tensor(sample)
                distribution = policy.get_distribution(obs)
                action = distribution.sample().detach()
                score = distribution.log_prob(action).sum()
                grads = torch.autograd.grad(score, tuple(parameters.values()), allow_unused=True)
                for (name, _), grad in zip(parameters.items(), grads):
                    if grad is not None:
                        square = grad.detach().square().cpu()
                        if name not in sums:
                            sums[name] = torch.zeros_like(square)
                        sums[name].add_(square)
        finally:
            policy.set_training_mode(prior_mode)
    if not sums or not all(torch.isfinite(v).all() for v in sums.values()):
        raise RuntimeError("Invalid policy Fisher estimate")
    return {
        "fisher": {k: v / n_samples for k, v in sums.items()},
        "anchor": {k: parameters[k].detach().cpu().clone() for k in sums},
        "n_samples": n_samples,
    }


class EWCPPO(PPO):
    def __init__(self, *args, ewc_lambda=0.0, **kwargs):
        self.ewc_lambda = float(ewc_lambda)
        self.ewc_terms = []
        super().__init__(*args, **kwargs)

    def consolidate(self, task_id, n_samples=128, seed=0):
        if any(t["task_id"] == task_id for t in self.ewc_terms):
            raise ValueError(f"Task {task_id} already consolidated")
        if not self.rollout_buffer.full:
            raise RuntimeError("Consolidation requires a completed rollout")
        buffer = self.rollout_buffer
        if not isinstance(buffer.observations, dict):
            raise TypeError("This implementation expects a Dict observation space")
        # SB3 flattens the time/env dimensions during its first get() call.
        observations = {k: v if buffer.generator_ready else buffer.swap_and_flatten(v)
                        for k, v in buffer.observations.items()}
        term = policy_fisher(self.policy, observations, n_samples, seed)
        term["task_id"] = task_id
        self.ewc_terms.append(term)

    def _device_terms(self):
        params = dict(self.policy.named_parameters())
        grouped = {}
        for term in self.ewc_terms:
            for name, fisher in term["fisher"].items():
                p = params[name]
                grouped.setdefault(name, []).append((fisher.to(p), term["anchor"][name].to(p)))
        return params, grouped

    def penalty(self):
        params, terms = self._device_terms()
        value = torch.zeros((), device=self.device)
        for name, items in terms.items():
            for fisher, anchor in items:
                value = value + 0.5 * self.ewc_lambda * (fisher * (params[name] - anchor).square()).sum()
        return value

    @contextmanager
    def regularized_gradients(self):
        handles = []
        if self.ewc_lambda and self.ewc_terms:
            params, terms = self._device_terms()
            for name, items in terms.items():
                parameter = params[name]

                def augment(gradient, parameter=parameter, items=items):
                    correction = torch.zeros_like(parameter)
                    for fisher, anchor in items:
                        correction.add_(fisher * (parameter.detach() - anchor))
                    return gradient + self.ewc_lambda * correction

                handles.append(parameter.register_hook(augment))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def train(self):
        with self.regularized_gradients():
            super().train()
        # SB3 train/loss remains its own objective; this metric reports the
        # additional penalty after the update, not a minibatch total loss.
        self.logger.record("ewc/penalty_after_update", float(self.penalty().detach()))
        self.logger.record("ewc/consolidated_tasks", len(self.ewc_terms))

