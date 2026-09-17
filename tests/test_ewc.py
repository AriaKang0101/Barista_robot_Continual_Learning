import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from barista_cl.ewc import EWCPPO, policy_fisher


class TinyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({"state": gym.spaces.Box(-10.0, 10.0, (2,), np.float32)})
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = self.np_random.normal(size=2).astype(np.float32)
        self.steps = 0
        return {"state": self.x.copy()}, {}

    def step(self, action):
        self.steps += 1
        self.x += float(action[0]) * 0.01
        return {"state": self.x.copy()}, -float((action[0] - 0.4)**2), False, self.steps >= 4, {}


def make_model(cls=EWCPPO, **kwargs):
    return cls("MultiInputPolicy", DummyVecEnv([TinyEnv]), n_steps=8, batch_size=4,
               n_epochs=2, seed=17, device="cpu", policy_kwargs={"net_arch": [8]}, **kwargs)


def test_lambda_zero_matches_unmodified_sb3():
    ordinary = make_model(PPO)
    ordinary.learn(16)
    reference = {k: v.clone() for k, v in ordinary.policy.state_dict().items()}
    ewc = make_model(ewc_lambda=0)
    ewc.learn(16)
    for k, v in ewc.policy.state_dict().items():
        torch.testing.assert_close(v, reference[k], rtol=0, atol=0)


def test_fisher_variance_score_not_squared_batch_mean():
    model = make_model()
    # State independent Gaussian mean => analytic F(mean)=1/sigma^2 and
    # F(log_sigma)=2. Repeated scores cancel if one squares a batch mean.
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.log_std.zero_()
    term = policy_fisher(model.policy, {"state": np.zeros((512, 2), np.float32)}, 512, seed=5)
    assert float(term["fisher"]["action_net.bias"]) == pytest.approx(1.0, abs=0.2)
    assert float(term["fisher"]["log_std"]) == pytest.approx(2.0, abs=0.6)
    assert not any("value_net" in k for k in term["fisher"])


def test_fisher_preserves_rng_and_model():
    model = make_model()
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    policy_fisher(model.policy, {"state": np.ones((8, 2), np.float32)}, 8, seed=123)
    assert torch.equal(torch_before, torch.get_rng_state())
    after = np.random.get_state()
    np.testing.assert_array_equal(numpy_before[1], after[1])
    for name, value in model.policy.state_dict().items():
        torch.testing.assert_close(before[name], value, rtol=0, atol=0)


def test_hooks_equal_explicit_loss_gradient_and_cleanup():
    model = make_model(ewc_lambda=2.5)
    model.learn(8)
    model.consolidate("A", 8)
    with torch.no_grad():
        for p in model.policy.parameters():
            p.add_(0.1)
    params = tuple(model.policy.parameters())
    explicit = torch.autograd.grad(model.penalty(), params, allow_unused=True)
    model.policy.zero_grad()
    with model.regularized_gradients():
        sum((p * 0).sum() for p in params).backward()
    for p, grad in zip(params, explicit):
        torch.testing.assert_close(p.grad, torch.zeros_like(p) if grad is None else grad)
        assert not p._backward_hooks
    with pytest.raises(RuntimeError):
        with model.regularized_gradients():
            raise RuntimeError("exercise cleanup")
    assert all(not p._backward_hooks for p in params)


def test_accumulation_checkpoint_and_training(tmp_path):
    model = make_model(ewc_lambda=1000)
    model.learn(8)
    model.consolidate("A", 8)
    first_anchor = {k: v.clone() for k, v in model.ewc_terms[0]["anchor"].items()}
    model.learn(8, reset_num_timesteps=False)
    model.consolidate("B", 8)
    assert [t["task_id"] for t in model.ewc_terms] == ["A", "B"]
    for k, v in first_anchor.items():
        torch.testing.assert_close(v, model.ewc_terms[0]["anchor"][k], rtol=0, atol=0)
    path = tmp_path / "checkpoint.zip"
    model.save(path)
    restored = EWCPPO.load(path, env=DummyVecEnv([TinyEnv]), device="cpu")
    assert restored.ewc_lambda == 1000
    assert len(restored.ewc_terms) == 2
    torch.testing.assert_close(restored.penalty(), model.penalty())
    restored.learn(8, reset_num_timesteps=False)
    assert restored.num_timesteps == 24
    assert np.isfinite(float(restored.penalty().detach()))

