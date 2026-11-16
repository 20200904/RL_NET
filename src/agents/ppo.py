"""PPO for discrete actions, written by hand in the single-file CleanRL style.

Pieces: rollout collection from a few in-process environments, GAE(lambda)
advantages, the clipped surrogate objective, a value loss and an entropy
bonus. No library does the learning; SB3 is only used elsewhere to check
that this implementation learns as well as a reference one.

``cost_penalty`` > 0 turns it into the fixed-penalty baseline (M2):
the training reward becomes ``reward - cost_penalty * cost``. Logged returns
are always the raw environment reward.
"""
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

EPISODE_FIELDS = [
    "global_step", "wall_s", "network", "ep_return", "ep_len", "solved",
    "cost", "cost_c1", "cost_c2",
]


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    num_envs: int = 4
    num_steps: int = 128          # rollout length per environment
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64              # MLP 64x64, not to be increased
    cost_penalty: float = 0.0
    seed: int = 0


def _layer(in_dim, out_dim, std=np.sqrt(2)):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Agent(nn.Module):
    """Separate 64x64 tanh MLPs for the policy and the value function."""

    def __init__(self, obs_dim, num_actions, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, num_actions, std=0.01),
        )
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, 1, std=1.0),
        )

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        dist = Categorical(logits=self.actor(obs))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.get_value(obs)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits = self.actor(torch.as_tensor(obs, dtype=torch.float32))
        if deterministic:
            return int(torch.argmax(logits))
        return int(Categorical(logits=logits).sample())


def train(env_fns, cfg, run_dir=None):
    """Train PPO. ``env_fns`` is a list of ``cfg.num_envs`` env constructors.

    Writes ``episodes.csv`` (one row per finished training episode) and
    ``model.pt`` into ``run_dir`` when it is given. Returns (agent, summary).
    """
    torch.set_num_threads(1)
    envs = [fn() for fn in env_fns]
    assert len(envs) == cfg.num_envs
    # seed after building the envs: scenario generation reseeds numpy
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    obs_dim = envs[0].observation_space.shape[0]
    num_actions = int(envs[0].action_space.n)
    agent = Agent(obs_dim, num_actions, cfg.hidden)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    n_envs, n_steps = cfg.num_envs, cfg.num_steps
    batch_size = n_envs * n_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = cfg.total_steps // batch_size

    obs_buf = torch.zeros((n_steps, n_envs, obs_dim))
    act_buf = torch.zeros((n_steps, n_envs), dtype=torch.long)
    logp_buf = torch.zeros((n_steps, n_envs))
    rew_buf = torch.zeros((n_steps, n_envs))
    done_buf = torch.zeros((n_steps, n_envs))
    val_buf = torch.zeros((n_steps, n_envs))

    episodes = []
    ep_ret = np.zeros(n_envs)
    ep_len = np.zeros(n_envs, dtype=int)
    ep_cost = np.zeros((n_envs, 2), dtype=int)
    ep_net = np.zeros(n_envs, dtype=int)

    next_obs = np.zeros((n_envs, obs_dim), dtype=np.float32)
    for i, env in enumerate(envs):
        next_obs[i], info = env.reset(seed=cfg.seed * 1000 + i)
        ep_net[i] = info.get("network", 0)
    next_obs = torch.as_tensor(next_obs)
    next_done = torch.zeros(n_envs)

    global_step = 0
    start = time.perf_counter()
    for update in range(num_updates):
        if cfg.anneal_lr:
            frac = 1.0 - update / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # ---- collect a rollout ----
        for step in range(n_steps):
            global_step += n_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value

            new_obs = np.zeros((n_envs, obs_dim), dtype=np.float32)
            for i, env in enumerate(envs):
                obs, reward, terminated, truncated, info = env.step(int(action[i]))
                cost = info.get("cost", 0)
                ep_ret[i] += reward
                ep_len[i] += 1
                ep_cost[i] += (info.get("cost_c1", 0), info.get("cost_c2", 0))
                rew_buf[step, i] = float(reward) - cfg.cost_penalty * cost
                done = terminated or truncated
                if done:
                    episodes.append(dict(
                        global_step=global_step,
                        wall_s=round(time.perf_counter() - start, 2),
                        network=int(ep_net[i]),
                        ep_return=float(ep_ret[i]),
                        ep_len=int(ep_len[i]),
                        solved=int(terminated),
                        cost=int(ep_cost[i].sum()),
                        cost_c1=int(ep_cost[i, 0]),
                        cost_c2=int(ep_cost[i, 1]),
                    ))
                    ep_ret[i], ep_len[i], ep_cost[i] = 0.0, 0, 0
                    obs, info = env.reset()
                    ep_net[i] = info.get("network", 0)
                new_obs[i] = obs
                next_done[i] = float(done)
            next_obs = torch.as_tensor(new_obs)

        # ---- GAE(lambda) advantages, computed backwards through the rollout ----
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            adv_buf = torch.zeros_like(rew_buf)
            last_gae = torch.zeros(n_envs)
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    not_done, value_next = 1.0 - next_done, next_value
                else:
                    not_done, value_next = 1.0 - done_buf[t + 1], val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * value_next * not_done - val_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * not_done * last_gae
                adv_buf[t] = last_gae
            ret_buf = adv_buf + val_buf

        b_obs = obs_buf.reshape(batch_size, obs_dim)
        b_act = act_buf.reshape(batch_size)
        b_logp = logp_buf.reshape(batch_size)
        b_adv = adv_buf.reshape(batch_size)
        b_ret = ret_buf.reshape(batch_size)

        # ---- policy and value update ----
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(batch_size)
            for lo in range(0, batch_size, minibatch_size):
                idx = perm[lo:lo + minibatch_size]
                _, new_logp, entropy, new_value = agent.get_action_and_value(
                    b_obs[idx], b_act[idx]
                )
                ratio = (new_logp - b_logp[idx]).exp()
                adv = b_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                # clipped surrogate: take the more pessimistic of the two
                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()
                v_loss = 0.5 * ((new_value - b_ret[idx]) ** 2).mean()
                loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

    wall = time.perf_counter() - start
    last = episodes[-100:]
    summary = dict(
        config=asdict(cfg),
        train_steps=global_step,
        train_wall_s=round(wall, 1),
        train_steps_per_sec=round(global_step / wall, 1) if wall > 0 else 0.0,
        train_episodes=len(episodes),
        last100_solve_rate=float(np.mean([e["solved"] for e in last])) if last else None,
        last100_ep_len=float(np.mean([e["ep_len"] for e in last])) if last else None,
        last100_cost=float(np.mean([e["cost"] for e in last])) if last else None,
    )
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            writer.writerows(episodes)
        torch.save(agent.state_dict(), run_dir / "model.pt")
        (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return agent, summary
