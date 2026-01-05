from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple, Callable
import random
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore

@dataclass
class Transition:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    next_mask: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, obs, mask, action, reward, next_obs, next_mask, done):
        self.buf.append(Transition(obs, mask, action, reward, next_obs, next_mask, done))


    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size) 
        obs = np.stack([t.obs for t in batch])
        mask = np.stack([t.mask for t in batch])
        actions = np.array([t.action for t in batch], np.int64)
        rewards = np.array([t.reward for t in batch], np.float32)
        next_obs = np.stack([t.next_obs for t in batch])
        next_mask = np.stack([t.next_mask for t in batch])
        dones = np.array([t.done for t in batch], np.float32)
        return obs, mask, actions, rewards, next_obs, next_mask, dones



class QNet(nn.Module):
    def __init__(self, in_channels: int, h: int, w: int, n_actions: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            flat = self.conv(dummy).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x)

class DQNAgent:
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        n_actions: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 150_000,
        target_update_every: int = 2000,
        device: Optional[str] = None,
    ):
        c, h, w = obs_shape
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q = QNet(c, h, w, n_actions).to(self.device)
        self.q_tgt = QNet(c, h, w, n_actions).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

        self.rb = ReplayBuffer(buffer_capacity)
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.learn_steps = 0
        self.target_update_every = target_update_every

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, eps: float) -> int:
        # mask: (A,) 0/1
        legal = np.flatnonzero(mask > 0.0)
        if legal.size == 0:
            return 0  # awaryjnie

        if random.random() < eps:
            return int(np.random.choice(legal))

        x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)      # (1,C,H,W)
        qvals = self.q(x).squeeze(0)                                         # (A,)

        # maskowanie: nielegalne -> -inf
        m = torch.from_numpy(mask).float().to(self.device)
        qvals = torch.where(m > 0.0, qvals, torch.tensor(-1e9, device=self.device))

        return int(torch.argmax(qvals).item())


    def push(self, obs, mask, action, reward, next_obs, next_mask, done):
        self.rb.push(obs, mask, action, reward, next_obs, next_mask, done)


    def can_learn(self) -> bool:
        return len(self.rb) >= self.batch_size

    def learn(self) -> float:
        obs, mask, actions, rewards, next_obs, next_mask, dones = self.rb.sample(self.batch_size)

        obs_t = torch.from_numpy(obs).float().to(self.device)
        mask_t = torch.from_numpy(mask).float().to(self.device)               # (B,A)
        next_obs_t = torch.from_numpy(next_obs).float().to(self.device)
        next_mask_t = torch.from_numpy(next_mask).float().to(self.device)

        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        q_sa = self.q(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # online wybiera akcję (argmax) ale tylko legalną w next_mask
            q_online_next = self.q(next_obs_t)                                 # (B,A)
            q_online_next = torch.where(next_mask_t > 0.0, q_online_next, torch.tensor(-1e9, device=self.device))
            next_actions = q_online_next.argmax(dim=1, keepdim=True)           # (B,1)

            # target sieć ocenia wybraną akcję
            q_tgt_next = self.q_tgt(next_obs_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * q_tgt_next * (1.0 - dones_t)


        loss = self.loss_fn(q_sa, target)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optim.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        return float(loss.item())

def train_dqn(
    env,
    episodes: int = 3000,
    max_steps: int = 150,
    warmup_steps: int = 10000,
    epsilon_decay_steps: int = 150_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    learn_every: int = 1,
    on_log: Optional[Callable[[int, int, float, float, float], None]] = None,
):
    """
    on_log(ep, total_steps, eps, avg_return, solved_rate)
    """
    obs0, mask0 = env.reset()
    agent = DQNAgent(obs_shape=obs0.shape, n_actions=mask0.shape[0])

    total_steps = 0
    solved = 0
    rets = []

    def eps_by_step(t: int) -> float:
        if t >= epsilon_decay_steps:
            return eps_end
        frac = t / float(epsilon_decay_steps)
        return eps_start + frac * (eps_end - eps_start)

    for ep in range(1, episodes + 1):
        obs, mask = env.reset()

        ep_ret = 0.0
        for _ in range(max_steps):
            eps = eps_by_step(total_steps)
            a = agent.act(obs, mask, eps)
            out, next_mask = env.step(a)

            agent.push(obs, mask, a, out.reward, out.obs, next_mask, out.done)

            obs, mask = out.obs, next_mask

            ep_ret += out.reward
            total_steps += 1

            if total_steps >= warmup_steps and agent.can_learn() and (total_steps % learn_every == 0):
                agent.learn()

            if out.done:
                break

        rets.append(ep_ret)
        if env.level.is_solved(env.state):
            solved += 1

        if on_log and (ep % 50 == 0):
            avg_ret = float(sum(rets[-50:]) / max(1, len(rets[-50:])))
            solved_rate = solved / ep
            on_log(ep, total_steps, eps_by_step(total_steps), avg_ret, solved_rate)

    return agent

@torch.no_grad()
def play_dqn(env, agent: DQNAgent, max_steps: int = 150) -> list[int]:
    moves_all = []
    obs, mask = env.reset()
    for _ in range(max_steps):
        a = agent.act(obs, mask, eps=0.0)
        out, next_mask = env.step(a)
        moves_all.extend(out.info.get("moves", []))
        obs, mask = out.obs, next_mask
        if out.done:
            break
    return moves_all

