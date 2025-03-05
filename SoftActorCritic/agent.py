import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from networks import PolicyNetwork, QNetwork

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_layers, 
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, device="cpu"):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Actor (policy) network.
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_layers).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin Critic networks.
        self.critic1 = QNetwork(state_dim, action_dim, hidden_layers).to(device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_layers).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Target networks.
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_layers).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_layers).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Automatic entropy tuning.
        self.target_entropy = -float(action_dim)  # heuristic: -|A|
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        return self.actor.get_action(state, deterministic)

    def update(self, replay_buffer, batch_size):
        # Sample a batch.
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state      = state.to(self.device)
        action     = action.to(self.device)
        reward     = reward.to(self.device)
        next_state = next_state.to(self.device)
        done       = done.to(self.device)

        # --- Update Critics ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_log_prob
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        # --- Update Policy ---
        new_action, log_prob, _ = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # --- Adjust temperature alpha ---
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Polyak averaging for target networks ---
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item()
        }