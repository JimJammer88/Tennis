import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from configuration import Configuration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SingleAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, agent_number,num_agents,state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_number = agent_number
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.config = Configuration()

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, self.config.actor_fc1, self.config.actor_fc2).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, self.config.actor_fc1, self.config.actor_fc2).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size*num_agents, action_size*num_agents, random_seed, self.config.critic_fc1, self.config.critic_fc2).to(device)
        self.critic_target = Critic(state_size*num_agents, action_size*num_agents, random_seed, self.config.critic_fc1, self.config.critic_fc2).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.config.lr_critic, weight_decay=self.config.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, mu = self.config.ou_mu, theta = self.config.ou_theta, sigma = self.config.ou_sigma)

        # Replay memory will be shared
#         # Replay memory
#         self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
#     def step(self, state, action, reward, next_state, done):
#         """Save experience in replay memory, and use random sample from buffer to learn."""
#         # Save experience / reward
#         self.memory.add(state, action, reward, next_state, done)

#         # Learn, if enough samples are available in memory
#         if len(self.memory) > BATCH_SIZE:
#             experiences = self.memory.sample()
#             self.learn(experiences, GAMMA)

    def get_number(self):
        return self.agent_number

    def act(self, state, epsilon, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += epsilon * self.noise.sample()
        #return np.clip(action, -1, 1)
        return action

    def reset(self):
        self.noise.reset()
        
    def action_pred_target(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states_local = states[:,self.agent_number,:]
        return self.actor_target(next_states_local).detach().numpy()
        
    def actions_next_target(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_states_local = next_states[:,self.agent_number,:]
        return self.actor_target(next_states_local).detach().numpy()
    
    def actions_next_local(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_states_local = next_states[:,self.agent_number,:]
        return self.actor_target(next_states_local).detach().numpy()

    def learn(self, experiences, all_pred_actions, all_next_actions, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        states_local = states[:,self.agent_number,:]
        actions_local = actions[:,self.agent_number,:]
        rewards_local = rewards[:,self.agent_number]
        next_states_local = next_states[:,self.agent_number,:]
        dones_local = dones[:,self.agent_number]
        
        
#         q_input_actions = actions
#         with torch.no_grad():
#             q_input_actions[:,self.agent_number,:] = self.actor_local(states_local)
        states_global = torch.reshape(states,(-1,self.num_agents*self.state_size))
        actions_global = torch.reshape(actions,(-1,self.num_agents*self.action_size))
        next_states_global = torch.reshape(next_states,(-1,self.num_agents*self.state_size))
#         pred = [actions if i == self.index else actions.detach() for i, actions in enumerate(all_pred_actions)]
#         pred_actions = torch.cat(all_pred_actions, dim=1).to(device)
        next_actions = torch.cat(all_next_actions, dim=1).to(device)
        
#         next_actions_t_global = torch.reshape(next_actions_t,(-1,self.num_agents*self.action_size))
#         next_actions_l_global = torch.reshape(next_actions_l,(-1,self.num_agents*self.action_size))
#         q_input_actions_reshape = torch.reshape(q_input_actions,(-1,self.num_agents*self.action_size))
        

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        #actions_next = self.actor_target(next_states_local) - now calcualted outside of method for all agents
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states_global, next_actions).squeeze()
        
       
        # Compute Q targets for current states (y_i)
        
        Q_targets = rewards_local + (gamma * Q_targets_next * (1 - dones_local))
        # Compute critic loss
        Q_expected = self.critic_local(states_global, actions_global).squeeze()
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        
        
        
        self.actor_optimizer.zero_grad()
        
        index = torch.tensor([self.agent_number]).to(device)
        actions_pred = [actions if i == index else actions.detach() for i, actions in enumerate(all_pred_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(states_global, actions_pred).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()
        
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
    
        self.state = x + dx
        return self.state

