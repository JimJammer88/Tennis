
from single_agent import SingleAgent
import torch
import numpy as np
from collections import namedtuple, deque
import random
from configuration import Configuration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MADDPG:
    def __init__(self, action_size, state_size, discount_factor=0.95, tau=0.02, random_seed = 2):
        super(MADDPG, self).__init__()
        
        self.config = Configuration()

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [SingleAgent(agent_number=0,num_agents=2,state_size= state_size, action_size = action_size, random_seed =0), 
                             SingleAgent(agent_number =1, num_agents = 2, state_size = state_size, action_size = action_size, random_seed =1)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.action_size = action_size
        self.state_size = state_size
        self.num_agents = 2
        self.memory = ReplayBuffer(action_size, self.config.buffer_size, self.config.batch_size, random_seed)
        self.timestep = 0
        
        
    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()
        
    def act(self, state, epsilon, add_noise=True):
    
        action = np.zeros((self.num_agents,self.action_size))
        for agent in self.maddpg_agent:
            action[agent.get_number(),:] = agent.act(state[agent.get_number()], epsilon)
        
        return action
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        
        self.timestep += 1
        self.memory.add(state, action, reward, next_state, done)
        

        # Learn, if enough samples are available in memory, num_update times every update_every timesteps
        if len(self.memory) > self.config.batch_size and self.timestep % self.config.update_every == 0:
            for i in range(self.config.num_update):
                experiences = self.memory.sample()
                self.learn(experiences, self.config.gamma)


    def learn(self, experiences, gamma):
        """update the critics and actors of all the agents """
        
        
        
        all_pred_actions = []
        all_next_actions = []
        for agent in self.maddpg_agent:
            states, _, _, next_states, _ = experiences
            agent_id = torch.tensor([agent.get_number()]).to(device)
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            all_pred_actions.append(agent.actor_local(state))
            all_next_actions.append(agent.actor_target(next_state))
            
        

#         all_next_actions = np.zeros((BATCH_SIZE,self.num_agents,self.action_size))
#         for agent in self.maddpg_agent:
#             all_next_actions[:,agent.get_number(),:] = agent.actions_next_target(experiences)
            
#         next_actions_t = torch.from_numpy(all_next_actions).float().to(device)
        
#         all_next_actions = np.zeros((BATCH_SIZE,self.num_agents,self.action_size))
#         for agent in self.maddpg_agent:
#             all_next_actions[:,agent.get_number(),:] = agent.actions_next_local(experiences)
            
#         next_actions_l = torch.from_numpy(all_next_actions).float().to(device)
        
        
        for agent in self.maddpg_agent:
            agent.learn(experiences, all_pred_actions, all_next_actions, gamma)
        

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
