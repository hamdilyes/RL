
import random
import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from env import EnvAgainstPolicy

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    """
    Basic neural net.
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.net(x)



# Define the state space
STATE_SHAPE = (6, 7)

# Define the action space
ACTION_SIZE = 7
ACTION_SPACE = [i for i in range(ACTION_SIZE)]

# Define the parameters for the DQN algorithm
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 100
buffer_capacity = 10_000

class DQNAgent:
    def __init__(self, state_shape, num_actions, alpha=0.001, gamma=0.95, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, memory_size=1000, batch_size=32):
        self.name = 'DQN Player'
        
        self.state_shape = state_shape
        self.num_actions = 7
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        
        
        self.reset()

    
    def get_action(self, state, mask):
        # Mask the Q-values for illegal actions
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        #print(state)
        q_values = self.get_q(state)
        q_values = np.where(mask > 0, q_values, -np.inf)
        
        # Choose the action with the highest Q-value among the legal actions
        legal_actions = np.where(mask > 0)[0]
        if len(legal_actions) == 0:
            print('No legal moves')
            time.sleep(4)
            return None
        else:
            return np.argmax(q_values)

    def get_epsilon_greedy_action(self, obs):
        state = obs['observation'][:, :, 0] - obs['observation'][:, :, 1]
        mask = obs['action_mask']
        
        if random.random() < self.epsilon:
            # Choose a random action among the legal actions
            legal_actions = np.where(mask > 0)[0]
            return np.random.choice(legal_actions)
        else:
            # Choose the action with the highest Q-value among the legal actions
            return self.get_action(state, mask)

        
    def update(self, state, action, reward, next_state, done):

        state = state['observation'][:, :, 0] - state['observation'][:, :, 1]
        next_state = next_state['observation'][:, :, 0] - next_state['observation'][:, :, 1]
        
        self.buffer.push(torch.tensor(state).unsqueeze(0), 
                           torch.tensor([[action]], dtype=torch.int64), 
                           torch.tensor([reward]), 
                           torch.tensor([done], dtype=torch.int64), 
                           torch.tensor(next_state).unsqueeze(0),
                          )

        if len(self.buffer) < self.batch_size:
            return np.inf

        # get batch
        transitions = self.buffer.sample(self.batch_size)

        state_batch, action_batch, reward_batch, terminated_batch, next_state_batch = tuple(
            [torch.cat(data) for data in zip(*transitions)]
        )
        #print(state_batch)
        values  = self.q_net.forward(state_batch).gather(1, action_batch)

        # Compute the ideal Q values
        with torch.no_grad():
            next_state_values = (1 - terminated_batch) * self.target_net(next_state_batch).max(1)[0]
            targets = next_state_values * self.gamma + reward_batch
        #print(values, targets.unsqueeze(1))
        loss = self.loss_function(values, targets.unsqueeze(1))

        # Optimize the model 
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()
        
        if not((self.n_steps+1) % 32): 
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        self.decrease_epsilon()
            
        self.n_steps += 1
        if done: 
            self.n_eps += 1
        

        return loss.detach().numpy()


        
    def get_q(self, state):
        """
        Compute Q function for a states
        """
        state_tensor = torch.tensor(state).unsqueeze(0).detach()
        with torch.no_grad():
            output = self.q_net.forward(state_tensor) # shape (1,  n_actions)
        return output.numpy()[0]  # shape  (n_actions)
    
    def decrease_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
    
    def reset(self):
        hidden_size = 128
        
        obs_size = self.state_shape[0] * self.state_shape[1]
        n_actions = self.num_actions
        
        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net =  Net(obs_size, hidden_size, n_actions)
        self.target_net = Net(obs_size, hidden_size, n_actions)
        
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=self.alpha)
        
        self.epsilon = self.epsilon
        self.n_steps = 0
        self.n_eps = 0

    def train_env(self,env,num_episodes,agent1):
        losses = []
        #print('Training')
        for _ in tqdm.tqdm(range(num_episodes)):
            self.env = EnvAgainstPolicy(env,agent1,first_player= random.randint(0,1)) #random.randint(0,1)
            self.env.reset()
            done = False
            obs, _, _, _, _ = env.last()
            
            while not done:
                #time.sleep(2)
                
                action = self.get_epsilon_greedy_action( obs)
                self.env.step(action)
                next_state, reward, done, truncated ,_ = self.env.last()
                #print(reward, done, truncated)
                
                loss_val = self.update(obs, action, reward, next_state, done)
                losses.append(loss_val)                    
                obs = next_state
                
                done = done or truncated
                
                if done:
                    break
        return losses

                

