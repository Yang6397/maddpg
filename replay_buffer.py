# encoding: utf-8
# desc: 经验池
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, state_dim, action_dim, batch_size):
        self.capacity = capacity
        self.obs_cap = np.empty((self.capacity, obs_dim))
        self.next_obs_cap = np.empty((self.capacity, obs_dim))
        self.state_cap = np.empty((self.capacity, state_dim))
        self.next_state_cap = np.empty((self.capacity, state_dim))
        self.action_cap = np.empty((self.capacity, action_dim))
        self.reward_cap = np.empty((self.capacity, 1))
        self.done_cap = np.empty((self.capacity, 1), dtype=bool)

        self.batch_size = batch_size
        self.current = 0

    def add_memo(self, obs, next_obs, state, next_state, action, reward, done):
        self.obs_cap[self.current] = obs
        self.next_obs_cap[self.current] = next_obs
        self.state_cap[self.current] = state
        self.next_state_cap[self.current] = next_state
        self.action_cap[self.current] = action
        self.reward_cap[self.current] = reward
        self.done_cap[self.current] = done

        self.current = (self.current + 1) % self.capacity


    def sample(self, idxes):
        obs = self.obs_cap[idxes]
        next_obs = self.next_obs_cap[idxes]
        state = self.state_cap[idxes]
        next_state = self.next_state_cap[idxes]
        action = self.action_cap[idxes]
        reward = self.reward_cap[idxes]
        done = self.done_cap[idxes]

        return obs, next_obs, state, next_state, action, reward, done
