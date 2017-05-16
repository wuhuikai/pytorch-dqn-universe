import numpy as np

import torch

class TransitionManager(object):
    def __init__(self, args):
        super(TransitionManager, self).__init__()

        args.hist_space = 1
        args.hist_idx = [(i+1)*args.hist_space-1 for i in range(args.hist_len)]
        args.recent_mem_size = args.hist_space*args.hist_len

        self.args = args

        self.n_states, self.insert_idx = 0, 0
        self.states = np.zeros((self.args.replay_memory, self.args.im_size, self.args.im_size), dtype=np.uint8)
        self.dones = np.zeros(self.args.replay_memory, dtype=np.bool)
        self.actions = np.zeros(self.args.replay_memory, dtype=np.uint8)
        self.rewards = np.zeros(self.args.replay_memory, dtype=np.float32)

        self.buffer_idx = None
        self.state_buffer = torch.FloatTensor(self.args.buffer_size, self.args.n_input_channels, self.args.im_size, self.args.im_size).zero_()
        self.action_buffer = torch.LongTensor(self.args.buffer_size).zero_()
        self.reward_buffer = torch.FloatTensor(self.args.buffer_size).zero_()
        self.next_state_buffer = torch.FloatTensor(self.args.buffer_size, self.args.n_input_channels, self.args.im_size, self.args.im_size).zero_()
        self.done_buffer = torch.FloatTensor(self.args.buffer_size).zero_()
        if self.args.gpu >= 0:
            self.state_buffer, self.next_state_buffer, self.reward_buffer, self.action_buffer, self.done_buffer = self.state_buffer.cuda(self.args.gpu), self.next_state_buffer.cuda(self.args.gpu), self.reward_buffer.cuda(self.args.gpu), self.action_buffer.cuda(self.args.gpu), self.done_buffer.cuda(self.args.gpu)

        self.recent_states = []
        self.recent_dones = []
        self.recent_actions = []

    def reset_recent(self):
        self.recent_states = []
        self.recent_dones = []
        self.recent_actions = []        

    def add_recent_state(self, state, done):
        state = np.asarray(state*255, dtype=np.uint8)

        if len(self.recent_states) == 0:
            self.recent_states = [np.zeros_like(state) for _ in range(self.args.recent_mem_size)]
            self.recent_dones = [True for _ in range(self.args.recent_mem_size)]

        self.recent_states.append(state)
        self.recent_dones.append(done)

        if len(self.recent_states) > self.args.recent_mem_size:
            self.recent_states.pop(0)
            self.recent_dones.pop(0)

    def get_recent(self):
        return self.concat_frames(0, True).astype(np.float32) / 255

    def concat_frames(self, idx, use_recent=False):
        states = self.recent_states if use_recent else self.states
        dones = self.recent_dones if use_recent else self.dones

        full_state = np.zeros((self.args.hist_len, self.args.im_size, self.args.im_size), dtype=np.uint8)

        start, done = 0, False
        for i in range(self.args.hist_len-1, 0, -1):
            for j in range(idx+self.args.hist_idx[i-1], idx+self.args.hist_idx[i]):
                if dones[j]:
                    start = i
                    done = True
                    break
            if done:
                break

        for i in range(start, self.args.hist_len):
            full_state[i] = states[idx+self.args.hist_idx[i]]

        return full_state

    def add(self, state, action, reward, done):
        if self.n_states < self.args.replay_memory:
            self.n_states += 1

        self.states[self.insert_idx] = np.asarray(state*255, dtype=np.uint8)
        self.actions[self.insert_idx] = action
        self.rewards[self.insert_idx] = reward
        self.dones[self.insert_idx] = done

        self.insert_idx = (self.insert_idx+1)%self.args.replay_memory

    def add_recent_action(self, action):
        if len(self.recent_actions) == 0:
            self.recent_actions = [0 for _ in range(self.args.recent_mem_size)]

        self.recent_actions.append(action)

        if len(self.recent_actions) > self.args.recent_mem_size:
            self.recent_actions.pop(0)

    def sample(self, batch_size):
        if self.buffer_idx == None or self.buffer_idx+batch_size > self.args.buffer_size:
            self.fill_buffer()

        idx = self.buffer_idx
        self.buffer_idx += batch_size

        return self.state_buffer[idx:self.buffer_idx], self.action_buffer[idx:self.buffer_idx], self.reward_buffer[idx:self.buffer_idx], self.next_state_buffer[idx:self.buffer_idx], self.done_buffer[idx:self.buffer_idx]

    def sample_one(self):
        while True:
            idx = np.random.randint(0, self.n_states - self.args.recent_mem_size)
            if not self.dones[idx+self.args.hist_idx[-1]]:
                break

        return self.get(idx)

    def get(self, idx):
        action_reward_idx = idx + self.args.hist_idx[-1]
        return self.concat_frames(idx), self.actions[action_reward_idx], self.rewards[action_reward_idx], self.concat_frames(idx+1), self.dones[action_reward_idx+1]

    def fill_buffer(self):
        self.buffer_idx = 0
        for i in range(self.args.buffer_size):
            state, action, reward, next_state, done = self.sample_one()
            self.state_buffer[i] = torch.from_numpy(state.astype(np.float32) / 255)
            self.action_buffer[i] = int(action)
            self.reward_buffer[i] = float(reward)
            self.next_state_buffer[i] = torch.from_numpy(next_state.astype(np.float32) / 255)
            self.done_buffer[i] = float(done)