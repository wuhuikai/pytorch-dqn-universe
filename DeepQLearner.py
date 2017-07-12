import copy

import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable

from skimage.color import rgb2yuv
from skimage.transform import resize

import visdom

from TransitionManager import TransitionManager

class DeepQLearner(object):
    def __init__(self, args):
        super(DeepQLearner, self).__init__()

        args.im_size = 84
        args.n_input_channels = args.hist_len*args.n_cols

        args.lr_start = args.lr
        args.lr_end = args.lr
        args.lr_end_time = 1000000
        args.eps = args.eps_start
        args.weight_decay = 0


        self.vis = visdom.Visdom(env='DQN:'+args.game_name)
        self.fps = 1./10
        self.time = time.time()

        self.r_max = 1
        self.args = args
        self.transitions = TransitionManager(self.args)
        self.last_state, self.last_action = None, None

        self.network = nn.Sequential(
            nn.Conv2d(self.args.n_input_channels, 32, 8, 4, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 512, 7),
            nn.ReLU(),
            nn.Conv2d(512, self.args.n_actions, 1)
        )

        if self.args.gpu >= 0:
            self.network = self.network.cuda(self.args.gpu)

        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.args.lr, momentum=0.95, alpha=0.95, eps=0.01, weight_decay=self.args.weight_decay)

        self.target_network = copy.deepcopy(self.network)

    def preprocess(self, input):
        input = np.asarray(input)

        if time.time() - self.time > self.fps:
            self.time = time.time()
            self.vis.image(input.transpose((2, 0, 1)), win=0)

        input = resize(input, (self.args.im_size, self.args.im_size), mode='reflect')
        input = rgb2yuv(input)[:,:,0]

        return input

    def reset(self, reward, test=False):
        self.transitions.reset_recent()

        if not self.args.no_rescale_reward and not test:
            self.r_max = max(self.r_max, reward)
        if not self.last_state is None and not test:
            reward = np.clip(reward, self.args.min_reward, self.args.max_reward)
            self.transitions.add(self.last_state, self.last_action, reward, False)
            self.transitions.add(self.last_state, self.last_action, 0, True)

        self.last_state, self.last_action = None, None

    def perceive(self, raw_state, reward, done, n_steps, test=False, test_eps=None):
        assert(not done)

        state = self.preprocess(raw_state)

        self.transitions.add_recent_state(state, False)
        cur_full_state = self.transitions.get_recent()[np.newaxis,:,:,:]
        action = self.e_greedy(cur_full_state, n_steps, test_eps)
        self.transitions.add_recent_action(action)
        
        if test:
            return action

        if not self.args.no_rescale_reward:
            self.r_max = max(self.r_max, reward)
        if not self.last_state is None:
            reward = np.clip(reward, self.args.min_reward, self.args.max_reward)
            self.transitions.add(self.last_state, self.last_action, reward, False)

        self.last_action = action
        self.last_state = state

        if n_steps > self.args.learn_start and n_steps % self.args.update_freq == 0:
            self.args.lr = max((self.args.lr_start-self.args.lr_end)*(self.args.lr_end_time-max(0, n_steps-self.args.learn_start))/self.args.lr_end_time
                             + self.args.lr_end, self.args.lr_end)
            for group in self.optimizer.param_groups:
                group['lr'] = self.args.lr

            for _ in range(self.args.n_replay):
                self.q_learning_minibatch()

        if n_steps % self.args.target_q_update_freq == 0:
            self.target_network = copy.deepcopy(self.network)

        return action

    def e_greedy(self, state, n_steps, test_eps):
        self.eps = test_eps if not test_eps is None else self.args.eps_end + max(0, (self.args.eps_start-self.args.eps_end)*(self.args.eps_end_time-max(0, n_steps-self.args.learn_start))/self.args.eps_end_time)
        
        if np.random.rand() < self.eps:
            return np.random.randint(self.args.n_actions)
        else:
            return self.greedy(state)

    def greedy(self, state):
        state = torch.from_numpy(state)
        if self.args.gpu >= 0:
            state = state.cuda(self.args.gpu)
        state_var = Variable(state, volatile=True)
        q = self.network(state_var).data.cpu().squeeze().numpy()
        action = np.random.choice(np.where(q == max(q))[0])

        return action

    def q_update(self, state, action, rewards, next_state, done, test=False):
        state_var, action_var, rewards_var, next_state_var, done_var = Variable(state, volatile=test), Variable(action.unsqueeze(1), volatile=test), Variable(rewards, volatile=test), Variable(next_state, volatile=test), Variable(done, volatile=test)

        target_q_max = self.target_network(next_state_var).squeeze().max(1)[0]
        target_q = target_q_max*self.args.discount*(1-done_var) + rewards_var/self.r_max
        q = self.network(state_var).squeeze().gather(1, action_var)

        return q, target_q, target_q_max.data.cpu().numpy(), (q - target_q).data.cpu().numpy()

    def q_learning_minibatch(self):
        state, action, rewards, next_state, done = self.transitions.sample(self.args.batch_size)
        
        q, target_q, _, _ = self.q_update(state, action, rewards, next_state, done)
        target_q = target_q.detach()
        
        self.network.zero_grad()
        loss = F.smooth_l1_loss(q, target_q)
        loss.backward()
        self.optimizer.step()