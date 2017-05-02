from __future__ import print_function

import os
import copy
import random
import pprint
import argparse

import numpy as np

import torch

import gym
import universe
from universe import wrappers

from DeepQLearner import DeepQLearner

def reset(env):
    observation_n = env.reset()
    while observation_n[0] == None or np.sum(observation_n[0]['vision']) == 0:
        observation_n, reward_n, done_n, info = env.step([env.action_space.sample()])

    return observation_n, reward_n, done_n, info

def main():
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning')
    parser.add_argument('--game_name', required=True, help='Name of the game to play')
    parser.add_argument('--valid_size', type=int, nargs='+', default=(195, 160), help='Valid aera')

    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--eps_start', type=float, default=1, help='Eps at start')
    parser.add_argument('--eps_end', type=float, default=0.1, help='Eps at end')
    parser.add_argument('--eps_end_time', type=int, default=1000000, help='# of frames between eps and eps_end')
    
    parser.add_argument('--n_cols', type=int, default=1, help='# of channels for input images')
    parser.add_argument('--hist_len', type=int, default=4, help='# of frames as history')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--update_freq', type=int, default=4)
    parser.add_argument('--n_replay', type=int, default=1, help='# of replays per update')
    
    parser.add_argument('--replay_memory', type=int, default=1000000, help='Size of replay memory')
    parser.add_argument('--buffer_size', type=int, default=512)
    parser.add_argument('--eval_size', type=int, default=500)
    
    parser.add_argument('--no_rescale_reward', action='store_true', default=False, help='Do not resacle the reward?')
    parser.add_argument('--min_reward', type=float, default=-1)
    parser.add_argument('--max_reward', type=float, default=1)
    

    parser.add_argument('--n_steps', type=int, default=50000000, help='# of training steps to perform')
    parser.add_argument('--eval_steps', type=int, default=125000, help='# of evaluation steps')
    parser.add_argument('--learn_start', type=int, default=50000, help='# of frames to skip before learning')
    parser.add_argument('--target_q_update_freq', type=int, default=10000)

    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--experiment', default='results', help='Name of folder for storing outputs')

    parser.add_argument('--save_intervel', type=int, default=125000, help='Frequency of model saving')
    parser.add_argument('--eval_intervel', type=int, default=250000, help='Frequency of greedy evaluation')

    args = parser.parse_args()
    print('Input arguments:')
    for key, value in sorted(vars(args).items()):
        print('\t{}: {}'.format(key, value))
    print('')

    # Set GPU
    torch.cuda.set_device(args.gpu)

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed_all(args.seed)

    # Init universe
    env = gym.make(args.game_name)
    env.configure(remotes=1)
    env = wrappers.experimental.SafeActionSpace(env)

    args.n_actions = len(env.action_space.actions)

    # eval_env = gym.make(args.game_name)
    # eval_env.configure(remotes=1)
    # eval_env = wrappers.experimental.SafeActionSpace(eval_env)

    dqn = DeepQLearner(args)

    history = []
    best_network = None
    
    observation_n, reward_n, done_n, info = reset(env)
    for step in range(args.n_steps):
        action = dqn.perceive(observation_n[0]['vision'], reward_n[0], done_n[0], step)
        next_observation_n, reward_n, done_n, info = env.step([env.action_space[action]])

        if done_n[0] or next_observation_n[0] is None:
            action = dqn.perceive(observation_n[0]['vision'], reward_n[0], done_n[0], step)
            observation_n, reward_n, done_n, info = reset(env)
        else:
            observation_n = next_observation_n

        if step == args.learn_start:
            state_eval, action_eval, rewards_eval, next_state_eval, done_eval = dqn.transitions.sample(args.eval_size)
            state_eval, action_eval, rewards_eval, next_state_eval, done_eval = state_eval.clone(), action_eval.clone(), rewards_eval.clone(), next_state_eval.clone(), done_eval.clone()

        if step % args.eval_intervel == 0 and step > args.learn_start:
            total_reward, n_episodes = 0, 0
        
            eval_observation_n, eval_reward_n, eval_done_n, _ = reset(env)
            for eval_step in range(args.eval_steps):
                eval_action = dqn.perceive(eval_observation_n[0]['vision'], eval_reward_n[0], eval_done_n[0], eval_step, True, 0.05)
                eval_observation_n, eval_reward_n, eval_done_n, _ = env.step([env.action_space[action]])   
                
                total_reward += eval_reward_n[0]
                if eval_done_n[0] or eval_observation_n[0] is None:
                    n_episodes += 1
                    eval_observation_n, eval_reward_n, eval_done_n, _ = reset(env)

            total_reward /= max(1, n_episodes)
            best_reward = None if len(history) == 0 else history[-1]['best_reward']
            if best_network is None or total_reward > best_reward:
                best_network = copy.deepcopy(dqn.network)
                best_reward = total_reward

            _, _, target_q_max, td_error = dqn.q_update(state_eval, action_eval, rewards_eval, next_state_eval, done_eval, test=True)
            target_q_max = np.mean(target_q_max)
            td_error = np.mean(np.abs(td_error))

            history.append({'reward': total_reward, 'v': target_q_max, 'td_error': td_error, 'best_reward': best_reward})

            print('Step: {}, V: {}, TD error: {}, reward: {}'.format(step, target_q_max, td_error, total_reward))

        if step % args.save_intervel == 0 and step > args.learn_start or step == args.n_steps-1:
            if not os.path.isdir(args.experiment):
                os.makedirs(args.experiment)
            torch.save(dqn.network.state_dict(), '%s/DQN_iter_%d.pth' % (args.experiment, step))
            if not best_network is None:
                torch.save(best_network.state_dict(), '%s/X_DQN_iter_%d.pth' % (args.experiment, step))

            with open(os.path.join(args.experiment, 'log'), 'w') as f:
                f.write(pprint.pformat(history))

if __name__ == '__main__':
    main()