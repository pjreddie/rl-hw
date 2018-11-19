import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

import gym


class Pytorch_Gym_Env(object):
	'''
	Wrapper for OpenAI-Gym environments so they can easily be used with pytorch
	'''

	def __init__(self, env_name, device='cpu'):
		self._env = gym.make(env_name)
		self.spec = self._env.spec
		self.action_space = self._env.action_space
		self.observation_space = self._env.observation_space
		self._device = device

	def reset(self):
		return torch.from_numpy(self._env.reset()).float().to(self._device).view(-1)

	def render(self, *args, **kwargs):
		return self._env.render(*args, **kwargs)

	def to(self, device):
		self._device = device

	def step(self, action):
		action = action.squeeze().detach().cpu().numpy()
		if action.ndim == 0:
			action = action[()]
		obs, reward, done, info = self._env.step(action)
		obs = torch.from_numpy(obs).float().to(self._device).view(-1)
		reward = torch.tensor(reward).float().to(self._device).view(1)
		done = torch.tensor(done).float().to(self._device).view(1)
		return obs, reward, done, info


class Generator(object):
	'''
	Generates rollouts of an environment using a policy
	'''
	
	def __init__(self, env, policy, horizon=None, drop_last_state=True):
		
		self.created = 0
		
		self.policy = policy
		self.env = env
		
		self.drop_last_state = drop_last_state
		
		self.horizon = self.env.spec.timestep_limit if horizon is None else horizon
	
	def __len__(self):
		return self.created
	
	def __iter__(self):
		return self
	
	def __next__(self):
		return self()
	
	def __call__(self, horizon=None, render=False):
		
		states = []
		actions = []
		rewards = []
		
		states.append(self.env.reset())
		horizon = self.horizon if horizon is None else horizon
		for _ in range(horizon):
			
			if render:
				self.env.render()
			
			actions.append(self.policy(states[-1]))
			
			state, reward, done, _ = self.env.step(actions[-1])
			
			states.append(state)
			rewards.append(reward)
			
			if done:
				break
		
		if self.drop_last_state:
			states.pop()
		
		states = torch.stack(states)
		actions = torch.stack(actions)
		rewards = torch.cat(rewards)
		
		self.created += 1
		
		return states, actions, rewards

# Potentially useful utility functions

def compute_returns(rewards, discount):
	'''
	Computes estimate of discounted reward from a sequence of rewards and the discount factor
	:param rewards: 1D tensor of rewards for an episode
	:param discount: discount factor
	:return: returns (discounted rewards)
	'''
	returns = rewards.clone()
	for i in range(len(returns) - 2, -1, -1):
		returns[i] += discount * returns[i + 1]
	
	return returns

def solve(A, b, out=None, bias=True):
	'''
	Solves for x to minimize (Ax-b)^2
	for some matrix A and vector b
	x is returned as a linear layer (either with or without a bias term)
	Will update out if given, otherwise the output will be a new linear layer
	:param A: D x N pytorch tensor
	:param b: N x 1 pytorch tensor
	:param out: instance of torch.nn.Linear(D,1)
	:param bias: learn a bias term in addition to weights
	:return: torch.nn.Linear(D, 1) instance where the weights (and bias) solve Ax=b
	'''
	# A: M x N
	# b: N x 1
	# x: M x 1
	
	if bias:
		A = torch.cat([A, torch.ones(*(A.size()[:-1] + (1,))).type_as(A)], -1)
	
	x, _ = torch.gels(b, A)
	
	if out is None:
		out = nn.Linear(A.size(-1) - 1, b.size(-1), bias=bias).to(A.device)
	
	out.weight.data.copy_(x[:A.size(-1) - 1].t())
	
	if bias:
		out.bias.data.copy_(x[A.size(-1) - 1:A.size(-1), 0])
	
	return out


def MLE(distrib):
	'''
	Returns Maximum liklihood estimate for the given distribution
	:param distrib: pytorch distribution, should be an instance of one of the distributions listed below
	:return: the maximum liklihood estimate for some parameter of the distribution (eg. mode)
	'''
	if isinstance(distrib, Normal):
		return distrib.loc
	elif isinstance(distrib, Categorical):
		return distrib.probs.max(-1)[1]
	raise Exception('Distribution {} not recognized (did you forget to add it to MLE function?)'.format(type(distrib)))


