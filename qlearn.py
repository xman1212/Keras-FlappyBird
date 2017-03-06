#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import bisect
import numpy as np
from collections import deque

import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

#ACTIONS_NAME = ['nothing', 'jump']
ACTIONS = [ [1.0, 0.0], [0.0, 1.0]]
NOTHING = 0
JUMP = 1

class ExpFrame( object ):
	''' 一帧完整的经验 '''
	__slots__ = ['curs', 'next_s', 'reward', 'quit', 'action']
	def __init__(self, cur_s, action, next_s, reward, quit):
		self.cur_s = cur_s
		self.next_s = next_s
		self.reward = reward
		self.quit = quit
		self.action = action

class GameAgent( object ):
	ImgBindSize = 4
	def __init__(self):
		self.imgs = []
		self.model = self.buildmodel()
		self.input_k = 60 * 100000.0
		self.input_cnt = 0	
		self.exp_k = 0.5 #action使用随机和不完全预测的策略比例
		self.skip_input = False # 加速前期好样本构造
	
	def buildmodel(self):
		print("Now we build the model")
		model = Sequential()
		model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name,dim_ordering=None: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(img_channels,img_rows,img_cols)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name,dim_ordering=None: normal(shape, scale=0.01, name=name), border_mode='same'))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name, dim_ordering=None: normal(shape, scale=0.01, name=name), border_mode='same'))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
		model.add(Activation('relu'))
		# 2个输出，表示预测的结果是两种输入状态的长远估值
		model.add(Dense(2,init=lambda shape, name: normal(shape, scale=0.01, name=name)))
	   

		try:
			model.load_weights("model.h5")
		except:
			pass

		adam = Adam(lr=1e-6)
		model.compile(loss='mse',optimizer=adam)
		print("We finish building the model")
		return model

	def new_round(self, img):
		self.imgs = [img, img, img, img]
	
	def observe(self, img):
		# 将当前观察到的图案缓存起来,形成历史图像序列
		self.imgs.append(img)
		self.imgs = self.imgs[-self.ImgBindSize:]
		img = np.stack(self.imgs, axis=0)
		r = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
		return r

	def next_action(self, status):
		if len(self.imgs) < 4:
			return ACTIONS[NOTHING]
		
		# 初期增加空白输入，有利于构造好样本
		if self.skip_input and random.random() < 0.8:
			return ACTIONS[NOTHING]

		# 随着输入次数变多，模型变好，应该更多倾向预测值
		self.input_cnt += 1
		r = random.random()
		if r < self.exp_k:
			#random
			if random.random() < 0.3:
				return ACTIONS[JUMP]
			else:
				return ACTIONS[NOTHING]
		else: 
			# predict
			q = self.model.predict(status) 
			action_idx = np.argmax(q)
			r = ACTIONS[action_idx]
		return r

class RoundStatus( object ):
	def __init__(self):
		self.all = []

	def __lt__(self, ins):
		return len(self.all)< len(ins.all)

	def save_exp(self, status):
		self.all.append(status)

	def life(self):
		return len(self.all)
	
	def all_reward(self):
		return [ x[3] for x in self.all]

	def finish(self):
		rewards = self.all_reward()
		#print("before award", rewards)
		l = len(self.all)
		# 时间越久奖励越高
		for i in range(l):
			#exp = (cur_s, action, next_s, reward, quit)
			exp = self.all[i]
			if exp[4]:
				break
			exp[3] += float(i) /( 200 * 30)
		# 死亡惩罚提前扩散
		step = 2
		l = len(rewards)
		dead_reward = rewards[ l - 1 ]
		for i in range(step):
			bi = l - i - 2
			exp = self.all[bi]
			exp[3] += dead_reward * 0.9 **(i+1)
		#print("endaward", self.all_reward())

	def __str__(self):
		return len(self.all)	

class Pool( object ):
	def __init__(self, maxpool=50, ordered=False):
		self.maxpool = maxpool
		self.all = []
		self.total = 0
		self.ordered =ordered
	
	def insert(self, one):
		life = one.life()
		if self.ordered:
			if len(self.all) < self.maxpool or life > self.all[0].life():
				# 有序pool，只插入优于现有的元素
				bisect.insort_left(self.all, one)
			else:
				return False
		else:
			# 无序随意了
			bisect.insort_left(self.all, one)
		self.total += one.life()		
		if len(self.all) > self.maxpool:
			if self.ordered:
				idx = 0	
			else:
				idx = random.randrange(len(self.all))
			unuse = self.all.pop(idx)
			self.total -= unuse.life()
		return True

	def len(self):
		return len(self.all)

	def max(self):
		if len(self.all) <=0:
			return 0
		return self.all[len(self.all)-1].life()

	def min(self):
		if len(self.all) <=0:
			return 0
		return self.all[0].life()
	
	def average(self):
		if len(self.all) <=0:
			return 0.01
		return float(self.total) / len(self.all)
	
	def __str__(self):
		return "range[%d~%d],avge:%f,size:%d"%(
			self.min(),self.max(),self.average(),self.len())

class PoolMgr( object ):
	def __init__(self):
		self.good = Pool(100, True)
		self.rand = Pool(100, False)
		self.total = 0
		self.insert_cnt = 1

	def get_trainsample(self, good=5, rand=5):
		# 最好的经验 + 随机的经验
		r = self._get_nsample(self.rand.all, rand)
		g = self._get_nsample(self.good.all, good)
		return random.choice( r + g )
	
	def _get_nsample(self, samples, n, top=False):
		if n > len(samples):
			n = len(asamples)
		if top:
			return samples[-n:]
		else:
			return random.sample(samples, n)

	def average(self):
		return float(self.total) / self.insert_cnt

	def insert(self, one):
		self.total += one.life()
		self.insert_cnt += 1
		if not self.good.insert(one):
			self.rand.insert(one)
	
	def __str__(self):
		return "avg:%f good:<%s> rand:<%s>"%(self.average(),self.good, self.rand)

def updateGameStep(game_state, action):
	screen_img, reward, quit = game_state.frame_step(action)
	img = skimage.color.rgb2gray(screen_img)
	img = skimage.transform.resize(img, (80,80))
	img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
	return img, reward, quit


def playOneRound(agent, good_pool_len = 0):
	frame_cnt = 0
	gins = game.GameState()
	action = ACTIONS[NOTHING]
	screen_img, reward, quit = updateGameStep(gins,action )		
	agent.new_round(screen_img)
	cur_s = agent.observe(screen_img)
	one_round = RoundStatus()
	while frame_cnt < 30 * 60 * 60:
		frame_cnt += 1
		action = agent.next_action( cur_s )
		screen_img, reward, quit = updateGameStep(gins, action )
		next_s = agent.observe(screen_img)
		act_idx = int(action[1])
		exp = [cur_s, act_idx, next_s, reward, quit]
		one_round.save_exp(exp)
		cur_s = next_s
		if quit:
			break
	#one_round.finish()
	return one_round

def trainQ():
	agent = GameAgent()
	pool = PoolMgr()
	round_cnt = 0
	round_max = 10000000
	all_round = []

	one_round = playOneRound(agent)
	pool.insert(one_round)
	while round_cnt < round_max:
		round_cnt += 1
		one_round = playOneRound(agent, pool.good.len())
		pool.insert(one_round)
		
		# 没有好经验暂时不训练,同事适当降低随机输入比例，易于构造好样本
		gavg,ravg = pool.good.average(), pool.rand.average()
		if pool.good.len() < 5 or gavg <= 50:
			print("round_cnt:%d %s"%(round_cnt, pool))
			agent.skip_input = True
			continue
		else:
			agent.skip_input = False

		# 随机和不完善策略的选择比例，尝试按经验池比例来
		agent.exp_k = float(gavg) / (gavg + ravg)

		good_cnt = 6
		rand_cnt = int(good_cnt * ravg / gavg)
		sample = pool.get_trainsample(good_cnt, rand_cnt)
		exps = sample.all

		img = exps[0][0]
		inputs = np.zeros((len(exps), img.shape[1], img.shape[2], img.shape[3]))   #32, 80, 80, 4
		targets_q = np.zeros((inputs.shape[0], len(ACTIONS)))						 #32, 2
		q_max = 0
		for i in range(len(exps)):
			cur_s, act_idx, next_s, reward, quit = exps[i]
			inputs[i] = cur_s
			# 预测当前状态的整体估值Q
			targets_q[i] = agent.model.predict(cur_s)
			# 根据实际观测到的结果修正Q
			if quit:
				# 游戏结束，则直接用结果修正之,无未来
				targets_q[i][act_idx] = reward
			else:
				# 有未来，需要计算后续状态Q并衰减加入
				pred_next_q = agent.model.predict(next_s)
				targets_q[i][act_idx] = reward + GAMMA * np.max(pred_next_q)
			
		# 用根据感测和预测修正后的Q，来训练网络
		loss = agent.model.train_on_batch(inputs, targets_q)	
		print("round_cnt:%d %s train:%d loss:%f qmax:%f,expk:%f"%(round_cnt, pool, sample.life(), loss, q_max,agent.exp_k))
		if round_cnt %5 == 0:
            		agent.model.save_weights("model.h5", overwrite=True)

def playGame(args):
	trainQ()

def main():
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-m','--mode', help='Train / Run', required=True)
	args = vars(parser.parse_args())
	playGame(args)

if __name__ == "__main__":
	main()
