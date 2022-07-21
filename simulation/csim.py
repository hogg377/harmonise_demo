#!/usr/bin/env python

import string
import numpy as np
import logging
import time
import random
import math

import simulation.environment as environment
import scipy
from numpy.linalg import norm
from scipy.spatial.distance import cdist, pdist, euclidean
from scipy.cluster.vq import kmeans, whiten


###########################################################################################


class cswarm(object):

	def __init__(self):

		# Agent opinons are black if == 1

		self.agents = []
		self.opinions = []
		self.quality = []
		self.black = []
		self.white = []
		self.exploring = []
		self.explore_counter = []
		self.disseminating = []
		self.dissem_counter = []
		self.dissem_period = 0
		self.explore_period = []
		
		self.speed = 0.5
		self.comm_range = 20
		self.size = 0
		self.headings = np.zeros(self.size)
		self.behaviour = 'none'

		self.centroids = []
		self.centermass = [0,0]
		self.median = [0,0]
		self.upper = [0,0]
		self.lower  = [0,0]
		self.spread = 0
		self.belief = 0

		self.param = 3
		self.map = None

		self.beacon_att = np.array([[]])
		self.beacon_rep = np.array([[]])

		self.origin = np.array([0,0])
		self.start = np.array([])

	def gen_agents(self):

		dim = 0.001
		self.agents = np.zeros((self.size,2))

		# Generate 50/50 distribution of opinions
		self.opinions = np.zeros(self.size)
		for n in range(int(self.size/2)):
			self.opinions[n] = 1

		self.quality = np.zeros(self.size)
		self.black = np.zeros(self.size)
		self.white = np.zeros(self.size)

		self.exploring = np.ones(self.size)
		self.explore_counter = np.zeros(self.size)
		self.disseminating = np.zeros(self.size)
		self.dissem_counter = np.zeros(self.size)
		self.explore_period = np.random.randint(25,50, self.size)		# self.dissem_period = np.zeros(self.size)

		#self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		for n in range(self.size):
			self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2) + self.origin[0], 0 + self.origin[1]])

		self.agents = 1.0*np.random.randint(-9,9, (self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		

	def reset(self):

		dim = 0.001
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		for n in range(self.size):
			self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2),0])

	def iterate(self):
		global env
		if self.behaviour == 'random':
			random_walk(self, self.param)
		if self.behaviour == 'rotate':
			random_walk(self, self.param)
		if self.behaviour == 'aggregate':
			aggregate(self, self.param)
		if self.behaviour == 'disperse':
			dispersion(self, np.array([0,0]), self.param)
		if self.behaviour == 'north':
			dispersion(self, np.array([0,1]), self.param)
		if self.behaviour == 'south':
			dispersion(self, np.array([0,-1]), self.param)
		if self.behaviour == 'west':
			dispersion(self, np.array([-1,0]), self.param)
		if self.behaviour == 'east':
			dispersion(self, np.array([1,0]), self.param)
		if self.behaviour == 'northwest':
			dispersion(self, np.array([-1,1]), self.param)
		if self.behaviour == 'northeast':
			dispersion(self, np.array([1,1]), self.param)
		if self.behaviour == 'southwest':
			dispersion(self, np.array([-1,-1]), self.param)
		if self.behaviour == 'southeast':
			dispersion(self, np.array([1,-1]), self.param)
		if self.behaviour == 'avoidance':
			avoidance(self, env)

	def get_state(self):

		totx = 0; toty = 0; totmag = 0
		# Calculate connectivity matrix between agents
		mag = cdist(self.agents, self.agents)
		totmag = np.sum(mag)
		totpos = np.sum(self.agents, axis=0)

		# calculate density and center of mass of the swarm
		self.spread = totmag/((self.size -1)*self.size)
		self.centermass[0] = (totpos[0])/(self.size)
		self.centermass[1] = (totpos[1])/(self.size)
		self.median = np.median(self.agents, axis = 0)
		self.upper = np.quantile(self.agents, 0.75, axis = 0)
		self.lower = np.quantile(self.agents, 0.25, axis = 0)
		self.belief = float(np.sum(self.opinions)/self.size)

		# Update agent decision making

		for n in range(self.size):
			# Loop through all agents and update their decision making states

			if self.exploring[n] == 1 and self.explore_counter[n] >= self.explore_period[n]:
				#print(self.quality)
				# Switch from exploring to disseminating
				self.exploring[n] = 0
				self.disseminating[n] = 1
				self.explore_counter[n] = 0
				# Set dissem period based on estimated quality
				#self.dissem_period = int(self.quality*50)
				#print('Agent %d quality is %.2f' %(n, self.quality[n]))

			if self.disseminating[n] == 1:

				if self.dissem_counter[n] > self.dissem_period:

					# Switch to exploring state
					self.exploring[n] = 1
					self.disseminating[n] = 0

					dist = mag <= self.comm_range

					# At the end of the dissem period pick a random agent and compare qualities
					inrange = np.where(dist[n] == True)
					#print('detected agents: ', inrange)
					choice = inrange[0][np.random.randint(0, len(inrange))]
					#print('\nRandom choice = ', choice)
					############### ERROR????????????? #################
					
					'''
					Possible error here, using pick to compare qualities with a random agent in the 
					swarm rather than using choice to pick agent in range?
					'''

					pick = np.random.randint(0, self.size)
					# compare qualities
					if self.quality[pick] > self.quality[n]:
						#input()
						#print('\n\nfor agent ', n)
						#print('My quality is: %d with %.2f' %(self.opinions[n], self.quality[n]))
						#print('Pick quality is: %d with %.2f' %(self.opinions[pick], self.quality[pick]))
						self.opinions[n] = self.opinions[pick]
					# Reset quality estimate
					self.black[n] = 0
					self.white[n] = 0
					self.dissem_counter[n] = 0

					# Give agent reandom exploration period
					self.explore_period[n] = np.random.randint(25, 50)
				else:
					self.dissem_counter[n] += 1




	def dissemination(self):

		# Check which agents are within comms range

		dist = cdist(self.agents, self.agents)
		dist = dist <= self.comm_range


	def check_grid(self, grid):

		# Get intersection of agents and grids
		matrix = cdist(self.agents, grid.grid_black)
		matrix = matrix <= (grid.grid_size/2)

		#print('Detection matrix black: ', matrix )

		# Sum along agent axis as each agent can only be in one grid at a time
		detb = np.sum(matrix, axis = 1)

		#print('Summed matrix black: ', detb )

		# Get intersection of agents and grids
		matrix = cdist(self.agents, grid.grid_white)
		matrix = matrix <= (grid.grid_size/2)

		# Sum along agent axis as each agent can only be in one grid at a time
		detw = np.sum(matrix, axis = 1)

		# Update the detection of grid detection
		for n in range(0, self.size):

			if self.exploring[n] == 1:
				# If agents opinon is black

				if self.opinions[n] == 1 and detb[n] != 0:

					self.black[n] += 1
					self.quality[n] = self.black[n]/self.explore_period[n]

				# if agent opinion is white
				if self.opinions[n] == 0 and detw[n] != 0:

					self.white[n] += 1
					self.quality[n] = self.white[n]/self.explore_period[n]

				self.explore_counter[n] += 1


	def copy(self):
		newswarm = swarm()
		newswarm.agents = self.agents[:]
		newswarm.speed = self.speed
		newswarm.size = self.size
		newswarm.behaviour = 'none'
		newswarm.map = self.map.copy()
		#newswarm.beacon_set = self.beacon_set
		return newswarm


class agent_state(object):

	def __init__(self):

		self.estimate = []


class gridset(object):

	def __init__(self):

		self.grid_size = 5
		self.grid_state = np.zeros(16*16)
		self.distribution = 0.6

		self.grid_pos = []
		self.grid_black = []
		self.grid_white = []

	def gen(self):

		# Set amount of grid black based on exact distribution
		self.grid_state = np.zeros(16*16)
		for a in range(int(np.rint(16*16*self.distribution))):
			self.grid_state[a] = 1
		# Shuffle list randomly 
		np.random.shuffle(self.grid_state)

		self.grid_black = np.zeros((int(np.sum(self.grid_state)), 2))
		self.grid_white = np.zeros((len(self.grid_state) - int(np.sum(self.grid_state)), 2))


		self.grid_pos = np.zeros((16*16, 2))

		dim = np.linspace(-37.5,37.5, 16)
		g = np.meshgrid(dim,dim)

		# Stack ordinates into grid pairs
		g = np.stack((g[0],g[1]), axis = 2)

		# Reduce dimensions into 1-D list of 2-D points
		self.grid_pos = g.reshape(-1, g.shape[-1])

		# Loop through grid state and allocate to black and white lists
		black = 0
		white = 0

		# Generate lists of black and white grid positions
		for n in range(len(self.grid_state)):

			if self.grid_state[n] == 1:

				self.grid_black[black] = self.grid_pos[n]
				black += 1
			else:

				self.grid_white[white] = self.grid_pos[n]
				white += 1

	def get_state(self, swarm):

		pass



def avoidance(agents, map):

	size = len(agents)
	# Compute vectors between agents and wall planes
	diffh = np.array([map.planeh-agents[n][1] for n in range(size)])
	diffv = np.array([map.planev-agents[n][0] for n in range(size)])
		
	# split agent positions into x and y arrays
	agentsx = agents.T[0]
	agentsy = agents.T[1]

	# Check intersection of agents with walls
	low = agentsx[:, np.newaxis] >= map.limh.T[0]
	up = agentsx[:, np.newaxis] <= map.limh.T[1]
	intmat = up*low

	# For larger environments
	A = 10; B = 10
	# For smaller environments
	#A = 2; B = 5

	# Compute force based vector and multiply by intersection matrix
	Fy = np.exp(-A*abs(diffh) + B)*diffh*intmat
	#Fy = -3/diffh*intmat*np.exp(-abs(diffh) + 3)
	#Fy = Fy*diffh*intmat

	low = agentsy[:, np.newaxis] >= map.limv.T[0]
	up = agentsy[:, np.newaxis] <= map.limv.T[1]
	intmat = up*low

	Fx = np.exp(-A*abs(diffv) + B)*diffv*intmat
	#Fx = -3/diffv*intmat*np.exp(-abs(diffv) + 3)
	#Fx = Fx*diffv*intmat

	# Sum the forces between every wall into one force.
	Fx = np.sum(Fx, axis=1)
	Fy = np.sum(Fy, axis=1)
	# Combine x and y force vectors
	
	F = np.stack((Fx, Fy), axis = 1)
	return F


def random_walk(swarm, param):

	alpha = 0.01; beta = 50

	noise = param*np.random.randint(-beta, beta+1, (swarm.size))
	swarm.headings += noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	# Agent avoidance
	R = 20; r = 2; A = 1; a = 20	
	a = np.zeros((swarm.size, 2))
	B = np.zeros((swarm.size, 2))
	
	A = avoidance(swarm.agents, swarm.map)
	a += A + G 

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	swarm.agents += W 


def dispersion(swarm, vector, param):

	R = param; r = 2; A = 1; a = 20

	# noise = 0.1*np.random.randint(-1., 2., (swarm.size,2))
	noise = np.random.uniform(-.1, .1, (swarm.size, 2))

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	A = avoidance(swarm.agents, swarm.map)
	B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	a += A + B - vector + noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += W

def aggregate(swarm, param):

	R = param; 
	r = 3.5
	A = 6.5
	a = 7.5
	# noise = 0.1*np.random.randint(-1., 2., (swarm.size,2))
	noise = np.random.uniform(-.1, .1, (swarm.size, 2))

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	avoid = avoidance(swarm.agents, swarm.map)
	
	k = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)
	b = -A*a*np.exp(-mag/a)[:,np.newaxis,:]*diff/(swarm.size-1)	
	k = np.sum(k, axis = 0).T
	b = np.sum(b, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	k += avoid + b + noise
	
	vecx = k.T[0]
	vecy = k.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += W

def beacon(swarm):

	Ba = np.array([0,0])
	Br = np.array([0,0])
	
	# Check whether beacons exist.
	if swarm.beacon_att.size != 0:
		#diffa = np.array([swarm.beacon_att-swarm.agents[n] for n in range(swarm.size)])
		diffa = swarm.beacon_att[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:]
		maga = cdist(swarm.beacon_att, swarm.agents)
		Ga = -1*2*np.exp(-maga/2)
		Ba = np.sum(Ga[:,np.newaxis,:]*diffa, axis = 0).T

	if swarm.beacon_rep.size != 0:
		#diffr = np.array([swarm.beacon_rep-swarm.agents[n] for n in range(swarm.size)])
		diffr = swarm.beacon_rep[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:]
		magr = cdist(swarm.beacon_rep, swarm.agents)
		Gr = 5*2*np.exp(-magr/2)
		Br = np.sum(Gr[:,np.newaxis,:]*diffr, axis = 0).T
	
	B = Ba + Br
	return B
	
def rotate(swarm, direction, param):

	noise = param*np.random.randint(direction[0], direction[1], swarm.size)
	swarm.headings += noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	# Agent avoidance
	R = 20; r = 2; A = 1; a = 20
	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)
	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis =0).T

	B = np.zeros((swarm.size, 2))
	B = beacon(swarm)
	A = avoidance(swarm.agents, swarm.map)
	a += G + A + B

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += W
	












