#!/usr/bin/env python

import string
import numpy as np
import logging
import time
import random
import math
import sys

import simulation.environment as environment
import scipy
from numpy.linalg import norm
from scipy.spatial.distance import cdist, pdist, euclidean
from scipy.cluster.vq import kmeans, whiten

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


###########################################################################################


class swarm(object):

	def __init__(self):

		self.agents = []
		self.speed = 0.5
		self.size = 0
		self.behaviour = 'none'

		self.centroids = []
		self.centermass = [0,0]
		self.median = [0,0]
		self.upper = [0,0]
		self.lower  = [0,0]
		self.spread = 0
		
		self.escape_death = False
		self.dead = []

		self.field = []
		self.grid = []

		self.param = 3
		self.map = 'none'
		self.beacon_att = np.array([[]])
		self.beacon_rep = np.array([[]])

		self.origin = np.array([0,0])
		self.start = np.array([])
		self.shadows = []
		self.died = 0

	def gen_agents(self):

		dim = 0.001
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		for n in range(self.size):
				self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2) + self.origin[0], 0 + self.origin[1]])
		self.shadows = np.zeros((4,self.size,2))

	def gen_agents_uniform(self, env):

		dim = 0.001
		self.dead = np.zeros(self.size)
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		
		x = np.random.uniform(-env.dimensions[1]/2, env.dimensions[1]/2, self.size)
		y = np.random.uniform(-env.dimensions[0]/2, env.dimensions[0]/2, self.size)

		self.agents = np.stack((x,y), axis = 1)
		

	def reset(self):

		dim = 0.001
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		for n in range(self.size):
			self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2),0])

	def iterate(self, noise):
		global env
		if self.behaviour == 'aggregate':
			aggregate(self, self.param, noise)
		if self.behaviour == 'flocking':
			flocking(self, self.param, noise)
		if self.behaviour == 'random':
			random_walk(self, self.param)
		if self.behaviour == 'rot_clock':
			rotate(self, [-2,2], self.param)
		if self.behaviour == 'rot_anti':
			rotate(self, [-1,3], self.param)
		if self.behaviour == 'disperse':
			dispersion(self, np.array([0,0]), self.param, noise)
		if self.behaviour == 'north':
			dispersion(self, np.array([0,1]), self.param,noise)
		if self.behaviour == 'south':
			dispersion(self, np.array([0,-1]), self.param,noise)
		if self.behaviour == 'west':
			dispersion(self, np.array([-1,0]), self.param,noise)
		if self.behaviour == 'east':
			dispersion(self, np.array([1,0]), self.param,noise)
		if self.behaviour == 'northwest':
			dispersion(self, np.array([-1,1]), self.param,noise)
		if self.behaviour == 'northeast':
			dispersion(self, np.array([1,1]), self.param,noise)
		if self.behaviour == 'southwest':
			dispersion(self, np.array([-1,-1]), self.param,noise)
		if self.behaviour == 'southeast':
			dispersion(self, np.array([1,-1]), self.param,noise)
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
		# self.upper = np.quantile(self.agents, 0.75, axis = 0)
		# self.lower = np.quantile(self.agents, 0.25, axis = 0)

		self.shadows[3] = self.shadows[2]
		self.shadows[2] = self.shadows[1]
		self.shadows[1] = self.shadows[0]
		self.shadows[0] = self.agents


	def copy(self):
		newswarm = swarm()
		newswarm.agents = self.agents[:]
		newswarm.speed = self.speed
		newswarm.size = self.size
		newswarm.behaviour = 'none'
		swarm.origin = self.origin
		newswarm.map = self.map.copy()
		newswarm.field = self.field
		newswarm.grid = self.grid
		#newswarm.beacon_set = self.beacon_set
		return newswarm


class target_set(object):

	def __init__(self):
		self.targets = []
		self.radius = 0
		self.found = 0
		self.coverage = 0
		self.old_state = np.zeros(len(self.targets))
		self.fitmap = []

	def set_state(self, state):


		if state == 'set1':
			self.targets = np.array([[-35,35],[-25,35],[-15,35],[-5,35],[5,35],[15,35],[25,35],[35,35],
							[-35,25],[-15,25],[-5,25],[5,25],[15,25],[25,25],[35,25],
							[-35,15],[-15,15],[-5,15],[5,15],[15,15],[25,15],[35,15],
							[-35,5],[-15,5],[15,5],[25,5],[35,5],
							[-35,-5],[-25,-5],[-15,-5],[15,-5],[25,-5],[35,-5],
							[-35,-15],[-25,-15],[-15,-15],[-5,-15],[5,-15],[15,-15],[25,-15],[35,-15],
							[-35,-25],[-25,-25],[-15,-25],[-5,-25],[5,-25],[15,-25],[25,-25],[35,-25],
							[-35,-35],[-25,-35],[-15,-35],[-5,-35],[5,-35],[15,-35],[25,-35],[35,-35]])

		if state == 'set2':
			self.targets = np.array([[-35,35],[-25,35],[-15,35],[-5,35],[5,35],[15,35],[25,35],[35,35],
							[-35,25],[-25,25], [-15,25],[-5,25],[5,25],[15,25],[25,25],[35,25],
							[-35,15],[-25,15], [-15,15],[-5,15],[5,15],[15,15],[25,15],[35,15],
							[-35,5], [-25,5],[-15,5],[15,5],[25,5],[35,5],
							[-35,-5],[-25,-5],[-15,-5],
							[-35,-15],[-25,-15],[-15,-15],[-5,-15],[5,-15],[15,-15],[25,-15],[35,-15],
							[-35,-25],[-25,-25],[-15,-25],[-5,-25],[5,-25],[15,-25],[25,-25],[35,-25],
							[-35,-35],[-25,-35],[-15,-35],[-5,-35],[5,-35],[15,-35],[25,-35],[35,-35]])

		if state == 'set3':
			self.targets = np.array([[-35,35],[-25,35],[-15,35],[-5,35],[5,35],[15,35],[25,35],[35,35],
							[-35,25],[-25,25], [-15,25],[-5,25],[5,25],[15,25],[25,25],[35,25],
							[-35,15],[-25,15], [-15,15],[-5,15],[5,15],[15,15],[25,15],[35,15],
							[-35,5], [-25,5],[-15,5],[15,5],[25,5],[35,5],
							[-35,-5],[-25,-5],[-15,-5], [15,-5], [25,-5], [35,-5],
							[-35,-15],[-25,-15],[-15,-15],[-5,-15],[5,-15],[15,-15],[25,-15],[35,-15],
							[-35,-25],[-25,-25],[-15,-25],[-5,-25],[5,-25],[15,-25],[25,-25],[35,-25],
							[-35,-35],[-25,-35],[-15,-35],[-5,-35],[5,-35],[15,-35],[25,-35],[35,-35]])

		if state == 'set4':
			self.targets = np.array([[-35,35],[-25,35],[-15,35],[-5,35],[5,35],[15,35],[25,35],[35,35],
							[-35,25],[-25,25],[-15,25],[-5,25],[5,25],[15,25],[25,25],[35,25],
							[-35,15],[-25,15],[-15,15],[-5,15],[5,15],[15,15],[25,15],[35,15],
							[-35,5],[-25,5],[-15,5],[15,5],[25,5],[35,5],
							[-35,-5],[-25,-5],[-15,-5],[15,-5],[25,-5],[35,-5],
							[-35,-15],[-25,-15],[-15,-15],[-5,-15],[5,-15],[15,-15],[25,-15],[35,-15],
							[-35,-25],[-25,-25],[-15,-25],[-5,-25],[5,-25],[15,-25],[25,-25],[35,-25],
							[-35,-35],[-25,-35],[-15,-35],[-5,-35],[5,-35],[15,-35],[25,-35],[35,-35]])

		if state == 'brlset':

			x = np.arange(-72.5, 74.9, 2.5)
			y = np.arange(-37.5, 39.9, 2.5)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1

		if state == 'minimap':

			x = np.arange(-17.5, 19.9, 2.5)
			y = np.arange(-17.5, 19.9, 2.5)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1

		if state == '1x1':

			x = np.arange(-9, 9.9, 1)
			y = np.arange(-9, 9.9, 1)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1

		if state == '2x1':

			x = np.arange(-19, 19.9, 1)
			y = np.arange(-9, 9.9, 1)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1

		if state == '3x1':

			x = np.arange(-29, 29.9, 1)
			y = np.arange(-9, 9.9, 1)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1

		if state == '2x2':

			x = np.arange(-19, 19.9, 1)
			y = np.arange(-19, 19.9, 1)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1

		if state == '2x3':

			x = np.arange(-19, 19.9, 1)
			y = np.arange(-29, 29.9, 1)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1

		if state == '3x3':

			x = np.arange(-29, 29.9, 1)
			y = np.arange(-29, 29.9, 1)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1

		if state == '4x4':

			x = np.arange(-39, 39.9, 1)
			y = np.arange(-39, 39.9, 1)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1
		
		
		if state == 'uniform':

			x = np.arange(-40, 45, 3)
			y = np.arange(-40, 45, 3)
			self.targets = np.zeros((len(x)*len(y), 2))
			
			count = 0
			for k in x:
				for j in y:
					self.targets[count][0] = k 
					self.targets[count][1] = j
					count += 1


	def fitness_map(self, env, swarm, timesteps, vis = False):

		'''
		Produce a target fitness map of the environment based on the
		diffusion of random walkers.

		'''
		granularity = 1
		# x = np.arange(-72.5,74.9,granularity)
		# y = np.flip(np.arange(-37.5,39.9,granularity))
		x = np.arange(-(env.dimensions[0]/2) + 1, (env.dimensions[0]/2)-.1,granularity)
		y = np.flip(np.arange(-(env.dimensions[1]/2) + 1, (env.dimensions[1]/2)-.1,granularity))

		pos = np.zeros((len(y),len(x)))

		# Set swarm of random walkers to produce probability map
		swarm.behaviour = 'random'
		swarm.param = 0.01

		total_nodes = len(x)*len(y)
		trials = 1
		noise = np.random.uniform(-.1,.1,(trials*timesteps, swarm.size, 2))

		t = 0
		while t <= trials*timesteps:

			# if t%timesteps == 0:
			# 	swarm.gen_agents()

			swarm.iterate(noise[t-1])
			swarm.get_state()
			
			# Check intersection of agents with targets
			mag = cdist(self.targets, swarm.agents)
			dist = mag <= 2.5

			# For each target sum the number of detections
			total = np.sum(dist, axis = 1)

			# Add the new detections to an array of the positions
			for n in range(len(self.targets)):

				# row = int((self.targets[n][1]+39)/granularity)
				# col = int((self.targets[n][0]+74)/granularity)
				row = int((self.targets[n][1]+ (env.dimensions[1]/2)-1)/granularity)
				col = int((self.targets[n][0]+ (env.dimensions[0]/2)-1)/granularity)

				if total[n] >= 1:
					pos[row][col] += 1
			
			t += 1
			sys.stdout.write("Fit map progress: %.2f   \r" % (100*t/(trials*timesteps)) )
			sys.stdout.flush()

		m = np.max(pos)
		pos = pos/(trials*timesteps)
		pos = pos/np.max(pos)

		if vis == True:
			#Visualize the probability heatmap
			plt.imshow(pos, origin='lower')
			plt.colorbar()
			plt.show()

		return pos


	def get_state(self, swarm, t, timesteps):

		score = 0
		# adjacency matrix of agents and targets
		mag = cdist(swarm.agents, self.targets)

		# Check which distances are less than detection range
		a = mag < self.radius
		# Sum over agent axis 
		detected = np.sum(a, axis = 0)
		# convert to boolean, targets with 0 detections set to false.
		detected = detected > 0
		# Check detection against previous state
		updated = np.logical_or(detected, self.old_state) 
		self.old_state = updated
		score = np.sum(updated)
		self.coverage = score/len(self.targets)	
		return score

	def ad_state(self, swarm, t):

		score = 0
		# adjacency matrix of agents and targets
		mag = cdist(swarm.agents, self.targets)

		# Check which distances are less than detection range
		a = mag < self.radius
		# Sum over agent axis 
		detected = np.sum(a, axis = 0)
		# convert to boolean, targets with 0 detections set to false.
		detected = detected > 0
		# Check detection against previous state
		# check which new targets were found
		new = np.logical_and(np.logical_xor(detected, self.old_state), detected) 

		updated = np.logical_or(detected, self.old_state) 
		self.old_state = updated
		score = np.sum(new)
		self.coverage = np.sum(updated)/len(self.targets)	

		return score


	def reset(self):
		self.old_state = np.zeros(len(self.targets))

class map(object):

	def __init__(self):

		self.name = ''
		self.obsticles = []
		self.force = 0
		self.walls = np.array([])
		self.wallh = np.array([])
		self.wallv = np.array([])
		self.planeh = np.array([])
		self.planev = np.array([])
		self.wall_points = np.array([])
		self.dimensions = [0,0]
		self.door_states = []
		self.bounded = True
		self.origin = np.array([0,0])

		self.swarm_origin = None

		self.sink = []
		self.source = []

	def copy(self):
		newmap = map()
		newmap.walls = self.walls[:]
		newmap.wallh = self.wallh[:]
		newmap.wallv = self.wallv[:]
		newmap.planeh = self.planeh[:]
		newmap.planev = self.planev[:]
		newmap.limh = self.limh[:]
		newmap.limv = self.limv[:]
		newmap.gen()
		return newmap

	def gen(self):

		# Perform pre-processing on map object for efficency
		self.walls = np.zeros((2*len(self.obsticles), 2))
		self.wallh = np.zeros((2*len(self.obsticles), 2))
		self.wallv = np.zeros((2*len(self.obsticles), 2))
		self.planeh = np.zeros(len(self.obsticles))
		self.planev = np.zeros(len(self.obsticles))
		self.limh = np.zeros((len(self.obsticles), 2))
		self.limv = np.zeros((len(self.obsticles), 2))

		for n in range(0, len(self.obsticles)):
			# if wall is vertical
			if self.obsticles[n].start[0] == self.obsticles[n].end[0]:
				self.wallv[2*n] = np.array([self.obsticles[n].start[0], self.obsticles[n].start[1]])
				self.wallv[2*n+1] = np.array([self.obsticles[n].end[0], self.obsticles[n].end[1]])

				self.planev[n] = self.wallv[2*n][0]
				self.limv[n] = np.array([np.min([self.obsticles[n].start[1], self.obsticles[n].end[1]])-0.5, 
											np.max([self.obsticles[n].start[1], self.obsticles[n].end[1]])+0.5])

			# if wall is horizontal
			if self.obsticles[n].start[1] == self.obsticles[n].end[1]:
				self.wallh[2*n] = np.array([self.obsticles[n].start[0], self.obsticles[n].start[1]])
				self.wallh[2*n+1] = np.array([self.obsticles[n].end[0], self.obsticles[n].end[1]])

				self.planeh[n] = self.wallh[2*n][1]
				self.limh[n] = np.array([np.min([self.obsticles[n].start[0], self.obsticles[n].end[0]])-0.5, 
											np.max([self.obsticles[n].start[0], self.obsticles[n].end[0]])+0.5])

			self.walls[2*n] = np.array([self.obsticles[n].start[0], self.obsticles[n].start[1]])
			self.walls[2*n+1] = np.array([self.obsticles[n].end[0], self.obsticles[n].end[1]])

		# Generate map points

		self.wall_points = []

		i = 0

		while i <= (len(self.walls)-1):

			increments = 100

			dify = self.walls[i+1][1] - self.walls[i][1]
			difx = self.walls[i+1][0] - self.walls[i][0]

			for k in range(increments):

				point = [self.walls[i][0] + k*(difx/increments), self.walls[i][1] + k*(dify/increments)] 
				self.wall_points.append(point)

			i += 2

		# Convert wall points into numpy array

		out = np.zeros((len(self.wall_points),2))

		for i in range(len(out)):
			out[i] = self.wall_points[i]

		self.wall_points = out

	def transposeX(self):

		'''
		Before calling env generation, transpose the positions of 
		obstacles in the x direction. 
		'''
		for i in range(len(self.obsticles)):

			self.obsticles[i].start[0] = -1*self.obsticles[i].start[0]
			self.obsticles[i].end[0] = -1*self.obsticles[i].end[0]


	def rotation(self, degrees):
		'''
		Perform rotation on environment about the origin

		Rotation matrix = [cos(theta)   sin(theta)]
			              [-sin(theta)    cos(theta)]
		'''
		degrees = math.radians(degrees)
		
		for i in range(0, len(self.obsticles)):

			x1 = self.obsticles[i].end[0]
			y1 = self.obsticles[i].end[1]
			x2 = self.obsticles[i].start[0]
			y2 = self.obsticles[i].start[1]
			# Rounding to one decimal place needed otherwise some walls are not
			# detected as horizontal/vertical and agents pass through them.

			self.obsticles[i].end[0] = round(x1*math.cos(degrees) + y1*math.sin(degrees), 1)
			self.obsticles[i].end[1] = round(-x1*math.sin(degrees) + y1*math.cos(degrees), 1)
			self.obsticles[i].start[0] = round(x2*math.cos(degrees) + y2*math.sin(degrees), 1)
			self.obsticles[i].start[1] = round(-x2*math.sin(degrees) + y2*math.cos(degrees), 1)


	def empty(self):

		self.dimensions = [80,80]

		dimx = 80
		dimy = 80

		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		


	def test(self):

		self.dimensions = [80,80]

		# Bounding Walls ---------------------------------
		box = environment.box(80, 80, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [-25,20]; wall.end = [25,20];
		self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-25,-20]; wall.end = [25,-20];
		self.obsticles.append(wall)

	def map1(self):

		self.dimensions = [80,80]

		wall = environment.wall(); wall.start = [45,10]; wall.end = [45,20]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [40,10]; wall.end = [45,10]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [40,20]; wall.end = [45,20]; self.obsticles.append(wall)



		# Bounding Walls ---------------------------------
		# box = environment.box(80, 80, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		wall = environment.wall(); wall.start = [-40,40]; wall.end = [40,40]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-40,40]; wall.end = [-40,-40]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [40,10]; wall.end = [40,-40]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-30,-40]; wall.end = [40,-40]; self.obsticles.append(wall)


		room = environment.room(20, 20, 10, 'top', [0, 0]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(20, 20, 7, 'bottom', [0, 30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(20, 30, 10, 'bottom', [25, 30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		#doorway = environment.doorway(30, 7, 'horizontal', [25, 10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		# wall = environment.wall(); wall.start = [10,10]; wall.end = [40,10];
		# self.obsticles.append(wall)
		doorway = environment.doorway(30, 10, 'horizontal', [25, 10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		

		box = environment.box(3, 3, [20, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(3, 3, [30, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(3, 3, [20, -10]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(3, 3, [30, -10]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		doorway = environment.doorway(30, 7, 'horizontal', [25, -20]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(30, 7, 'vertical', [10, -25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		doorway = environment.doorway(30, 7, 'horizontal', [-25, -10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		room = environment.room(30, 10, 7, 'right', [-35, -25]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		box = environment.box(15, 3, [-2, -25]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(15, 3, [-18, -25]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		box = environment.box(30, 5, [-25, 15]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		#corridor = environment.corridor(30, 5, 'horizontal', [30,-10]); [self.obsticles.append(corridor.walls[x]) for x in range(0, len(corridor.walls))]

	def map2(self):

		self.dimensions = [80,80]
		# Bounding Walls ---------------------------------
		box = environment.box(80, 80, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		room = environment.room(20, 20, 10, 'top', [0, 0]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		doorway = environment.doorway(20, 7, 'vertical', [20, 30]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(20, 7, 'horizontal', [30, 20]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		doorway = environment.doorway(30, 10, 'vertical', [10, 25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(30, 10, 'horizontal', [25, 10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		wall = environment.wall(); wall.start = [-10,40]; wall.end = [-10,10]; self.obsticles.append(wall)

		doorway = environment.doorway(30, 7, 'horizontal', [25, -20]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		wall = environment.wall(); wall.start = [-10,-20]; wall.end = [10,-20]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-10,-20]; wall.end = [-10,-40]; self.obsticles.append(wall)

		room = environment.room(20, 20, 7, 'top', [-30, -30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(20, 20, 7, 'bottom', [-30, 0]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		doorway = environment.doorway(30, 7, 'vertical', [-20, 25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		box = environment.box(7, 15, [25, -5]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		#box = environment.box(15, 7, [30, 30]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

	def map3(self):

		self.dimensions = [80,80]
		
		# Bounding Walls ---------------------------------
		box = environment.box(80, 80, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [-10,-10]; wall.end = [10,-10]; self.obsticles.append(wall)

		doorway = environment.doorway(20, 7, 'horizontal', [0, 10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(20, 7, 'vertical', [10, 0]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(20, 7, 'vertical', [-10, 0]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		doorway = environment.doorway(40, 10, 'horizontal', [0, 20]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		#doorway = environment.doorway(40, 10, 'horizontal', [0, -20]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		wall = environment.wall(); wall.start = [-20, -20]; wall.end = [20,-20]; self.obsticles.append(wall)

		doorway = environment.doorway(30, 10, 'vertical', [-20, -5]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		wall = environment.wall(); wall.start = [-40, 10]; wall.end = [-20,10]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [20, 30]; wall.end = [20,-20]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [20, -20]; wall.end = [40,-20]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-20, 40]; wall.end = [-20,20]; self.obsticles.append(wall)

		doorway = environment.doorway(10, 7, 'vertical', [-20, 15]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		room = environment.room(10, 30, 7, 'top', [-25, -35]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(10, 30, 7, 'top', [25, -35]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

	def map4(self):


		self.dimensions = [80,80]
		# Bounding Walls ---------------------------------
		box = environment.box(80, 80, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]


		# Dynamic doorways
		
		# Randomly set state of doors
		doornum = 6
		totopen = 4
		choice = [x for x in range(1, doornum+1)]
		open_doors = [False for x in range(doornum)]

		for n in range(totopen):
			# Pick random doors to open
			pick = random.randint(0,len(choice)-1)
			open_doors[choice[pick]-1] = True
			# Remove option
			choice.remove(choice[pick])

		# Set door states
		if open_doors[0] == True: 
			doorway = environment.doorway(30, 10, 'horizontal', [25, 0]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		else:
			wall = environment.wall(); wall.start = [10, 0]; wall.end = [40,0]; self.obsticles.append(wall)

		if open_doors[1] == True: 
			doorway = environment.doorway(30, 10, 'horizontal', [-25, 0]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		else:
			wall = environment.wall(); wall.start = [-40, 0]; wall.end = [-10,0]; self.obsticles.append(wall)

		if open_doors[2] == True: 
			doorway = environment.doorway(30, 10, 'vertical', [-10, 25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		else:
			wall = environment.wall(); wall.start = [-10, 10]; wall.end = [-10,40]; self.obsticles.append(wall)

		if open_doors[3] == True: 
			doorway = environment.doorway(30, 10, 'vertical', [10, 25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		else:
			wall = environment.wall(); wall.start = [10, 10]; wall.end = [10,40]; self.obsticles.append(wall)

		if open_doors[4] == True: 
			doorway = environment.doorway(30, 10, 'vertical', [-10, -25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		else:
			wall = environment.wall(); wall.start = [-10, -10]; wall.end = [-10,-40]; self.obsticles.append(wall)

		if open_doors[5] == True: 
			doorway = environment.doorway(30, 10, 'vertical', [10, -25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		else:
			wall = environment.wall(); wall.start = [10, -10]; wall.end = [10,-40]; self.obsticles.append(wall)

		# Static elements

		doorway = environment.doorway(20, 7, 'horizontal', [0, 10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(20, 7, 'horizontal', [0, -10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		wall = environment.wall(); wall.start = [-10, 10]; wall.end = [-10,-10]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [10, 10]; wall.end = [10,-10]; self.obsticles.append(wall)

	def map5(self):

		self.dimensions = [80,80]
		# Bounding Walls ---------------------------------
		box = environment.box(80, 80, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]


		# Dynamic doorways
		doorA = True
		doorB = True
		
		# Randomly set state of doors
		closed = bool(random.getrandbits(1))

		# Set door states
		if closed == True and doorA == True:
			wall = environment.wall(); wall.start = [10, 10]; wall.end = [40,10]; self.obsticles.append(wall)
		else:
			doorway = environment.doorway(30, 10, 'horizontal', [25, 10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		# Randomly set state of doors
		closed = np.logical_not(closed)

		if closed == True and doorB == True:
			wall = environment.wall(); wall.start = [-30, -10]; wall.end = [-10,-10]; self.obsticles.append(wall)
		else:
			doorway = environment.doorway(30, 7, 'horizontal', [-25, -10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]


		# Static elements		

		room = environment.room(20, 20, 10, 'top', [0, 0]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(20, 20, 7, 'bottom', [0, 30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(20, 30, 10, 'bottom', [25, 30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		box = environment.box(3, 3, [20, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(3, 3, [30, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(3, 3, [20, -10]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(3, 3, [30, -10]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		doorway = environment.doorway(30, 7, 'horizontal', [25, -20]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(30, 7, 'vertical', [10, -25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

	
		room = environment.room(30, 10, 7, 'right', [-35, -25]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		box = environment.box(15, 3, [-2, -25]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		box = environment.box(15, 3, [-18, -25]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		box = environment.box(30, 5, [-25, 15]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		#corridor = environment.corridor(30, 5, 'horizontal', [30,-10]); [self.obsticles.append(corridor.walls[x]) for x in range(0, len(corridor.walls))]

	def map6(self):

		self.dimensions = [80,80]
		# Bounding Walls ---------------------------------
		box = environment.box(80, 80, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]


		# Dynamic doorways
		
		# Randomly set state of doors
		state = bool(random.getrandbits(1))
		state = False
		# Set door states
		if state == True: 
			doorway = environment.doorway(30, 10, 'vertical', [-10, 25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		else:
			wall = environment.wall(); wall.start = [-10,40]; wall.end = [-10,10]; self.obsticles.append(wall)


		room = environment.room(20, 20, 10, 'top', [0, 0]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		doorway = environment.doorway(20, 7, 'vertical', [20, 30]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(20, 7, 'horizontal', [30, 20]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		doorway = environment.doorway(30, 10, 'vertical', [10, 25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(30, 10, 'horizontal', [25, 10]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		doorway = environment.doorway(30, 7, 'horizontal', [25, -20]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		wall = environment.wall(); wall.start = [-10,-20]; wall.end = [10,-20]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-10,-20]; wall.end = [-10,-40]; self.obsticles.append(wall)

		room = environment.room(20, 20, 7, 'top', [-30, -30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(20, 20, 7, 'bottom', [-30, 0]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		doorway = environment.doorway(30, 7, 'vertical', [-20, 25]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		box = environment.box(7, 15, [25, -5]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		#box = environment.box(15, 7, [30, 30]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

	def brl(self):

		dim_ratio = 0.6666
		size = 160
		# Bounding Walls ---------------------------------
		box = environment.box(size*dim_ratio, size, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		box = environment.box(80, 150, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		self.dimensions = [size, size*dim_ratio]


		# Dynamic doorways
		
		# Randomly set state of doors
		state = bool(random.getrandbits(1))
		state = False
		# Set door states
		if state == True: 
			wall = environment.wall(); wall.start = [-20,-4]; wall.end = [-5,-4]; self.obsticles.append(wall)


		# Central bays
		room = environment.room(4, 5, 2.5, 'right', [-16.5, -6]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(4, 5, 2.5, 'right', [-16.5, -10]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]


		room = environment.room(8, 8, 5, 'left', [-4, -8]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'left', [-4, -16]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		# Central bays
		room = environment.room(8, 8, 5, 'top', [4, -8]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'bottom', [4, -16]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]


		# offices bays
		room = environment.room(6, 8, 4, 'bottom', [15, -7]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(6, 8, 4, 'top', [15, -17]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		# Central bays right#
		room = environment.room(8, 7, 5, 'top', [25.5, -8]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 7, 5, 'top', [32.5, -8]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 12, 5, 'bottom', [30, -16]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(8, 8, 5, 'top', [43, -8]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'bottom', [43, -16]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(8, 8, 5, 'top', [51, -8]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'bottom', [51, -16]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]


		# flight arena 

		doorway = environment.doorway(16, 5, 'horizontal', [16, 0]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		wall = environment.wall(); wall.start = [24, 0]; wall.end = [24,24]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [8, 24]; wall.end = [24,24]; self.obsticles.append(wall)

		# Assisted living
		x = 4
		room = environment.double_entrance(8, 8, 5, 'side', [x, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 8, 5, 'side', [x, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 8, 5, 'side', [x, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.double_entrance(8, 8, 5, 'side', [x-11, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 8, 5, 'side', [x-11, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 8, 5, 'side', [x-11, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		x = -24

		doorway = environment.doorway(10, 5, 'horizontal', [-15.5, 0]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		room = environment.double_entrance(8, 8, 5, 'side', [x, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 8, 5, 'side', [x, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 8, 5, 'side', [x, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.double_entrance(8, 8, 5, 'side', [x-11, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 8, 5, 'side', [x-11, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 8, 5, 'side', [x-11, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		wall = environment.wall(); wall.start = [-39, 24]; wall.end = [0,24]; self.obsticles.append(wall)


		# Computer vision
		x = 28
		room = environment.room(8, 8, 5, 'right', [x, 28]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'right', [x, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'right', [x, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'right', [x, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(8, 8, 5, 'left', [x+11, 28]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'left', [x+11, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'left', [x+11, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'left', [x+11, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]



		# Kitchen
		wall = environment.wall(); wall.start = [-10,-40]; wall.end = [-10,-30]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-5,-25]; wall.end = [4,-25]; self.obsticles.append(wall)

		# workshop
		room = environment.room(15, 12, 5, 'top', [10, -32.5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(6, 16, 5, 'top', [24, -37]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(6, 13, 5, 'top', [38.5, -37]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]


		room = environment.room(6, 10, 6, 'bottom', [21, -28]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(6, 10, 6, 'bottom', [40, -28]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		#bottom right bays 
		
		room = environment.room(8, 8, 5, 'right', [49, -28]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'right', [49, -36]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]


		room = environment.room(8, 16, 5, 'left', [67, -8]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 16, 5, 'left', [67, -16]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(12, 16, 5, 'left', [67, -26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 16, 5, 'left', [67, -36]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]



		# Bottom left

		room = environment.room(16, 12, 5, 'bottom', [-64, -26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(16, 12, 5, 'bottom', [-52, -26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(16, 16, 5, 'bottom', [-38, -26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(16, 12, 5, 'bottom', [-24, -26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

	
		room = environment.room(14, 10, 5, 'top', [-34, -11]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(14, 10, 5, 'top', [-24, -11]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		doorway = environment.doorway(6, 4, 'vertical', [-18, -37]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		# Seminar room
		doorway = environment.doorway(6, 3, 'horizontal', [-55, -4]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(8, 4, 'vertical', [-70, -14]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
	

		#diag = environment.diag([-70,-10],[-65,-4], 6); [self.obsticles.append(diag.walls[x]) for x in range(0, len(diag.walls))]
		#diag = environment.diag([-75,-5],[-34, 40], 40); [self.obsticles.append(diag.walls[x]) for x in range(0, len(diag.walls))]

		wall = environment.wall(); wall.start = [-70, -4]; wall.end = [-58,-4]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-70, -10]; wall.end = [-70,-4]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-52, -4]; wall.end = [-52,-18]; self.obsticles.append(wall)



		# arthurs office

		room = environment.room(7, 5, 3, 'right', [-49.5, -7.5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(7, 5, 3, 'right', [-49.5, -14.5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(7, 5, 3, 'left', [-41.5, -7.5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(7, 5, 3, 'left', [-41.5, -14.5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		

		# farscope room
		# doorway = environment.doorway(22, 5, 'horizontal', [-50, 0]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		# #diag = environment.diag([-61,0],[-39, 24], 20); [self.obsticles.append(diag.walls[x]) for x in range(0, len(diag.walls))]
		# wall = environment.wall(); wall.start = [-39, 0]; wall.end = [-39,24]; self.obsticles.append(wall)

		room = environment.room(24, 22, 3, 'bottom', [-50,12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		
		#top rooms

		room = environment.double_entrance(16, 25, 6, 'side', [-21, 32]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		doorway = environment.doorway(8, 4, 'vertical', [24, 36]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		# top right rooms
		room = environment.room(10, 6, 3, 'bottom', [46, 5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(10, 6, 3, 'bottom', [52, 5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(12, 12, 6, 'right', [49, 16]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(10, 6, 3, 'top', [46, 27]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(10, 6, 3, 'top', [52, 27]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(12, 16, 6, 'left', [67, 18]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(12, 16, 6, 'left', [67, 30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(6, 16, 6, 'bottom', [67, 9]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

	def brl_simp(self, door_states):

		dim_ratio = 0.6666
		size = 160
		self.name = 'brl_simple'
		self.door_states = door_states
		# Bounding Walls ---------------------------------
		box = environment.box(size*dim_ratio, size, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		box = environment.box(80, 150, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		self.dimensions = [size, size*dim_ratio]

		
		# Set door states
		if door_states[5] == 1:
			wall = environment.wall(); wall.start = [24, -12]; wall.end = [30,-12]; self.obsticles.append(wall)
		if door_states[6] == 1:
			wall = environment.wall(); wall.start = [-18, -40]; wall.end = [-18,-34]; self.obsticles.append(wall)
		if door_states[7] == 1:
			wall = environment.wall(); wall.start = [-19, -6]; wall.end = [-10,-6]; self.obsticles.append(wall)
		if door_states[3] == 1:
			wall = environment.wall(); wall.start = [-75, 20]; wall.end = [-61,20]; self.obsticles.append(wall)
		if door_states[4] == 1:
			wall = environment.wall(); wall.start = [53, 8]; wall.end = [59,8]; self.obsticles.append(wall)
		if door_states[0] == 1:
			wall = environment.wall(); wall.start = [-20, 0]; wall.end = [-13,0]; self.obsticles.append(wall)
		if door_states[1] == 1:
			wall = environment.wall(); wall.start = [-20, -4]; wall.end = [-20,0]; self.obsticles.append(wall)
		if door_states[2] == 1:
			wall = environment.wall(); wall.start = [10, 0]; wall.end = [20,0]; self.obsticles.append(wall)
		

		

		# Central bays
		room = environment.room(14, 16, 8, 'left', [-2, -12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(14, 16, 8, 'right', [40, -12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		doorway = environment.doorway(10, 6, 'horizontal', [27, -12]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
	

		room = environment.room(14, 16, 8, 'top', [14, -12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]


		# flight arena 

		doorway = environment.doorway(18, 7, 'horizontal', [15, 0]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		wall = environment.wall(); wall.start = [24, 0]; wall.end = [24,24]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [6, 24]; wall.end = [24,24]; self.obsticles.append(wall)

		# Assisted living
		x = 3
		room = environment.double_entrance(8, 6, 5, 'side', [x, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		x = -25

		doorway = environment.doorway(12, 6, 'horizontal', [-16, 0]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		room = environment.double_entrance(8, 6, 5, 'side', [x, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		wall = environment.wall(); wall.start = [-39, 24]; wall.end = [0,24]; self.obsticles.append(wall)


		# # Computer vision
		x = 28

		room = environment.double_entrance(16, 17, 7, 'top', [32.5, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(12, 17, 7, 'top', [32.5, 6]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		

		# Kitchen
		wall = environment.wall(); wall.start = [-10,-40]; wall.end = [-10,-30]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-5,-25]; wall.end = [4,-25]; self.obsticles.append(wall)

		# workshop
		room = environment.room(15, 15, 7, 'top', [10, -32.5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(15, 20, 7, 'top', [27.5, -32.5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		

		#bottom right bays 
		
		room = environment.room(8, 8, 5, 'right', [49, -28]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'right', [49, -36]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]


		room = environment.room(8, 16, 5, 'left', [67, -8]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 16, 5, 'left', [67, -16]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(12, 16, 5, 'left', [67, -26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 16, 5, 'left', [67, -36]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]



		# Bottom left

		room = environment.room(16, 12, 5, 'bottom', [-64, -26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(16, 12, 5, 'bottom', [-52, -26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(16, 16, 5, 'bottom', [-38, -26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(16, 12, 5, 'bottom', [-24, -26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

	
		room = environment.room(14, 10, 5, 'top', [-34, -11]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(14, 10, 5, 'top', [-24, -11]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		doorway = environment.doorway(6, 4, 'vertical', [-18, -37]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		# Seminar room
		doorway = environment.doorway(7, 5, 'horizontal', [-56, -4]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
		doorway = environment.doorway(8, 6, 'vertical', [-70, -14]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]
	
		wall = environment.wall(); wall.start = [-70, -4]; wall.end = [-58,-4]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-70, -10]; wall.end = [-70,-4]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-52, -4]; wall.end = [-52,-18]; self.obsticles.append(wall)



		# arthurs office
		room = environment.room(24, 22, 7, 'bottom', [-50,12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		
		#top rooms

		room = environment.double_entrance(16, 25, 6, 'side', [-21, 32]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		doorway = environment.doorway(12, 6, 'vertical', [24, 34]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		# top right rooms
		room = environment.room(10, 12, 6, 'bottom', [47, 5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(10, 12, 6, 'top', [47, 27]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		#room = environment.room(10, 6, 3, 'bottom', [52, 5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(12, 12, 7, 'right', [47, 16]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		#room = environment.room(10, 6, 3, 'top', [46, 27]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		#room = environment.room(10, 6, 3, 'top', [52, 27]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(12, 16, 6, 'left', [67, 18]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(12, 16, 6, 'left', [67, 30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(6, 16, 6, 'bottom', [67, 9]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

	def brl_mod(self, door_states):

		dim_ratio = 0.6666
		size = 160
		self.name = 'brl_simple'
		self.door_states = door_states
		# Bounding Walls ---------------------------------
		box = environment.box(size*dim_ratio, size, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		box = environment.box(80, 150, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		self.dimensions = [size, size*dim_ratio]

		
		# Set door states
		if door_states[0] == 1:
			wall = environment.wall(); wall.start = [-42, 18]; wall.end = [-34,18]; self.obsticles.append(wall)
		if door_states[1] == 1:
			wall = environment.wall(); wall.start = [4, -10]; wall.end = [14,-10]; self.obsticles.append(wall)
		if door_states[2] == 1:
			wall = environment.wall(); wall.start = [-18, 0]; wall.end = [-18,-10]; self.obsticles.append(wall)
		if door_states[3] == 1:
			wall = environment.wall(); wall.start = [-4, -7]; wall.end = [4,-7]; self.obsticles.append(wall)
		

		#Center column
		
		x = -22

		room = environment.room(8, 8, 5, 'right', [x, -28]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 8, 5, 'right', [x, -36]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]


		room = environment.room(14, 8, 6, 'left', [x+14, -7]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		wall = environment.wall(); wall.start = [4, -20]; wall.end = [4,-5]; self.obsticles.append(wall)

		room = environment.room(12, 16, 5, 'right', [x+18, -26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 16, 5, 'right', [x+18, -36]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		# top right rooms
		room = environment.room(10, 12, 8, 'bottom', [x-2, 5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(10, 12, 6, 'top', [x-2, 27]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(12, 12, 7, 'right', [x-2, 16]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(12, 16, 6, 'left', [x+18, 14]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(12, 16, 6, 'left', [x+18, 26]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 16, 5, 'left', [x+18, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]



		# Bottom left
		room = environment.room(8, 20, 6, 'top', [-36, -36]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(8, 20, 6, 'top', [-65, -36]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(15, 12, 7, 'right', [-69, -16.5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		

		
		y = -16.5
		x = -25.5

		room = environment.room(15, 15, 7, 'top', [x, y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(15, 20, 7, 'left', [x-17.5, y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		

		# Middle left

		room = environment.room(10, 10, 6, 'left', [-42, -4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(10, 10, 6, 'left', [-42, 6]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		
		room = environment.room(10, 12, 7, 'right', [-69, -4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(10, 12, 7, 'right', [-69, 6]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		
		# top left

		room = environment.double_entrance(14, 16, 8, 'top', [-38, 25]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(14, 12, 7, 'right', [-69, 25]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(14, 8, 7, 'left', [-50, 25]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		


		x = 20
		y = -26
		# Bottom right

		room = environment.room(16, 12, 5, 'bottom', [x, y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(16, 12, 5, 'bottom', [x+12, y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(16, 16, 5, 'bottom', [x+26  , y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(16, 12, 5, 'bottom', [x + 40 , y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		y = -12

		room = environment.room(12, 12, 5, 'top', [x, y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(12, 12, 5, 'top', [x+12, y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(12, 16, 5, 'top', [x+26  , y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(12, 12, 5, 'top', [x + 40 , y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]



		# top right
		y = 35
		room = environment.room(10, 12, 6, 'bottom', [20, y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(10, 12, 6, 'bottom', [32, y]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(12, 12, 7, 'bottom', [54, 6]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(12, 12, 7, 'right', [54, 18]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(10, 12, 6, 'right', [54, 29]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.double_entrance(12, 15, 7, 'top', [67.5, 6]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]


		# Assisted living
		x = 18
		room = environment.double_entrance(8, 6, 5, 'side', [x, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		wall = environment.wall(); wall.start = [20, 24]; wall.end = [32,24]; self.obsticles.append(wall)

		x = 45

		doorway = environment.doorway(12, 6, 'horizontal', [x-19, 0]); [self.obsticles.append(doorway.walls[x]) for x in range(0, len(doorway.walls))]

		room = environment.double_entrance(8, 6, 5, 'side', [x, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 12]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.double_entrance(8, 6, 5, 'side', [x-11, 4]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		wall = environment.wall(); wall.start = [10, 24]; wall.end = [20,24]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [35, 24]; wall.end = [45,24]; self.obsticles.append(wall)


	def maze(self):

		dim_ratio = 1
		size = 80
	
		# Bounding Walls ---------------------------------
		box = environment.box(size*dim_ratio, size, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		box = environment.box(80, 80, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		self.dimensions = [size, size*dim_ratio]

		wall = environment.wall(); wall.start = [-40, -30]; wall.end = [-10,-30]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-10, -30]; wall.end = [-10,-10]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [0, -40]; wall.end = [0,0]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-20, 0]; wall.end = [0,0]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-20, 0]; wall.end = [-20,-20]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-20, -20]; wall.end = [-30,-20]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-30, -20]; wall.end = [-30,0]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-40, 10]; wall.end = [10,10]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [10, 10]; wall.end = [10,-30]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [10, -30]; wall.end = [20,-30]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [20, -30]; wall.end = [20,-20]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [20, -20]; wall.end = [30,-20]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [30, -40]; wall.end = [30,-30]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [30, -20]; wall.end = [30,0]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [30, 0]; wall.end = [20,0]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [40, 10]; wall.end = [20,10]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [20, 10]; wall.end = [20,20]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [20, 20]; wall.end = [-10,20]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-20, 30]; wall.end = [-20,10]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-20, 30]; wall.end = [30,30]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [30, 30]; wall.end = [30,20]; self.obsticles.append(wall)


	def room1(self):


		dim_ratio = 1
		size = 20

		dimx = 20
		dimy = 20
		# Bounding Walls ---------------------------------
		#box = environment.box(size*dim_ratio, size, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		self.dimensions = [size, size*dim_ratio]

	def room2(self):


		dim_ratio = 1
		size = 20

		dimx = 20
		dimy = 20
		# Bounding Walls ---------------------------------
		#box = environment.box(size*dim_ratio, size, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -(dimy)/2]; wall.end = [0,0]; self.obsticles.append(wall)

		self.dimensions = [size, size*dim_ratio]

	def room3(self):


		dim_ratio = 1
		size = 20

		dimx = 20
		dimy = 20
		# Bounding Walls ---------------------------------
		#box = environment.box(size*dim_ratio, size, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -7.5]; wall.end = [0,7.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-7.5,0]; wall.end = [7.5,0]; self.obsticles.append(wall)

		self.dimensions = [size, size*dim_ratio]

	def room4(self):


		dim_ratio = 1
		size = 20

		dimx = 20
		dimy = 20
		# Bounding Walls ---------------------------------
		#box = environment.box(size*dim_ratio, size, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -(dimy)/2]; wall.end = [0,-2]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [0,2]; wall.end = [0, (dimy)/2]; self.obsticles.append(wall)

		self.dimensions = [size, size*dim_ratio]

	def room44(self):


		dim_ratio = 1
		size = 20

		dimx = 40
		dimy = 20
		# Bounding Walls ---------------------------------
		#box = environment.box(size*dim_ratio, size, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		if self.bounded == True:
			box = environment.box(dimy, dimx, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [-10, -10]; wall.end = [-10,-2]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-10,2]; wall.end = [-10, 10]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [10, -10]; wall.end = [10,-2]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [10,2]; wall.end = [10, 10]; self.obsticles.append(wall)

		self.dimensions = [dimy, dimx]

	def room5(self):


		dim_ratio = 1
		size = 20

		dimx = 20
		dimy = 20
		# Bounding Walls ---------------------------------
		#box = environment.box(size*dim_ratio, size, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]
		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -dimy/2]; wall.end = [0,(dimy/2) - 5]; self.obsticles.append(wall)
		

		self.dimensions = [size, size*dim_ratio]

	def room6(self):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20
		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -7.5]; wall.end = [0,0]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-7.5, 0]; wall.end = [0,0]; self.obsticles.append(wall)

		self.dimensions = [size, size*dim_ratio]

	def room66(self):

		dim_ratio = 1
		size = 20
		dimx = 40
		dimy = 20
		if self.bounded == True:
			box = environment.box(dimy, dimx, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [10, -7.5]; wall.end = [10,0]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [2.5, 0]; wall.end = [10,0]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-10, -7.5]; wall.end = [-10,0]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-17.5, 0]; wall.end = [-10,0]; self.obsticles.append(wall)

		self.dimensions = [dimy, dimx]

	def room67(self):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20
		if self.bounded == True:
			box = environment.box(dimx, 2*dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [5, -7.5]; wall.end = [5,7.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [15, -7.5]; wall.end = [15,7.5]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [-10, -7.5]; wall.end = [-10,0]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-17.5, 0]; wall.end = [-10,0]; self.obsticles.append(wall)

		self.dimensions = [size, size*dim_ratio]

	def room7(self):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20
		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [-4, -7.5]; wall.end = [-4,7.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [4, -7.5]; wall.end = [4,7.5]; self.obsticles.append(wall)

		self.dimensions = [size, size*dim_ratio]

	def room8(self):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20

		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -7.5]; wall.end = [0,7.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-5, -7.5]; wall.end = [5,-7.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-5, 7.5]; wall.end = [5,7.5]; self.obsticles.append(wall)

		self.dimensions = [size, size*dim_ratio]

	def room9(self):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20

		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -7]; wall.end = [0,7]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-6, -7]; wall.end = [0,-7]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-6, 7]; wall.end = [0,7]; self.obsticles.append(wall)

		self.dimensions = [size, size*dim_ratio]

	def room10(self):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20

		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -10]; wall.end = [0,-2.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-10, 0]; wall.end = [-2.5,0]; self.obsticles.append(wall)

		self.dimensions = [size, size*dim_ratio]

	def room11(self):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20

		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -10]; wall.end = [0,-2.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-10, 0]; wall.end = [-2.5,0]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [2.5, 0]; wall.end = [10,0]; self.obsticles.append(wall)

		self.dimensions = [size, size*dim_ratio]

	def room12(self):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20

		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -10]; wall.end = [0,-2.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [0, 10]; wall.end = [0,2.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-10, 0]; wall.end = [-2.5,0]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [2.5, 0]; wall.end = [10,0]; self.obsticles.append(wall)

		self.dimensions = [size, size*dim_ratio]

	def room13(self):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20

		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -10]; wall.end = [0,7]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-7.5, 0]; wall.end = [7.5,0]; self.obsticles.append(wall)
	

		self.dimensions = [size, size*dim_ratio]

	def room14(self):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20

		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		wall = environment.wall(); wall.start = [0, -10]; wall.end = [0,7.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-7.5, 0]; wall.end = [7.5,0]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-7.5, 7.5]; wall.end = [7.5,7.5]; self.obsticles.append(wall)
	

		self.dimensions = [size, size*dim_ratio]


	def roomA(self, origin, degrees):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20

		walls = []


		wall = environment.wall(); wall.start = [0 , -10 ]; wall.end = [0 ,-2 ]; walls.append(wall)
		wall = environment.wall(); wall.start = [0 , 10 ]; wall.end = [0 ,2 ]; walls.append(wall)

		wall = environment.wall(); wall.start = [10 , -10 ]; wall.end = [10 ,10 ]; walls.append(wall)
		wall = environment.wall(); wall.start = [0 , 10 ]; wall.end = [10 ,10 ]; walls.append(wall)
		wall = environment.wall(); wall.start = [0 , -10 ]; wall.end = [10 ,-10 ]; walls.append(wall)
		
		# Rotate module
		degrees = math.radians(degrees)
		
		for i in range(0, len(walls)):

			x1 = walls[i].end[0]
			y1 = walls[i].end[1]
			x2 = walls[i].start[0]
			y2 = walls[i].start[1]
			# Rounding to one decimal place needed otherwise some walls are not
			# detected as horizontal/vertical and agents pass through them.

			walls[i].end[0] = round(x1*math.cos(degrees) + y1*math.sin(degrees), 1)
			walls[i].end[1] = round(-x1*math.sin(degrees) + y1*math.cos(degrees), 1)
			walls[i].start[0] = round(x2*math.cos(degrees) + y2*math.sin(degrees), 1)
			walls[i].start[1] = round(-x2*math.sin(degrees) + y2*math.cos(degrees), 1)

		# Perform origin shift after rotation
		for i in range(0, len(walls)):
			walls[i].end[0] += origin[0]
			walls[i].end[1] += origin[1]
			walls[i].start[0] += origin[0]
			walls[i].start[1] += origin[1]
			self.obsticles.append(walls[i])

		self.dimensions = [dimx, dimy]


	def roomB(self, origin, degrees):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20

		walls = []

		wall = environment.wall(); wall.start = [-10, -10]; wall.end = [10,-10]; walls.append(wall)
		wall = environment.wall(); wall.start = [10, -10]; wall.end = [10,10]; walls.append(wall)
		wall = environment.wall(); wall.start = [-10, 10]; wall.end = [10,10]; walls.append(wall)

		wall = environment.wall(); wall.start = [-10, 10]; wall.end = [-10,0]; walls.append(wall)
		wall = environment.wall(); wall.start = [-10, 0]; wall.end = [0,0]; walls.append(wall)

		degrees = math.radians(degrees)		
		for i in range(0, len(walls)):

			x1 = walls[i].end[0]
			y1 = walls[i].end[1]
			x2 = walls[i].start[0]
			y2 = walls[i].start[1]
			# Rounding to one decimal place needed otherwise some walls are not
			# detected as horizontal/vertical and agents pass through them.

			walls[i].end[0] = round(x1*math.cos(degrees) + y1*math.sin(degrees), 1)
			walls[i].end[1] = round(-x1*math.sin(degrees) + y1*math.cos(degrees), 1)
			walls[i].start[0] = round(x2*math.cos(degrees) + y2*math.sin(degrees), 1)
			walls[i].start[1] = round(-x2*math.sin(degrees) + y2*math.cos(degrees), 1)

		# Perform origin shift after rotation
		for i in range(0, len(walls)):
			walls[i].end[0] += origin[0]
			walls[i].end[1] += origin[1]
			walls[i].start[0] += origin[0]
			walls[i].start[1] += origin[1]
			self.obsticles.append(walls[i])

		self.dimensions = [dimx, dimy]

	def roomC(self, origin, degrees):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20
		walls = []

		wall = environment.wall(); wall.start = [-10, -10 ]; wall.end = [10 ,-10 ]; walls.append(wall)
		wall = environment.wall(); wall.start = [-10, 10 ]; wall.end = [10 ,10 ]; walls.append(wall)
		
		wall = environment.wall(); wall.start = [-4 , -7.5 ]; wall.end = [-4 ,7.5 ]; walls.append(wall)
		wall = environment.wall(); wall.start = [4 , -7.5 ]; wall.end = [4 ,7.5 ]; walls.append(wall)

		degrees = math.radians(degrees)		
		for i in range(0, len(walls)):

			x1 = walls[i].end[0]
			y1 = walls[i].end[1]
			x2 = walls[i].start[0]
			y2 = walls[i].start[1]
			# Rounding to one decimal place needed otherwise some walls are not
			# detected as horizontal/vertical and agents pass through them.

			walls[i].end[0] = round(x1*math.cos(degrees) + y1*math.sin(degrees), 1)
			walls[i].end[1] = round(-x1*math.sin(degrees) + y1*math.cos(degrees), 1)
			walls[i].start[0] = round(x2*math.cos(degrees) + y2*math.sin(degrees), 1)
			walls[i].start[1] = round(-x2*math.sin(degrees) + y2*math.cos(degrees), 1)

		# Perform origin shift after rotation
		for i in range(0, len(walls)):
			walls[i].end[0] += origin[0]
			walls[i].end[1] += origin[1]
			walls[i].start[0] += origin[0]
			walls[i].start[1] += origin[1]
			self.obsticles.append(walls[i])

		self.dimensions = [dimx, dimy]
		self.source = [-10, -10, 10]
		self.sink = [10, -10, 10]

	def roomCC(self):

		dim_ratio = 1
		size = 20
		dimx = 40
		dimy = 20
		walls = []

		wall = environment.wall(); wall.start = [-20, -10]; wall.end = [20,-10]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-20, 10]; wall.end = [20,10]; self.obsticles.append(wall)
		
		wall = environment.wall(); wall.start = [-14, -7.5]; wall.end = [-14,7.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-6, -7.5]; wall.end = [-6,7.5]; self.obsticles.append(wall)

		wall = environment.wall(); wall.start = [6, -7.5]; wall.end = [6,7.5]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [14, -7.5]; wall.end = [14,7.5]; self.obsticles.append(wall)
		self.dimensions = [dimy, dimx]

	def roomCD(self, origin, degrees):


		self.roomC([-10,0], 0)
		self.roomD([10,0], 0)

		size = 20
		dimx = 40
		dimy = 20
		self.dimensions = [dimx, dimy]

	def roomCCA(self, origin, degrees):


		self.roomC([-20,0], 0)
		self.roomC([0,0], 0)
		self.roomA([20,0], 0)

		size = 20
		dimx = 60
		dimy = 20
		self.dimensions = [dimx, dimy]

	def roomDC(self, origin, degrees):


		self.roomD([-10,0], 0)
		self.roomC([10,0], 0)

		size = 20
		dimx = 40
		dimy = 20
		self.dimensions = [dimx, dimy]

	def roomDD180(self, origin, degrees):


		self.roomD([-10,0], 0)
		self.roomD([10,0], 180)

		size = 20
		dimx = 40
		dimy = 20
		self.dimensions = [dimx, dimy]

	def roomCA(self, origin, degrees):


		self.roomC([-10,0], 0)
		self.roomA([10,0], 0)

		size = 20
		dimx = 40
		dimy = 20
		self.dimensions = [dimx, dimy]

	def roomDA(self, origin, degrees):


		self.roomD([-10,0], 0)
		self.roomA([10,0], 0)

		size = 20
		dimx = 40
		dimy = 20
		self.dimensions = [dimx, dimy]

	def roomADA(self, origin, degrees):


		self.roomA([-20,0], 180)
		self.roomD([0,0], 180)
		self.roomA([20,0], 0)

		size = 20
		dimx = 60
		dimy = 20
		self.dimensions = [dimx, dimy]

	def roomACA(self, origin, degrees):


		self.roomA([-20,0], 180)
		self.roomC([0,0], 0)
		self.roomA([20,0], 0)

		size = 20
		dimx = 60
		dimy = 20
		self.dimensions = [dimx, dimy]



	def roomD(self, origin, degrees):

		dim_ratio = 1
		size = 20
		dimx = 20
		dimy = 20

		walls = []

		wall = environment.wall(); wall.start = [-10, -10]; wall.end = [10,-10]; walls.append(wall)
		wall = environment.wall(); wall.start = [-10, 10]; wall.end = [10,10]; walls.append(wall)
		
		wall = environment.wall(); wall.start = [0, -10]; wall.end = [0,6]; walls.append(wall)
		
		# Code to rotate module
		degrees = math.radians(degrees)		
		for i in range(0, len(walls)):

			x1 = walls[i].end[0]
			y1 = walls[i].end[1]
			x2 = walls[i].start[0]
			y2 = walls[i].start[1]
			# Rounding to one decimal place needed otherwise some walls are not
			# detected as horizontal/vertical and agents pass through them.

			walls[i].end[0] = round(x1*math.cos(degrees) + y1*math.sin(degrees), 1)
			walls[i].end[1] = round(-x1*math.sin(degrees) + y1*math.cos(degrees), 1)
			walls[i].start[0] = round(x2*math.cos(degrees) + y2*math.sin(degrees), 1)
			walls[i].start[1] = round(-x2*math.sin(degrees) + y2*math.cos(degrees), 1)


		# Perform origin shift after rotation
		for i in range(0, len(walls)):
			walls[i].end[0] += origin[0]
			walls[i].end[1] += origin[1]
			walls[i].start[0] += origin[0]
			walls[i].start[1] += origin[1]
			self.obsticles.append(walls[i])


		self.dimensions = [dimx, dimy]


	def roomDD(self):

		dim_ratio = 1
		size = 20
		dimx = 40
		dimy = 20

		wall = environment.wall(); wall.start = [-20, -10]; wall.end = [20,-10]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [-20, 10]; wall.end = [20,10]; self.obsticles.append(wall)
		
		wall = environment.wall(); wall.start = [-10, -10]; wall.end = [-10,6]; self.obsticles.append(wall)
		wall = environment.wall(); wall.start = [10, -10]; wall.end = [10,6]; self.obsticles.append(wall)
		
		self.dimensions = [dimx, dimy]


	def roomX(self, origin, degrees):


		self.roomA([30,-10], 270)
		self.roomA([30,10], 270)
		self.roomA([30,10], 90)
		self.roomA([30,-10], 90)

		self.roomC([10,10], 90)
		self.roomC([10,10], -90)

		self.roomC([-10,10], 90)
		self.roomC([-10,10], -90)

		dim_ratio = 1
		size = 20
		dimx = 40
		dimy = 60
		self.dimensions = [dimx, dimy]

	def roomY(self, origin, degrees):
		# 3x3 module environment

		if self.bounded == True:
			box = environment.box(60, 60, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		# top row
		self.roomA([-20,20], 180)
		self.roomD([0,20], 0)
		self.roomD([20,20], 180)

		# middle row
		self.roomA([-20,0], 180)
		self.roomD([0,0], 180)
		self.roomA([20,0], 0)

		# bottom row
		self.roomD([-20,-20], 0)
		self.roomC([0,-20], 0)
		self.roomA([20,-20], 0)
		
		dim_ratio = 1
		size = 20
		dimx = 60
		dimy = 60
		self.dimensions = [dimx, dimy]

	def roomZ(self, origin, degrees):
		# 3x3 module environment

		if self.bounded == True:
			box = environment.box(60, 60, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		# top row
		self.roomC([-20,20], 0)
		self.roomC([0,20], 0)
		self.roomA([20,20], 0)

		# middle row
		self.roomC([-20,0], 90)
		self.roomA([0,0], -90)
		self.roomC([20,0], 90)

		# bottom row
		self.roomA([-20,-20], 180)
		self.roomA([0,-20], 90)
		self.roomA([20,-20], 0)
		
		dim_ratio = 1
		size = 20
		dimx = 60
		dimy = 60
		self.dimensions = [dimx, dimy]

	def roomAlpha1(self, origin, degrees):
		# 3x3 module environment

		if self.bounded == True:
			box = environment.box(60, 60, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		# top row
		self.roomC([-20,20], 0)
		self.roomC([0,20], 0)
		self.roomC([20,20], 0)

		# middle row
		self.roomC([-20,0], 0)
		self.roomC([0,0], 0)
		self.roomC([20,0], 0)

		# bottom row
		self.roomC([-20,-20], 0)
		self.roomC([0,-20], 0)
		self.roomC([20,-20], 0)
		
		dim_ratio = 1
		size = 20
		dimx = 60
		dimy = 60
		self.dimensions = [dimx, dimy]

	def roomAlpha2(self, origin, degrees):
		# 3x3 module environment

		if self.bounded == True:
			box = environment.box(60, 60, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		# top row
		self.roomC([-20,20], 0)
		self.roomC([0,20], 0)
		self.roomD([20,20], 0)

		# middle row
		self.roomC([-20,0], 0)
		self.roomC([0,0], 0)
		self.roomD([20,0], 0)

		# bottom row
		self.roomC([-20,-20], 0)
		self.roomC([0,-20], 0)
		self.roomD([20,-20], 0)
		
		dim_ratio = 1
		size = 20
		dimx = 60
		dimy = 60
		self.dimensions = [dimx, dimy]

	def roomAlpha3(self, origin, degrees):
		# 3x3 module environment

		if self.bounded == True:
			box = environment.box(60, 60, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		# top row
		self.roomC([-20,20], 0)
		self.roomD([0,20], 0)
		self.roomC([20,20], 0)

		# middle row
		self.roomD([-20,0], 0)
		self.roomC([0,0], 0)
		self.roomD([20,0], 0)

		# bottom row
		self.roomC([-20,-20], 0)
		self.roomD([0,-20], 0)
		self.roomC([20,-20], 0)
		
		dim_ratio = 1
		size = 20
		dimx = 60
		dimy = 60
		self.dimensions = [dimx, dimy]

	def roomAlpha4(self, origin, degrees):
		# 3x3 module environment

		if self.bounded == True:
			box = environment.box(60, 60, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		# top row
		self.roomC([-20,20], 0)
		self.roomD([0,20], 0)
		self.roomD([20,20], 0)

		# middle row
		self.roomC([-20,0], 0)
		self.roomD([0,0], 0)
		self.roomD([20,0], 0)

		# bottom row
		self.roomC([-20,-20], 0)
		self.roomD([0,-20], 0)
		self.roomD([20,-20], 0)
		
		dim_ratio = 1
		size = 20
		dimx = 60
		dimy = 60
		self.dimensions = [dimx, dimy]

	def roomAlpha5(self, origin, degrees):
		# 3x3 module environment

		if self.bounded == True:
			box = environment.box(60, 60, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		# top row
		self.roomD([-20,20], 0)
		self.roomD([0,20], 0)
		self.roomD([20,20], 0)

		# middle row
		self.roomD([-20,0], 0)
		self.roomD([0,0], 0)
		self.roomD([20,0], 0)

		# bottom row
		self.roomD([-20,-20], 0)
		self.roomD([0,-20], 0)
		self.roomD([20,-20], 0)
		
		dim_ratio = 1
		size = 20
		dimx = 60
		dimy = 60
		self.dimensions = [dimx, dimy]

	def roomAlpha6(self, origin, degrees):
		# 3x3 module environment

		if self.bounded == True:
			box = environment.box(60, 60, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		# top row
		self.roomC([-20,20], 0)
		self.roomD([0,20], 0)
		self.roomC([20,20], 0)

		# middle row
		self.roomC([-20,0], 0)
		self.roomD([0,0], 0)
		self.roomC([20,0], 0)

		# bottom row
		self.roomD([-20,-20], 0)
		self.roomD([0,-20], 0)
		self.roomD([20,-20], 0)
		
		dim_ratio = 1
		size = 20
		dimx = 60
		dimy = 60
		self.dimensions = [dimx, dimy]

	def bench1(self, origin, degrees):

		dim_ratio = 1
		size = 20
		dimx = 80
		dimy = 80

		walls = []

		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		# bottom left snake
		wall = environment.wall(); wall.start = [-10,-30]; wall.end = [-10,30]; walls.append(wall)
		wall = environment.wall(); wall.start = [-30, -30]; wall.end = [-10,-30]; walls.append(wall)
		wall = environment.wall(); wall.start = [-40, -20]; wall.end = [-20,-20]; walls.append(wall)
		wall = environment.wall(); wall.start = [-30, -10]; wall.end = [-10,-10]; walls.append(wall)
		

		wall = environment.wall(); wall.start = [0, -40]; wall.end = [0,-10]; walls.append(wall)
		wall = environment.wall(); wall.start = [0, -10]; wall.end = [20,-10]; walls.append(wall)
		wall = environment.wall(); wall.start = [20, -10]; wall.end = [20,10]; walls.append(wall)

		wall = environment.wall(); wall.start = [0, 0]; wall.end = [10,0]; walls.append(wall)
		wall = environment.wall(); wall.start = [10, 0]; wall.end = [10,20]; walls.append(wall)
		wall = environment.wall(); wall.start = [10, 20]; wall.end = [30,20]; walls.append(wall)
		wall = environment.wall(); wall.start = [20, 10]; wall.end = [30,10]; walls.append(wall)

		wall = environment.wall(); wall.start = [20, 0]; wall.end = [30,0]; walls.append(wall)

		room = environment.room(20, 20, 8, 'top', [10, -30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(20, 20, 8, 'top', [30, -30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		wall = environment.wall(); wall.start = [20, 20]; wall.end = [20,30]; walls.append(wall)

		room = environment.room(20, 10, 8, 'left', [5, 10]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(20, 20, 8, 'bottom', [-30, 30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		

		wall = environment.wall(); wall.start = [-20, 40]; wall.end = [-20,10]; walls.append(wall)


		# Code to rotate module
		degrees = math.radians(degrees)		
		for i in range(0, len(walls)):

			x1 = walls[i].end[0]
			y1 = walls[i].end[1]
			x2 = walls[i].start[0]
			y2 = walls[i].start[1]
			# Rounding to one decimal place needed otherwise some walls are not
			# detected as horizontal/vertical and agents pass through them.

			walls[i].end[0] = round(x1*math.cos(degrees) + y1*math.sin(degrees), 1)
			walls[i].end[1] = round(-x1*math.sin(degrees) + y1*math.cos(degrees), 1)
			walls[i].start[0] = round(x2*math.cos(degrees) + y2*math.sin(degrees), 1)
			walls[i].start[1] = round(-x2*math.sin(degrees) + y2*math.cos(degrees), 1)

		# Perform origin shift after rotation
		for i in range(0, len(walls)):
			walls[i].end[0] += origin[0]
			walls[i].end[1] += origin[1]
			walls[i].start[0] += origin[0]
			walls[i].start[1] += origin[1]
			self.obsticles.append(walls[i])


		self.dimensions = [dimx, dimy]
		self.swarm_origin = np.array([-5,-35])

	def bench2(self, origin, degrees):

		dim_ratio = 1
		size = 20
		dimx = 80
		dimy = 80

		walls = []

		if self.bounded == True:
			box = environment.box(dimx, dimy, [0, 0]); [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

		# bottom left snake
		wall = environment.wall(); wall.start = [-30,-30]; wall.end = [-30,30]; walls.append(wall)
		wall = environment.wall(); wall.start = [-30,-30]; wall.end = [10,-30]; walls.append(wall)
		wall = environment.wall(); wall.start = [20,-40]; wall.end = [20,-20]; walls.append(wall)
		wall = environment.wall(); wall.start = [20,-20]; wall.end = [-10,-20]; walls.append(wall)
		wall = environment.wall(); wall.start = [-30,-20]; wall.end = [-20,-20]; walls.append(wall)

		wall = environment.wall(); wall.start = [-30,30]; wall.end = [0,30]; walls.append(wall)
		

		room = environment.room(20, 20, 8, 'left', [20, 20]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		
		wall = environment.wall(); wall.start = [40,0]; wall.end = [20,0]; walls.append(wall)
		wall = environment.wall(); wall.start = [10,10]; wall.end = [10,-10]; walls.append(wall)
		
		room = environment.room(10, 20, 8, 'bottom', [30, -5]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]
		room = environment.room(20, 20, 8, 'top', [30, -30]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		room = environment.room(30, 20, 8, 'right', [-20, 15]); [self.obsticles.append(room.walls[x]) for x in range(0, len(room.walls))]

		wall = environment.wall(); wall.start = [-20,-10]; wall.end = [10,-10]; walls.append(wall)



		# Code to rotate module
		degrees = math.radians(degrees)		
		for i in range(0, len(walls)):

			x1 = walls[i].end[0]
			y1 = walls[i].end[1]
			x2 = walls[i].start[0]
			y2 = walls[i].start[1]
			# Rounding to one decimal place needed otherwise some walls are not
			# detected as horizontal/vertical and agents pass through them.

			walls[i].end[0] = round(x1*math.cos(degrees) + y1*math.sin(degrees), 1)
			walls[i].end[1] = round(-x1*math.sin(degrees) + y1*math.cos(degrees), 1)
			walls[i].start[0] = round(x2*math.cos(degrees) + y2*math.sin(degrees), 1)
			walls[i].start[1] = round(-x2*math.sin(degrees) + y2*math.cos(degrees), 1)

		# Perform origin shift after rotation
		for i in range(0, len(walls)):
			walls[i].end[0] += origin[0]
			walls[i].end[1] += origin[1]
			walls[i].start[0] += origin[0]
			walls[i].start[1] += origin[1]
			self.obsticles.append(walls[i])


		self.dimensions = [dimx, dimy]




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
	# A = 10; B = 10
	# For smaller environments
	# A = 2; B = 5

	A = 10; B = 20

	# Compute force based vector and multiply by intersection matrix
	Fy = np.exp(-A*np.abs(diffh) + B)*diffh*intmat
	#Fy = .1*np.power(diffh, -1*np.ones((len(diffh), len(diffh[0]))))*intmat

	#Fy = -3/diffh*intmat*np.exp(-abs(diffh) + 3)
	#Fy = Fy*diffh*intmat

	low = agentsy[:, np.newaxis] >= map.limv.T[0]
	up = agentsy[:, np.newaxis] <= map.limv.T[1]
	intmat = up*low
	now = time.time()
	Fx = np.exp(-A*np.abs(diffv) + B)*diffv*intmat

	#Fx = .2*np.power(diffv, -1*np.ones((len(diffv), len(diffv[0]))))*intmat
	#Fx = -3/diffv*intmat*np.exp(-abs(diffv) + 3)
	#Fx = Fx*diffv*intmat

	# Sum the forces between every wall into one force.
	Fx = np.sum(Fx, axis=1)
	Fy = np.sum(Fy, axis=1)
	# Combine x and y force vectors
	#F = np.array([[Fx[n], Fy[n]] for n in range(size)])
	F = np.stack((Fx, Fy), axis = 1)
	taken = 1500*1000*(time.time()- now)
	#print('taken = ', taken)


	return F

def continuous_boundary(agents, map):

	# Check if agent passes wall bounds. i.e. does agent intersect with area.

	# If yes, which side has the agent passed through?

	# Mirror agent back around to opposite wall.

	# split agent positions into x and y arrays
	agentsx = agents.T[0]
	agentsy = agents.T[1]

	# Set boundary size relative to environment dimensions
	scale = 2

	# Check left and right boundaries
	right = agentsx >=  scale*(map.dimensions[0]/2)
	left = agentsx <= -scale*(map.dimensions[0]/2)

	agentsx += -scale*(map.dimensions[0])*right
	agentsx += scale*(map.dimensions[0])*left

	# Check top and bottom boundaries
	top = agentsy >=  scale*(map.dimensions[1]/2)
	bottom = agentsy <= -scale*(map.dimensions[1]/2)

	agentsy += -scale*(map.dimensions[1])*top
	agentsy += scale*(map.dimensions[1])*bottom

	agents = np.stack((agentsx, agentsy), axis = 1)	

	return agents


def boundary_death(swarm, map):

	# Check if agent passes wall bounds. i.e. does agent intersect with area.

	# If yes, which side has the agent passed through?

	# Mirror agent back around to opposite wall.

	# split agent positions into x and y arrays
	agentsx = swarm.agents.T[0]
	agentsy = swarm.agents.T[1]

	scale = 1.0

	# Check left and right boundaries
	right = agentsx >=  scale*(map.dimensions[1]/2)
	left = agentsx <= -scale*(map.dimensions[1]/2)

	# Check top and bottom boundaries
	top = agentsy >=  scale*(map.dimensions[0]/2)
	bottom = agentsy <= -scale*(map.dimensions[0]/2)

	death = np.logical_or(np.logical_or(left,right) , np.logical_or(top,bottom))

	randx = np.random.uniform(-(map.dimensions[0])/2, (map.dimensions[0])/2, swarm.size)
	randy = np.random.uniform(-(map.dimensions[1])/2, (map.dimensions[1])/2, swarm.size)
	random_pos = np.stack((randx, randy), axis = 1)

	new_pos = death[:, np.newaxis]*random_pos

	# Set dead agents pos to origin
	swarm.agents = np.logical_not(death)[:,np.newaxis]*swarm.agents
	# Add new positions only to dead agents
	swarm.agents += new_pos

	return np.sum(death)

def boundary_death_flow(swarm, map):

	# Check if agent passes wall bounds. i.e. does agent intersect with area.

	# If yes, which side has the agent passed through?

	# Mirror agent back around to opposite wall.

	# split agent positions into x and y arrays
	agentsx = swarm.agents.T[0]
	agentsy = swarm.agents.T[1]

	scale = 1.0

	# source death
	source_death = agentsx <= map.source[0]

	# sink death
	sink_death = agentsx >= map.sink[0]

	death = np.logical_or(sink_death, source_death)

	randx = np.random.uniform(-(map.dimensions[0])/2, (map.dimensions[0])/2, swarm.size)
	randy = np.random.uniform(-(map.dimensions[1])/2, (map.dimensions[1])/2, swarm.size)
	random_pos = np.stack((randx, randy), axis = 1)

	new_pos = death[:, np.newaxis]*random_pos

	# Set dead agents pos to origin
	swarm.agents = np.logical_not(death)[:,np.newaxis]*swarm.agents
	# Add new positions only to dead agents
	swarm.agents += new_pos

	return np.sum(sink_death)

def boundary_loss(swarm, map):

	# Check if agent passes wall bounds. i.e. does agent intersect with area.

	# If yes, which side has the agent passed through?

	# Mirror agent back around to opposite wall.

	# split agent positions into x and y arrays
	agentsx = agents.T[0]
	agentsy = agents.T[1]

	scale = 1.2

	# Check left and right boundaries
	right = agentsx >=  scale*(map.dimensions[1]/2)
	left = agentsx <= -scale*(map.dimensions[1]/2)

	agentsx += -scale*(map.dimensions[1])*right
	agentsx += scale*(map.dimensions[1])*left

	# Check top and bottom boundaries
	top = agentsy >=  scale*(map.dimensions[0]/2)
	bottom = agentsy <= -scale*(map.dimensions[0]/2)

	agentsy += -scale*(map.dimensions[0])*top
	agentsy += scale*(map.dimensions[0])*bottom

	agents = np.stack((agentsx, agentsy), axis = 1)	

	return agents



def potentialField_map(env):

	'''
	Produce an avoidance map of the environment to reduce the need for
	computation every simulation loop.

	Produces an array of vectors describing the avoidance forces
	at all possible positions within the environment.

	Granularity has to be set to be small enough to avoid
	aliasing problems with agent positions.

	'''
	# Set granularity of field map
	granularity = 0.5

	x = np.arange(-75, 74.9, granularity)
	y = np.arange(-70, 69.9, granularity)
	positions = np.zeros((len(x)*len(y), 2))

	count = 0
	for k in y:
		for j in x:
			positions[count][0] = j 
			positions[count][1] = k
			count += 1

	size = len(positions)
	# Compute vectors between agents and wall planes
	
	diffh = np.array([env.planeh-positions[n][1] for n in range(size)])
	diffv = np.array([env.planev-positions[n][0] for n in range(size)])
	
	# split agent positions into x and y arrays
	agentsx = positions.T[0]
	agentsy = positions.T[1]

	# Check intersection of agents with walls
	low = agentsx[:, np.newaxis] >= env.limh.T[0]
	up = agentsx[:, np.newaxis] <= env.limh.T[1]
	intmat = up*low

	# For larger environments
	#A = 10; B = 10
	# For smaller environments
	A = 2; B = 5

	# Compute force based vector and multiply by intersection matrix
	Fy = np.exp(-A*np.abs(diffh) + B)*diffh*intmat
	
	low = agentsy[:, np.newaxis] >= env.limv.T[0]
	up = agentsy[:, np.newaxis] <= env.limv.T[1]
	intmat = up*low
	now = time.time()
	Fx = np.exp(-A*np.abs(diffv) + B)*diffv*intmat

	# Sum the forces between every wall into one force.
	Fx = np.sum(Fx, axis=1)
	Fy = np.sum(Fy, axis=1)
	# Combine x and y force vectors

	F = np.stack((Fx, Fy), axis = 1)

	return F, positions

def fieldmap_avoidance(swarm):

	'''
	For each agent look-up the avoidance vector for that
	agent based on it's current position.

	'''

	F = np.zeros((swarm.size, 2))

	x = np.arange(-75,74.8,0.5)
	y = np.arange(-70,69.9,0.5)

	# Take the potential field map
	field_map = swarm.field.reshape((len(y),len(x),2))

	# Round agent positions to increments of 0.5
	x = np.round(2*swarm.agents.T[0])/2
	y = np.round(2*swarm.agents.T[1])/2

	inx = np.round(2*(y+70))
	iny = np.round(2*(x+75))

	for n in range(swarm.size):
		# Take agent position and find position in grid
		F[n] = field_map[int(inx[n])][int(iny[n])]
	
	return F

def plot_field(swarm):

	fig = plt.figure()

	x = np.arange(-75,74.8,0.5)
	y = np.arange(-70,69.9,0.5)

	X,Y = np.meshgrid(x, y)

	f = np.zeros((len(x),len(y),2))

	print('shape of field ', np.shape(swarm.field))
	f = swarm.field.reshape((len(y),len(x),2))

	print('shape f mag ', np.shape(f))

	mag = np.linalg.norm(f, axis = 2)
	mag = (mag <= 300)*mag
	
	plt.imshow(mag)
	input()



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

def aggregate(swarm, param, noise):
	
	R = param; r = 3.5; A = 6.5; a = 7.5

	#noise = np.random.uniform(-.1, .1, (swarm.size, 2))

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	Avoid = fieldmap_avoidance(swarm)
	B = beacon(swarm)
	
	repel = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	repel = np.sum(repel, axis = 0).T

	attract = A*a*np.exp(-mag/a)[:,np.newaxis,:]*diff/(swarm.size-1)	
	attract = np.sum(attract, axis = 0).T

	total = 0
	total += Avoid + B + noise + repel - attract
	
	vecx = total.T[0]
	vecy = total.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += W 
	swarm.agents = continuous_boundary(swarm.agents, swarm.map)

def dispersion(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	#noise = np.random.uniform(-.1, .1, (swarm.size, 2))

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#A = avoidance(swarm.agents, swarm.map)
	#now = time.time()
	#A = fieldmap_avoidance(swarm)
	A = avoidance(swarm.agents, swarm.map)
	#taken = 1500*(time.time()-now)
	#print('\n\nAvoidance time is ', taken)

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
	swarm.agents = continuous_boundary(swarm.agents, swarm.map)
	
	


def rotate(swarm, direction, param):

	noise = param*np.random.randint(direction[0], direction[1], swarm.size)
	swarm.headings += noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	# Agent avoidance
	R = 2; r = 2; A = 1; a = 20
	# Compute euclidean distance between agents
	# mag = cdist(swarm.agents, swarm.agents)
	# # Compute vectors between agents
	# diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 
	# a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	# a = np.sum(a, axis =0).T

	a = np.zeros((swarm.size,2))
	B = np.zeros((swarm.size, 2))
	B = beacon(swarm)
	A = avoidance(swarm.agents, swarm.map)
	a += G + A + B

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	swarm.agents += W 
	swarm.agents = continuous_boundary(swarm.agents, swarm.map)

def random_walk(swarm, param):

	alpha = 0.01; beta = 50

	noise = param*np.random.randint(-beta, beta, (swarm.size))
	swarm.headings += noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	# Agent avoidance
	R = 20; r = 2; A = 1; a = 20	
	# Compute euclidean distance between agents
	# mag = cdist(swarm.agents, swarm.agents)
	# # Compute vectors between agents
	# diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 
	# a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	# a = np.sum(a, axis =0).T
	a = np.zeros((swarm.size, 2))

	B = np.zeros((swarm.size, 2))
	B = beacon(swarm)
	A = avoidance(swarm.agents, swarm.map)
	a += A + G + B

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += W

	swarm.agents = continuous_boundary(swarm.agents, swarm.map)


def bug_walk(swarm, param):

	alpha = 0.01; beta = 50

	noise = param*np.random.randint(-beta, beta+1, (swarm.size))
	swarm.headings += noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	# Agent avoidance
	R = 20; r = 2; A = 1; a = 20	
	# Compute euclidean distance between agents
	# mag = cdist(swarm.agents, swarm.agents)
	# # Compute vectors between agents
	# diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 
	# a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	# a = np.sum(a, axis =0).T
	a = np.zeros((swarm.size, 2))

	B = np.zeros((swarm.size, 2))
	B = beacon(swarm)
	A = avoidance(swarm.agents, swarm.map)
	a += A + G + B

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += W 

def flocking(swarm, param, noise):

	R = 30; r = 3.5; A = 10.5; a = 5.5

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Determine headings
	nearest = mag <= 2

	# n x n matrix of headings of agents which are adjacent
	neighbour_headings = swarm.headings*nearest

	# Sum headings for each agent
	neighbour_headings_tot = np.sum(neighbour_headings, axis = 1)

	# average by number of neighbours

	new_headings = neighbour_headings_tot/(np.sum(nearest, axis = 1))

	# average headings with neighbours
	swarm.headings =  (new_headings + 0.01*np.random.randint(-10,11, swarm.size))

	# Calculate new heading vector
	strength = 10
	gx = strength*np.cos(swarm.headings)
	gy = strength*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])
	

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#Avoid = fieldmap_avoidance(swarm)
	
	repel = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	repel = np.sum(repel, axis = 0).T

	attract = A*a*np.exp(-mag/a)[:,np.newaxis,:]*diff/(swarm.size-1)	
	attract = np.sum(attract, axis = 0).T

	total = 0
	total +=  noise + repel + G - attract
	
	vecx = total.T[0]
	vecy = total.T[1]
	angles = np.arctan2(vecy, vecx)

	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += W 
	swarm.agents = continuous_boundary(swarm.agents, swarm.map)
	

	
	















