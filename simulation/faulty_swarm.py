
import numpy as np
import simulation.environment as environment
import simulation.asim as asim
import sys
import time

from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, LineString, LinearRing
from shapely.ops import split, triangulate
from shapely import affinity
import math

from scipy.spatial.distance import cdist, pdist, euclidean



class swarm(object):

	def __init__(self):

		self.agents = []
		self.speed = 0.2
		self.size = 0
		self.behaviour = []

		self.field = []
		self.grid = []

		self.param = 3
		self.map = 'none'

		self.origin = np.array([0,0])
		self.start = np.array([])

		self.died = 0
		self.shadows = []

		self.time = 0

		self.period = []

		self.info_gain = None
		self.agent_mem_length = 20
		self.agent_mem = np.zeros((self.agent_mem_length, self.size))

		self.direct_vectors = None

		### social algorithm params ###

		self.happiness = None # Binary list indicating un-happy or happy
		self.prev_happiness = 0
		self.happiness_threshold = 0
		self.update_rate = 10 # Opinion sharing frequency 
		self.previous_state = None # Agents remember previous position when last sharing opinions
		self.longterm_state = None
		self.opinion_timer = None # Agents share opinions at different times
		self.opinion_timelimit = None
		self.comm_range = 5

		self.last_behaviour = None

		self.long_happiness = None
		self.longterm_counter = None
		self.longterm_timelimit = None

		self.happiness_noise = None
		self.comm_prob = 1.0

		# variables for collision based happiness
		self.collision_count = None
		self.collision_threshold = .5

		# variables for coverage based happiness

		self.objective_count = None
		self.agent_objective_states = None

		# Sensor distance errors
		self.sensor_fault = None
		self.sensor_mean = 0
		self.sensor_dev = 0

		# Motor speed errors
		self.motor_speeds = None
		self.motor_error = None
		self.motor_mean = 0
		self.motor_dev = 0

		# Motor heading error
		self.heading_error = None
		self.heading_mean = 0
		self.heading_dev = 0

		# Malicious robots 
		self.malicious_stopped = None

		self.faulty_robot = None
		self.malicious_robot = None

		self.malicious_broadcasters = None

		# Collision states for agents and walls --------
		self.collision_state = None
		self.collision_timeout = None

		self.wallCollision_state = None
		self.wallCollision_timeout = None


	def gen_agents(self):

		dim = 0.001
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		for n in range(self.size):
			self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2) + self.origin[0], 0 + self.origin[1]])

		#self.behaviour = np.zeros(self.size)

		self.behaviour = np.random.randint(0,9, self.size)
		self.shadows = np.zeros((4,self.size,2))
		self.info_gain = 4*np.random.randint(0, 2, self.size)

		### Anti-consensus ###


		self.happiness = np.zeros(self.size)
		self.prev_happiness = np.zeros(self.size)
		self.long_happiness = np.zeros(self.size)
		self.happiness_threshold = 0
		self.previous_state = np.zeros((self.size, 2))
		# Agents check opinions at slightly varying rates for asynchronous updates
		self.opinion_timelimit = np.random.randint(1*self.update_rate-2, 1*self.update_rate+2, self.size)
		self.opinion_timer = np.zeros(self.size)
		self.previous_state = np.zeros((self.size, 2))
		# np.copyto(self.previous_state, self.agents)
		self.longterm_state = self.agents
		self.longterm_counter = np.zeros(self.size)
		self.longterm_timelimit = np.random.randint(3*self.update_rate-2, 3*self.update_rate+2, self.size)

		self.happiness_noise = np.zeros(self.size)
		# self.happiness_noise = np.random.normal(-0.4, 0.1, self.size)

		self.collision_count = np.zeros(self.size)

		self.objective_count = np.zeros(self.size)
		self.agent_objective_states = [None for x in range(self.size)]
		# mag = 0.4
		# self.happiness_noise = np.random.uniform(-mag, mag, self.size)
		self.sensor_fault = np.zeros(self.size)
		self.doorway_occupied = np.zeros(6)

		self.direct_vectors = np.zeros((self.size,2))

		# =============== Paramters for collision detection ================
		self.collision_state = np.zeros(self.size)
		self.collision_timeout = np.zeros(self.size)

		self.wallCollision_state = np.zeros(self.size)
		self.wallCollision_timeout = np.zeros(self.size)

	def gen_agents_uniform(self, env):

		dim = 0.001
		self.dead = np.zeros(self.size)
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		#self.behaviour = np.zeros(self.size)
		self.behaviour = np.zeros(self.size)
		
		x = np.random.uniform(-env.dimensions[1]/2, env.dimensions[1]/2, self.size)
		y = np.random.uniform(-env.dimensions[0]/2, env.dimensions[0]/2, self.size)

		self.agents = np.stack((x,y), axis = 1)
		self.shadows = np.zeros((4,self.size,2))
		self.period = np.zeros(self.size)
		

	def reset(self):

		dim = 0.001
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		for n in range(self.size):
			self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2),0])

	def iterate(self, faulty_swarm, noise):

		# Check for collisions

		# force = 60
		directed_fields(self, faulty_swarm, self.param, noise)


	def get_state_opinion(self):

		# Function to update opinions and compare with neighbours

		totx = 0; toty = 0; totmag = 0
		# Calculate connectivity matrix between agents
		mag = cdist(self.agents, self.agents)
		totmag = np.sum(mag)

		# calculate density and center of mass of the swarm
		self.spread = totmag/((self.size -1)*self.size)
		self.median = np.median(self.agents, axis = 0)
		
		self.shadows[3] = self.shadows[2]
		self.shadows[2] = self.shadows[1]
		self.shadows[1] = self.shadows[0]
		self.shadows[0] = self.agents

		# Matrix of agent neighbourhoods
		prox = mag <= self.comm_range

		# step down timer
		self.opinion_timer = self.opinion_timer + np.ones(self.size)
		# Check update status of agents

		for n in range(0, self.size):
			# Loop through all agents and update their decision making states

			# if self.behaviour[n] != 10:
			# update agent happiness each timestep over 5 steps

			if self.malicious_broadcasters[n] == 1:
				# -------------------------- Added broadcaster malicious behaviours ---------------------------------
				self.happiness[n] = 1

				broadcast_range = 20
				broadcast_prox = mag <= broadcast_range

				inrange = np.where(broadcast_prox[n] == True)[0]
				

				# For each agent in range, command to switch to bad behaviour
				for i in range(len(inrange)):
					index = inrange[i]

					if (self.happiness[n] - self.happiness[index]) >= 0.4 and np.random.uniform(0,1) <= 0.9:

						# if neighbour has low empowerment, they should adopt agents opinion
						self.behaviour[index] = self.behaviour[n]
						# Agent resets timer as they switch to new behaviour
						self.opinion_timer[index] = 0

			else:

				if self.opinion_timer[n] >= 5:
				# have far have they moved from previous position?
					self.happiness[n] = cdist([self.previous_state[n]], [self.agents[n]])[0][0]/self.opinion_timer[n]

					max_val = self.speed

					inrange = np.where(prox[n] == True)[0]

					neighbour_num = len(inrange)
					max_neighbour = 10
					if neighbour_num >= max_neighbour:
						neighbour_num = 9

					# Normalize between 0 to 1
					self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
					#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2
					# Add noise to happiness measures

					self.happiness[n] += self.happiness_noise[n]


				#High empowerment individuals continue to broadcast opinion

				if self.happiness[n] - self.prev_happiness[n] >= 0.2 or self.happiness[n] >= 0.5:

					inrange = np.where(prox[n] == True)[0]

					for k in range(len(inrange)):

							# Loop through neighbours and broadcast opinion

							# *** Ignore agents with the same opinion 
							if np.random.uniform(0,1) <= self.happiness[n] and np.random.uniform(0,1) <= self.comm_prob:

								if (self.happiness[n] - self.happiness[inrange[k]]) >= 0.4 and self.behaviour[n] != self.behaviour[inrange[k]]:

									# if neighbour has low empowerment, they should adopt agents opinion

									self.behaviour[inrange[k]] = self.behaviour[n]

									# Trigger chain event indicating an increae in empowerment by taking on broadcast behaviour

									# *** Averaging the value will decay the chain of broadcast
									# self.happiness[inrange[k]] = (self.happiness[inrange[k]] + self.happiness[n]) / 2

									# self.opinion_timer[inrange[k]] = self.opinion_timelimit[inrange[k]]

				#Check whether it's time to update opinion

				if self.opinion_timer[n] >= self.opinion_timelimit[n]:
					# Update opinion

					self.opinion_timer[n] = 0
					opinion_buffer = 1.1 # percent difference

					# Compare opinion with neighbours

					inrange = np.where(prox[n] == True)[0]

					max_opinion = 0
					index = -1
					for i in range(len(inrange)):
						if self.happiness[inrange[i]] > max_opinion and np.random.uniform(0,1) <= self.comm_prob:

							index = inrange[i]
							max_opinion = self.happiness[inrange[i]]


					############# Update measure based on distance and time ##############

					# have far have they moved from previous position?
					self.happiness[n] = cdist([self.previous_state[n]], [self.agents[n]])[0][0]/self.opinion_timelimit[n]

					max_val = self.speed

					neighbour_num = len(inrange)
					max_neighbour = 10
					if neighbour_num >= max_neighbour:
						neighbour_num = 9

					# Normalize between 0 to 1
					self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
					#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2

					self.happiness[n] += self.happiness_noise[n]

					

					# first check own performance

					if self.happiness[n] - self.prev_happiness[n] <= 0.2 and self.happiness[n] <= 0.5:

						#Something needs to change!
						self.prev_happiness[n] = self.happiness[n]

						# Compare best opinion with own opinion

						# Only change if they're opinion is different to yours

						if index >= 0 and max_opinion >= opinion_buffer*self.happiness[n] and self.behaviour[n] != self.behaviour[index]:

							# **** SWITCH TO BEHAVIOUR OF NEIGHBOUR WITH HIGHER PERFORMANCE

							self.behaviour[n] = self.behaviour[index]
						else:

							# If no one else is performing better than you, try random behaviour
							options = [1,2,3,4,5,6,7,8]

							#options = [0,10,11,9]

							# options.remove(self.behaviour[index])
							
							# self.behaviour[n] = np.random.choice(options)

							# Pick a behaviour that no one has tried

							options.remove(self.behaviour[n])

							for k in range(len(inrange)):

								if self.behaviour[inrange[k]] in options and np.random.uniform(0,1) <= self.comm_prob:
									options.remove(self.behaviour[inrange[k]])

							# Make random choice from remaining behaviours
							if len(options) != 0:
								self.behaviour[n] = np.random.choice(options)
							else:
								# Reset options if all neighbours are doing different behaviours
								options = [1,2,3,4,5,6,7,8]
								#options = [0,10,11,9]
								self.behaviour[n] = np.random.choice(options)

						# All neighbours take on chosen behaviour
						for k in range(len(inrange)):
							if np.random.uniform(0,1) <= 0.5 and np.random.uniform(0,1) <= self.comm_prob:
								self.behaviour[inrange[k]] = self.behaviour[n]
								self.opinion_timer[inrange[k]] = 0

					self.collision_count[n] = 0

					# self.previous_state[n] = self.agents[n][:]
					np.copyto(self.previous_state[n], self.agents[n])

					# Reset agent timer
					# *** If happiness is low check opinion again more quickly
					self.opinion_timelimit[n] = (1+self.happiness[n])*np.random.randint(self.update_rate-2 , self.update_rate+2)


		# Increment longterm timer
		self.longterm_counter = self.longterm_counter + np.ones(self.size)

		#np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
		for n in range(self.size):

			if self.behaviour[n] == 1:
				self.direct_vectors[n] = np.array([0,1])
			if self.behaviour[n] == 2:
				self.direct_vectors[n] = np.array([0,-1])
			if self.behaviour[n] == 3:
				self.direct_vectors[n] = np.array([1,0])
			if self.behaviour[n] == 4:
				self.direct_vectors[n] = np.array([-1,0])
			if self.behaviour[n] == 5:
				self.direct_vectors[n] = np.array([-1,1])
			if self.behaviour[n] == 6:
				self.direct_vectors[n] = np.array([1,1])
			if self.behaviour[n] == 7:
				self.direct_vectors[n] = np.array([1,-1])
			if self.behaviour[n] == 8:
				self.direct_vectors[n] = np.array([-1,-1])
	
	def malicious_broacast_update(self):

		broadcast_range = 20
		agents = np.where(self.malicious_broadcasters == 1)[0]



		mag = cdist(self.agents, self.agents)
		
		# Matrix of agent neighbourhoods
		prox = mag <= broadcast_range

		for agent in agents:

			# always set agnets happiness to 1

			# Find which agents are in range
			inrange_agents = np.where(prox[n] == True)[0]

			for i in range(len(inrange)):
				if self.happiness[inrange[i]] > max_opinion and np.random.uniform(0,1) <= self.comm_prob:

					index = inrange[i]
					max_opinion = self.happiness[inrange[i]]


	def get_state(self):

		totx = 0; toty = 0; totmag = 0
		# Calculate connectivity matrix between agents
		mag = cdist(self.agents, self.agents)

		totmag = np.sum(mag)
		#totpos = np.sum(self.agents, axis=0)

		# calculate density and center of mass of the swarm
		self.spread = totmag/((self.size -1)*self.size)
		# self.centermass[0] = (totpos[0])/(self.size)
		# self.centermass[1] = (totpos[1])/(self.size)
		self.median = np.median(self.agents, axis = 0)
		# self.upper = np.quantile(self.agents, 0.75, axis = 0)
		# self.lower = np.quantile(self.agents, 0.25, axis = 0)
		self.shadows[3] = self.shadows[2]
		self.shadows[2] = self.shadows[1]
		self.shadows[1] = self.shadows[0]
		self.shadows[0] = self.agents

		# Update directed field vectors


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

class malicious_swarm(object):

	def __init__(self):

		self.agents = []
		self.speed = 0.2
		self.size = 0
		self.behaviour = []

		self.centroids = []
		self.centermass = [0,0]
		self.median = [0,0]
		self.spread = 0

		self.field = []
		self.grid = []

		self.param = 3
		self.map = 'none'
		self.beacon_att = np.array([[]])
		self.beacon_rep = np.array([[]])

		self.origin = np.array([0,0])
		self.start = np.array([])

		self.died = 0
		self.shadows = []
		self.funcdict = {"random": asim.random_walk, "rot_clock": asim.rotate}


		self.beh_beacons = None

		self.time = 0

		self.period = []

		self.info_gain = None
		self.agent_mem_length = 20
		self.agent_mem = np.zeros((self.agent_mem_length, self.size))

		self.direct_vectors = None

		### social algorithm params ###

		self.happiness = None # Binary list indicating un-happy or happy
		self.prev_happiness = 0
		self.happiness_threshold = 0
		self.update_rate = 10 # Opinion sharing frequency 
		self.previous_state = None # Agents remember previous position when last sharing opinions
		self.longterm_state = None
		self.opinion_timer = None # Agents share opinions at different times
		self.opinion_timelimit = None
		self.comm_range = 5

		self.last_behaviour = None

		self.long_happiness = None
		self.longterm_counter = None
		self.longterm_timelimit = None

		self.happiness_noise = None
		self.comm_prob = 1.0

		# variables for collision based happiness
		self.collision_count = None
		self.collision_threshold = .5

		# variables for coverage based happiness

		self.objective_count = None
		self.agent_objective_states = None

		# Sensor distance errors
		self.sensor_fault = None
		self.sensor_mean = 0
		self.sensor_dev = 0

		# Motor speed errors
		self.motor_speeds = None
		self.motor_error = None
		self.motor_mean = 0
		self.motor_dev = 0

		# Motor heading error
		self.heading_error = None
		self.heading_mean = 0
		self.heading_dev = 0

		self.malicious_stopped = None

		self.faulty_robot = None
		self.malicious_robot = None

	def gen_agents(self):

		dim = 0.001
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		for n in range(self.size):
			self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2) + self.origin[0], 0 + self.origin[1]])

		#self.behaviour = np.zeros(self.size)

		self.behaviour = np.random.randint(0,9, self.size)
		self.shadows = np.zeros((4,self.size,2))
		self.info_gain = 4*np.random.randint(0, 2, self.size)

		### Anti-consensus ###


		self.happiness = np.zeros(self.size)
		self.prev_happiness = np.zeros(self.size)
		self.long_happiness = np.zeros(self.size)
		self.happiness_threshold = 0
		self.previous_state = np.zeros((self.size, 2))
		# Agents check opinions at slightly varying rates for asynchronous updates
		self.opinion_timelimit = np.random.randint(1*self.update_rate-2, 1*self.update_rate+2, self.size)
		self.opinion_timer = np.zeros(self.size)
		self.previous_state = np.zeros((self.size, 2))
		# np.copyto(self.previous_state, self.agents)
		self.longterm_state = self.agents
		self.longterm_counter = np.zeros(self.size)
		self.longterm_timelimit = np.random.randint(3*self.update_rate-2, 3*self.update_rate+2, self.size)

		self.happiness_noise = np.zeros(self.size)
		# self.happiness_noise = np.random.normal(-0.4, 0.1, self.size)

		self.collision_count = np.zeros(self.size)

		self.objective_count = np.zeros(self.size)
		self.agent_objective_states = [None for x in range(self.size)]
		# mag = 0.4
		# self.happiness_noise = np.random.uniform(-mag, mag, self.size)
		self.sensor_fault = np.zeros(self.size)
		self.doorway_occupied = np.zeros(6)

		self.direct_vectors = np.zeros((self.size,2))

	def gen_agents_uniform(self, env):

		dim = 0.001
		self.dead = np.zeros(self.size)
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		#self.behaviour = np.zeros(self.size)
		self.behaviour = np.zeros(self.size)
		
		x = np.random.uniform(-env.dimensions[1]/2, env.dimensions[1]/2, self.size)
		y = np.random.uniform(-env.dimensions[0]/2, env.dimensions[0]/2, self.size)

		self.agents = np.stack((x,y), axis = 1)
		self.shadows = np.zeros((4,self.size,2))
		self.period = np.zeros(self.size)
		

	def reset(self):

		dim = 0.001
		self.agents = np.zeros((self.size,2))
		self.headings = 0.0314*np.random.randint(-100,100 ,self.size)
		for n in range(self.size):
			self.agents[n] = np.array([dim*n - (dim*(self.size-1)/2),0])

	def iterate(self, noise):

		# Check for collisions

		force = 60
		malicious_dispersion(self, np.array([0,0]), force, noise)


	def get_state_opinion(self):

		# Function to update opinions and compare with neighbours

		totx = 0; toty = 0; totmag = 0
		# Calculate connectivity matrix between agents
		mag = cdist(self.agents, self.agents)
		totmag = np.sum(mag)
		#totpos = np.sum(self.agents, axis=0)

		# calculate density and center of mass of the swarm
		self.spread = totmag/((self.size -1)*self.size)
		# self.centermass[0] = (totpos[0])/(self.size)
		# self.centermass[1] = (totpos[1])/(self.size)
		self.median = np.median(self.agents, axis = 0)
		# self.upper = np.quantile(self.agents, 0.75, axis = 0)
		# self.lower = np.quantile(self.agents, 0.25, axis = 0)
		self.shadows[3] = self.shadows[2]
		self.shadows[2] = self.shadows[1]
		self.shadows[1] = self.shadows[0]
		self.shadows[0] = self.agents

		# Matrix of agent neighbourhoods
		prox = mag <= self.comm_range


		# step down timer

		self.opinion_timer = self.opinion_timer + np.ones(self.size)
		# Check update status of agents

		for n in range(0, self.size):
			# Loop through all agents and update their decision making states

			# if self.behaviour[n] != 10:
			# update agent happiness each timestep over 5 steps

			if self.opinion_timer[n] >= 5:
			# have far have they moved from previous position?
				self.happiness[n] = cdist([self.previous_state[n]], [self.agents[n]])[0][0]/self.opinion_timer[n]

				# print('Swarm previous state: ', self.previous_state[n])
				# print('Swarm current state: ', self.agents[n])
			

				max_val = self.speed

				inrange = np.where(prox[n] == True)[0]

				neighbour_num = len(inrange)
				max_neighbour = 10
				if neighbour_num >= max_neighbour:
					neighbour_num = 9

				# Normalize between 0 to 1
				self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
				#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2
				# Add noise to happiness measures

				self.happiness[n] += self.happiness_noise[n]

				# print('Swarm happiness for agent %d is %.2f' % (n, self.happiness[n]))
				# input()

			#High empowerment individuals continue to broadcast opinion

			if self.happiness[n] - self.prev_happiness[n] >= 0.2 or self.happiness[n] >= 0.5:

				inrange = np.where(prox[n] == True)[0]

				for k in range(len(inrange)):

						# Loop through neighbours and broadcast opinion

						# *** Ignore agents with the same opinion 
						if np.random.uniform(0,1) <= self.happiness[n] and np.random.uniform(0,1) <= self.comm_prob:

							if (self.happiness[n] - self.happiness[inrange[k]]) >= 0.4 and self.behaviour[n] != self.behaviour[inrange[k]]:

								# if neighbour has low empowerment, they should adopt agents opinion

								self.behaviour[inrange[k]] = self.behaviour[n]

								# Trigger chain event indicating an increae in empowerment by taking on broadcast behaviour

								# *** Averaging the value will decay the chain of broadcast
								# self.happiness[inrange[k]] = (self.happiness[inrange[k]] + self.happiness[n]) / 2

								# self.opinion_timer[inrange[k]] = self.opinion_timelimit[inrange[k]]

			#Check whether it's time to update opinion

			if self.opinion_timer[n] >= self.opinion_timelimit[n]:
				# Update opinion

				self.opinion_timer[n] = 0
			

				opinion_buffer = 1.1 # percent difference

				# Compare opinion with neighbours

				inrange = np.where(prox[n] == True)[0]

				max_opinion = 0
				index = -1
				for i in range(len(inrange)):
					if self.happiness[inrange[i]] > max_opinion and np.random.uniform(0,1) <= self.comm_prob:

						index = inrange[i]
						max_opinion = self.happiness[inrange[i]]


				############# Update measure based on distance and time ##############

				# have far have they moved from previous position?
				self.happiness[n] = cdist([self.previous_state[n]], [self.agents[n]])[0][0]/self.opinion_timelimit[n]

				max_val = self.speed

				neighbour_num = len(inrange)
				max_neighbour = 10
				if neighbour_num >= max_neighbour:
					neighbour_num = 9

				# Normalize between 0 to 1
				self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
				#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2

				self.happiness[n] += self.happiness_noise[n]

				

				# first check own performance

				if self.happiness[n] - self.prev_happiness[n] <= 0.2 and self.happiness[n] <= 0.5:

					#Something needs to change!

					self.prev_happiness[n] = self.happiness[n]

					# Compare best opinion with own opinion

					# Only change if they're opinion is different to yours

					if index >= 0 and max_opinion >= opinion_buffer*self.happiness[n] and self.behaviour[n] != self.behaviour[index]:

						# **** SWITCH TO BEHAVIOUR OF NEIGHBOUR WITH HIGHER PERFORMANCE

						self.behaviour[n] = self.behaviour[index]
					else:

						# If no one else is performing better than you, try random behaviour


						options = [1,2,3,4,5,6,7,8]

						#options = [0,10,11,9]

						# options.remove(self.behaviour[index])
						
						# self.behaviour[n] = np.random.choice(options)

						# Pick a behaviour that no one has tried

						options.remove(self.behaviour[n])

						for k in range(len(inrange)):

							if self.behaviour[inrange[k]] in options and np.random.uniform(0,1) <= self.comm_prob:
								options.remove(self.behaviour[inrange[k]])

						# Make random choice from remaining behaviours
						if len(options) != 0:
							self.behaviour[n] = np.random.choice(options)
						else:
							# Reset options if all neighbours are doing different behaviours
							options = [1,2,3,4,5,6,7,8]
							#options = [0,10,11,9]
							self.behaviour[n] = np.random.choice(options)

					# All neighbours take on chosen behaviour
					for k in range(len(inrange)):
						if np.random.uniform(0,1) <= 0.5 and np.random.uniform(0,1) <= self.comm_prob:
							self.behaviour[inrange[k]] = self.behaviour[n]
							self.opinion_timer[inrange[k]] = 0

				self.collision_count[n] = 0

				# self.previous_state[n] = self.agents[n][:]
				np.copyto(self.previous_state[n], self.agents[n])

				# Reset agent timer
				# *** If happiness is low check opinion again more quickly
				self.opinion_timelimit[n] = (1+self.happiness[n])*np.random.randint(self.update_rate-2 , self.update_rate+2)

		
			

					# Increment longterm timer
		self.longterm_counter = self.longterm_counter + np.ones(self.size)

		#np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
		for n in range(self.size):

			if self.behaviour[n] == 1:
				self.direct_vectors[n] = np.array([0,1])
			if self.behaviour[n] == 2:
				self.direct_vectors[n] = np.array([0,-1])
			if self.behaviour[n] == 3:
				self.direct_vectors[n] = np.array([1,0])
			if self.behaviour[n] == 4:
				self.direct_vectors[n] = np.array([-1,0])
			if self.behaviour[n] == 5:
				self.direct_vectors[n] = np.array([-1,1])
			if self.behaviour[n] == 6:
				self.direct_vectors[n] = np.array([1,1])
			if self.behaviour[n] == 7:
				self.direct_vectors[n] = np.array([1,-1])
			if self.behaviour[n] == 8:
				self.direct_vectors[n] = np.array([-1,-1])
	


	def get_state(self):

		totx = 0; toty = 0; totmag = 0
		# Calculate connectivity matrix between agents
		mag = cdist(self.agents, self.agents)



		totmag = np.sum(mag)
		#totpos = np.sum(self.agents, axis=0)

		# calculate density and center of mass of the swarm
		self.spread = totmag/((self.size -1)*self.size)
		# self.centermass[0] = (totpos[0])/(self.size)
		# self.centermass[1] = (totpos[1])/(self.size)
		self.median = np.median(self.agents, axis = 0)
		# self.upper = np.quantile(self.agents, 0.75, axis = 0)
		# self.lower = np.quantile(self.agents, 0.25, axis = 0)
		self.shadows[3] = self.shadows[2]
		self.shadows[2] = self.shadows[1]
		self.shadows[1] = self.shadows[0]
		self.shadows[0] = self.agents

		# Update directed field vectors


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

def collision_detection(swarm):

	# Check avoidance forces for agents
	Avoid = asim.avoidance(swarm.agents, swarm.map)

	# If force is higher than x, register collision

	# Forces greater than threshold

	force_threshold = swarm.collision_threshold

	# Compute magnitude of each force vector

	force_mag = np.linalg.norm(Avoid, axis = 1)

	# Threshold forces for collisions
	collisions = force_mag >= force_threshold



	# input()

	# Add collision detections to counter
	swarm.collision_count += collisions


	# input()

def objective_detection(swarm, targets):



	score = 0
	# adjacency matrix of agents and targets
	mag = cdist(swarm.agents, targets.targets)

	# Check which distances are less than detection range
	a = mag < targets.radius

	#print(a)
	# Sum over agent axis 
	detected = np.sum(a, axis = 0)
	# convert to boolean, targets with 0 detections set to false.
	detected = detected > 0
	# Check detection against previous state. If a target is already found return false.
	#updated = np.logical_or(detected, targets.old_state) 


	

	# Loop through agents and check 


	for n in range(swarm.size):

		updated = np.logical_or(a[n], swarm.agent_objective_states[n]) 

		# For each agent check their detected targets a[n] against old state
		found = np.logical_xor(a[n], swarm.agent_objective_states[n])*a[n]

		if np.sum(found) > 0:

			swarm.objective_count[n] += 1
	

		swarm.agent_objective_states[n] = updated[:]

def robot_faults(swarm, noise):


	for n in range(swarm.size):

		if swarm.sensor_fault[n] == 1:

			mag[n] = mag[n] + np.random.normal(swarm.sensor_mean, swarm.sensor_dev, (swarm.size))
			# Values less than 0 set to zero
			mag[n] = (mag[n] > 0)*mag[n]

	slowAgents = np.where(swarm.motor_error == 1)[0]

	for agent in slowAgents:

		swarm.motor_speeds[agent] = np.random.normal(swarm.motor_mean, swarm.motor_dev)


def malicious_dispersion(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	# Define center points of doorways where agents will stop.
	doorways = np.array([[0,10],[0,20],[-25,-10],[25,20],[-10,15],[25,10]])
	detection_range = 5
	stop_range = 1
	states = swarm.behaviour == 10

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)


	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	Avoid = asim.avoidance(swarm.agents, swarm.map)
		
	rep = 1*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	rep = np.sum(rep, axis = 0).T

	heading_noise = np.random.uniform(-.5, .5, (swarm.size))

	# Move only agents with random behaviour set
	# states = swarm.behaviour == 0

	swarm.headings += states*heading_noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])


	rep += Avoid - vector + noise + G

	# Add behaviour to attract agents to doorways.

	# Check distance to doorways
	door_mag = cdist(swarm.agents, doorways)

	# Compute vectors between doorways
	door_diff = swarm.agents[:,:,np.newaxis]-doorways.T[np.newaxis,:,:] 
	# print(door_mag)
	# door_att = -R*r*np.exp(-mag/r)[:,np.newaxis,:]*door_diff/(swarm.size-1)	
	# door_att = np.sum(door_att, axis = 0).T

	door_att = 10*((door_mag)/detection_range)[:,np.newaxis,:]*door_diff/swarm.size
	# door_att = door_att*(door_mag < detection_range)[:,np.newaxis,:]	
	# door_att = np.sum(door_att, axis = 2)
	# print('Doorway attraction forces: ', door_att)
	malicious_agentID = np.where(swarm.behaviour == 10)[0]
	# print(malicious_agentID)

	# print(np.shape(door_att))
	stopped_agent = np.where(swarm.behaviour == 99)[0]

	for agent in stopped_agent:
		# Add bumbling behaviour once stopped
		
		doorway_ID = np.where(door_mag[agent] <= detection_range)[0][0]
		# print(doorway_ID)	
		# Check whether agent has moved out of stopping distance
		# print('distance to door = ', door_mag[agent][doorway_ID])
		if door_mag[agent][doorway_ID] > stop_range:
			# input()
			# rep[agent][0] += .1*door_att[agent][0][doorway_ID]
			# rep[agent][1] += door_att[agent][1][doorway_ID]

			stopped_states = swarm.behaviour == 99
			vecx = door_att[agent][0][doorway_ID]
			vecy = door_att[agent][1][doorway_ID]
			angles = np.arctan2(vecy, vecx)
			Wx = .5*swarm.speed*np.cos(angles)
			Wy = .5*swarm.speed*np.sin(angles)

			#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
			W = -np.stack((Wx, Wy))
			# states = states*(stop > 0.7)
			swarm.agents[agent] += W

		else:
			swarm.agents[agent] += np.random.uniform(-.2, .2, (2))



	# Cycle through randomly chosen malicious agents
	for agent in malicious_agentID:
		# Check if agent is intersecting with doorway

		if np.sum((door_mag[agent] < detection_range)) >= 1:

			# print('doorway mag: ', door_mag)
			doorway_ID = np.where(door_mag[agent] <= detection_range)[0][0]

			if swarm.doorway_occupied[doorway_ID] == 0:
				# print('Doorway id = ', doorway_ID)
				# Check if agent is within stopping distance 
				if np.sum((door_mag[agent] < stop_range)) >= 1 and np.random.uniform(0,1,) <= 0.7:
					# Check whether doorway is already occuied
					if swarm.doorway_occupied[doorway_ID] == 0:
						# print('Agent %d is stopping' % (agent))
						swarm.behaviour[agent] = 99
						# Set that doorway is occupied and other agents cannot stop here.
						swarm.doorway_occupied[doorway_ID] = 1

				# Move towards stopping point
				else:

					rep[agent][0] += door_att[agent][0][doorway_ID]
					rep[agent][1] += door_att[agent][1][doorway_ID]
	

	
	vecx = rep.T[0]
	vecy = rep.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	# states = states*(stop > 0.7)
	swarm.agents += states[:,np.newaxis]*W

def directed_fields(swarm, faulty_swarm, param, noise):

	# Combined odirected field behaviour with each using an assigned heading vector

	R = param; r = 2; A = 1; a = 20

	# states = swarm.behaviour == 1
	# print('Swarm direction vectors: ', swarm.direct_vectors)

	total_agents = np.concatenate((swarm.agents, faulty_swarm.agents), axis = 0)

	# Compute euclidean distance between agents

	# ------------ Add some error on to agent to agent distance measurement -------------
	mag = cdist(swarm.agents, swarm.agents)
	for n in range(swarm.size):

		if swarm.sensor_fault[n] == 1:

			mag[n] = mag[n] + np.random.normal(swarm.sensor_mean, swarm.sensor_dev, (swarm.size))
			# Values less than 0 set to zero
			mag[n] = (mag[n] > 0)*mag[n]

	# ----------- Agents with motor faults move slower ------------
	slowAgents = np.where(swarm.motor_error == 1)[0]

	for agent in slowAgents:

		swarm.motor_speeds[agent] = np.random.normal(swarm.motor_mean, swarm.motor_dev)

	# -------------------------------------------------------------

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	Avoid = avoidance(swarm, swarm.map)
	repel = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	# print('repel shape: ', repel)
	repel = np.sum(repel, axis = 0).T

	repel += Avoid - swarm.direct_vectors + noise

	# ---------------------- Add avoidance from malicious agents -----------------------

	faulty_mag = cdist(swarm.agents, faulty_swarm.agents)
	diff = swarm.agents[:,:,np.newaxis] - faulty_swarm.agents.T[np.newaxis,:,:] 

	faulty_repel = R*r*np.exp(-faulty_mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	faulty_repel = np.sum(faulty_repel, axis = 2)

	repel -= faulty_repel
	
	vecx = repel.T[0]
	vecy = repel.T[1]
	angles = np.arctan2(vecy, vecx)
	

	# ------ add heading error to agent movement --------
	heading_error = np.where(swarm.heading_error == 1)[0]
	for agent in heading_error:

		angles[agent] += np.random.normal(swarm.heading_mean, swarm.heading_dev)


	Wx = swarm.speed*swarm.motor_speeds*np.cos(angles)
	Wy = swarm.speed*swarm.motor_speeds*np.sin(angles)

	W = -np.stack((Wx, Wy), axis = 1)
	# Agents which have not collided update their position normally
	swarm.agents += np.logical_not(swarm.collision_state)[:,np.newaxis]*np.logical_not(swarm.wallCollision_state)[:,np.newaxis]*W



def collision_check(swarm, faulty_swarm):

	# Check for collisions between agents

	# Agents which collide remain static for period X before the collision is resolved and agents continue exploration

	collision_radius = 1.2

	# Time buffer to allow agents to correct collision
	collision_delayBuffer = 10

	# Time which collided agents remain inactive
	collision_duration = 30

	adjacency_matrix = cdist(swarm.agents, swarm.agents)

	collided_agents = adjacency_matrix <= collision_radius

	# turn nxn matrix into flat list
	collided_agents = np.sum(collided_agents, axis = 0)
	# print('collisions: ', collided_agents)
	# input()

	# collided agents includes detection with self. therefore needs to be more than 1 for collision
	swarm.collision_state = collided_agents >= 2


	for n in range(swarm.size):

		if swarm.collision_state[n] == 1:
			swarm.collision_timeout[n] += 1

		if swarm.collision_timeout[n] > collision_duration:
			# Agent is free to move again
			swarm.collision_state[n] = 0
		
		if swarm.collision_timeout[n] > collision_duration + collision_delayBuffer:
			swarm.collision_timeout[n] = 0


	# return collision_pos


def avoidance(swarm, map):

	size = len(swarm.agents)
	# Compute vectors between swarm.agents and wall planes
	
	diffh = np.array([map.planeh-swarm.agents[n][1] for n in range(size)])
	diffv = np.array([map.planev-swarm.agents[n][0] for n in range(size)])

	sensor_error = np.random.normal(1, 0.2, swarm.size)

	sensor_error = swarm.sensor_fault*sensor_error

	collision_dist = .8

	detected_collisionsH = abs(diffh) <= collision_dist
	detected_collisionsV = abs(diffv) <= collision_dist

	# diffh += sensor_error[:,np.newaxis]
	# diffv += sensor_error[:,np.newaxis]
	
	# split agent positions into x and y arrays
	agentsx = swarm.agents.T[0]
	agentsy = swarm.agents.T[1]

	# Check intersection of agents with walls
	low = agentsx[:, np.newaxis] >= map.limh.T[0]
	up = agentsx[:, np.newaxis] <= map.limh.T[1]
	intmat = up*low

	# *** collisions are only triggered when agnets intersect length of wall
	detected_collisionsH = np.logical_and(detected_collisionsH, intmat)

	# For larger environments
	# A = 10; B = 10
	# For smaller environments
	# A = 2; B = 5

	A = 10; B = 20

	# Compute force based vector and multiply by intersection matrix
	Fy = np.exp(-A*(np.abs(diffh) + sensor_error[:,np.newaxis]) + B)*diffh*intmat
	#Fy = .1*np.power(diffh, -1*np.ones((len(diffh), len(diffh[0]))))*intmat

	#Fy = -3/diffh*intmat*np.exp(-abs(diffh) + 3)
	#Fy = Fy*diffh*intmat

	low = agentsy[:, np.newaxis] >= map.limv.T[0]
	up = agentsy[:, np.newaxis] <= map.limv.T[1]
	intmat = up*low

	# *** collisions are only triggered when agnets intersect length of wall
	detected_collisionsV = np.logical_and(detected_collisionsV, intmat)


	Fx = np.exp(-A*(np.abs(diffv) + sensor_error[:,np.newaxis]) + B)*diffv*intmat

	#Fx = .2*np.power(diffv, -1*np.ones((len(diffv), len(diffv[0]))))*intmat
	#Fx = -3/diffv*intmat*np.exp(-abs(diffv) + 3)
	#Fx = Fx*diffv*intmat

	# Sum the forces between every wall into one force.
	Fx = np.sum(Fx, axis=1)
	Fy = np.sum(Fy, axis=1)
	# Combine x and y force vectors
	#F = np.array([[Fx[n], Fy[n]] for n in range(size)])
	F = np.stack((Fx, Fy), axis = 1)

	########### Add repulsion from endpoints of walls ##################


	endpoints = np.concatenate((map.limh, map.limv))
	# print('Wall endpoints: ', map.limh.T)
	# print('Wall endpoints: ', map.walls)

	new_array = [tuple(row) for row in map.walls]
	uniques = np.unique(new_array, axis = 0)

	# print('Remove endpoint duplicates: ', uniques)
	# print('Length difference is ', len(map.walls) - len(uniques))

	endpoints = uniques

	end_mag = cdist(swarm.agents, endpoints)

	diff = swarm.agents[:,:,np.newaxis] - endpoints.T[np.newaxis,:,:] 

	R = 15; r = 1.5

	repel = R*r*np.exp(-end_mag + r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	# print('repel shape: ', repel)
	repel = np.sum(repel, axis = 2)

	F -= repel
	

	swarm.wallCollision_state = np.logical_or(np.sum(detected_collisionsH, axis = 1), np.sum(detected_collisionsV, axis = 1))

	# print('Agents which have collided with walls: ', swarm.wallCollision_state)

	collision_duration = 25
	collision_delayBuffer = 2

	for n in range(swarm.size):

		if swarm.wallCollision_state[n] == 1:
			swarm.wallCollision_timeout[n] += 1

		if swarm.wallCollision_timeout[n] > collision_duration:
			# Agent is free to move again
			swarm.wallCollision_state[n] = 0
		
		if swarm.wallCollision_timeout[n] > collision_duration + collision_delayBuffer:
			swarm.wallCollision_timeout[n] = 0

	# input()

	return F






