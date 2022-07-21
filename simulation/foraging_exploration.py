
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



class pheromones(object):

	def __init__(self, size):

		self.positions = np.array([[999,999]])
		self.radius = 3
		self.decay_rate = 10

	def default_set(self):

		'''
		Beacon types:

		0 - Random walk
		1 - clockwise
		2 - Anti-clockwise
		3 - dispersion
		'''

		# Bench1 map
		# self.beacons = np.array([[15,-5], [20,15], [10,-30], [30,-30], [-30,30], [-30,-30], [-20,-20], [-30,-10] , [-20,0], [0, -5]])
		# self.radius = np.array([10])
		# self.behaviour_type = np.array([2, 1, 3, 1, 2, 1, 2, 1, 2,1])

		# Bench2 map
		self.beacons = np.array([[999,999]])
		self.radius = np.array([3])
		# self.behaviour_type = np.array([1, 2, 1, 1, 2, 1, 2, 1, 1])
		self.decay = 10

	def pheromone_repulsion(self, swarm):

		mag = cdist(self.positions, swarm.agents)

		# Reset agent behaviours before detecting intersection with beacons
		swarm.behaviour = 0*np.ones(swarm.size)

		

		




class ant_swarm(object):

	def __init__(self):

		self.agents = []
		self.speed = 0.5
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

		### Anti-consensus ###

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



		self.collision_count = np.zeros(self.size)

		self.objective_count = np.zeros(self.size)
		self.agent_objective_states = [None for x in range(self.size)]
		# mag = 0.4
		# self.happiness_noise = np.random.uniform(-mag, mag, self.size)
		self.sensor_fault = np.zeros(self.size)
		self.doorway_occupied = np.zeros(6)

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

		
		forage(self, 0.01)
	

	
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



def forage(swarm, param, pheromones):

	alpha = 0.01; beta = 50

	noise = param*np.random.randint(-beta, beta, (swarm.size))

	# Move only agents with random behaviour set
	states = swarm.behaviour == 0

	swarm.headings += states*noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	# Agent avoidance
	R = 20; r = 2; A = 1; a = 20	
	
	a = np.zeros((swarm.size, 2))

	#B = np.zeros((swarm.size, 2))
	#B = beacon(swarm)
	phero_rep = pheromones.pheromone_repulsion(swarm)

	A = asim.avoidance(swarm.agents, swarm.map)
	a += A + G + phero_rep

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.stack((Wx, Wy), axis = 1)

	# Move only agents with random behaviour set
	states = swarm.behaviour == 0
	swarm.agents += states[:,np.newaxis]*W


	###### Drop new pheromones


	


	swarm.agents = asim.continuous_boundary(swarm.agents, swarm.map)

def rotate_clock(swarm, direction, param):

	noise = param*np.random.randint(-2, 2, swarm.size)
	
	states = swarm.behaviour == 11

	swarm.headings += states*noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	a = np.zeros((swarm.size,2))
	B = np.zeros((swarm.size, 2))
	B = asim.beacon(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	a += G + A + B

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])

	# Move only agents with random behaviour set
	states = swarm.behaviour == 11

	swarm.agents += states[:,np.newaxis]*W 
	swarm.agents = asim.continuous_boundary(swarm.agents, swarm.map)

def rotate_anti(swarm, direction, param):

	noise = param*np.random.randint(-1, 3, swarm.size)
	
	states = swarm.behaviour == 99

	swarm.headings += states*noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	a = np.zeros((swarm.size,2))
	B = np.zeros((swarm.size, 2))
	B = asim.beacon(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	a += G + A + B

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])

	# Move only agents with random behaviour set
	states = swarm.behaviour == 10

	swarm.agents += states[:,np.newaxis]*W 
	swarm.agents = asim.continuous_boundary(swarm.agents, swarm.map)


def flocking(swarm, repel, attract, comm_range, align, noise):

	argname = ['repel', 'attract', 'align']
	args = [repel, attract, align]
	for n in range(len(args)):
		if args[n] > 1 or args[n] < 0:
			raise ValueError("Value %s must be within the range of 0 to 1." % (argname[n]))

	R = repel; r = 3; A = attract; a = 3

	states = swarm.behaviour == 22

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Determine headings
	nearest = mag <= comm_range

	# n x n matrix of headings of agents which are adjacent
	neighbour_headings = swarm.headings*nearest

	# Sum headings for each agent
	neighbour_headings_tot = np.sum(neighbour_headings, axis = 1)

	# average headings with neighbours
	new_headings = neighbour_headings_tot/(np.sum(nearest, axis = 1))

	# Determine the difference between current heading and neighbour avg
	heading_diff = swarm.headings - new_headings

	# Adjust heading to neighbours. Degree of alignment determined by align param
	swarm.headings -= states*((align*heading_diff) + 0.01*np.random.randint(-10,11, swarm.size))

	# Calculate new heading vector
	strength = 0.05
	gx = strength*np.cos(swarm.headings)
	gy = strength*np.sin(swarm.headings)
	#G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])
	G = np.stack((gx, gy), axis = 1)
	

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	avoid = asim.avoidance(swarm.agents, swarm.map)
	#avoid = bsim.fieldmap_avoidance(swarm)
	#mag = mag*nearest
	repel = 10*R*r*np.exp(-3*mag/comm_range)[:,np.newaxis,:]*diff/(swarm.size-1)
	repel = repel*nearest[:,np.newaxis,:]
	repel = np.sum(repel, axis = 0).T

	attract = 10*A*a*np.exp(-3*mag/comm_range)[:,np.newaxis,:]*diff/(swarm.size-1)
	attract = attract*nearest[:,np.newaxis,:]	
	attract = np.sum(attract, axis = 0).T

	total = 0
	# total +=  noise + repel + G - attract - avoid
	
	direction = np.ones((swarm.size, 2))

	total +=  G - repel + attract - avoid
	

	vecx = total.T[0]
	vecy = total.T[1]
	angles = np.arctan2(vecy, vecx)

	swarm.headings = (2*np.pi + angles) * (angles < 0) + angles*(angles > 0)

	swarm.headings = states*angles + np.logical_not(states)*swarm.headings 

	Wx = swarm.speed*np.cos(swarm.headings)
	Wy = swarm.speed*np.sin(swarm.headings)


	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W



def flocking_infoGain(swarm, repel, attract, comm_range, align, noise):

	argname = ['repel', 'attract', 'align']
	args = [repel, attract, align]
	for n in range(len(args)):
		if args[n] > 1 or args[n] < 0:
			raise ValueError("Value %s must be within the range of 0 to 1." % (argname[n]))

	R = repel; r = 3; A = attract; a = 3

	states = swarm.behaviour == 10

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Determine headings
	nearest = mag <= comm_range

	# n x n matrix of headings of agents which are adjacent
	neighbour_headings = swarm.headings*nearest

	# Sum headings for each agent
	neighbour_headings_tot = np.sum(neighbour_headings, axis = 1)

	# average headings with neighbours
	new_headings = neighbour_headings_tot/(np.sum(nearest, axis = 1))

	# Determine the difference between current heading and neighbour avg
	heading_diff = swarm.headings - new_headings

	# Adjust heading to neighbours. Degree of alignment determined by align param
	swarm.headings -= states*((align*heading_diff) + 0.01*np.random.randint(-10,11, swarm.size))

	# Calculate new heading vector
	strength = 0.05
	gx = strength*np.cos(swarm.headings)
	gy = strength*np.sin(swarm.headings)
	#G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])
	G = np.stack((gx, gy), axis = 1)
	

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	avoid = asim.avoidance(swarm.agents, swarm.map)
	#avoid = bsim.fieldmap_avoidance(swarm)
	#mag = mag*nearest
	repel = 10*R*r*np.exp(-3*mag/comm_range)[:,np.newaxis,:]*diff/(swarm.size-1)
	repel = repel*nearest[:,np.newaxis,:]
	repel = np.sum(repel, axis = 0).T

	attract = 10*A*a*np.exp(-3*mag/comm_range)[:,np.newaxis,:]*diff/(swarm.size-1)
	attract = attract*nearest[:,np.newaxis,:]	
	attract = np.sum(attract, axis = 0).T


	info_repel = 10*R*r*np.exp(-3*mag/comm_range)[:,np.newaxis,:]*diff/(swarm.size-1)
	print(info_repel.shape)
	info_repel = swarm.info_gain*info_repel
	info_repel = info_repel*nearest[:,np.newaxis,:]
	info_repel = np.sum(info_repel, axis = 0).T
	print(swarm.info_gain)


	total = 0
	# total +=  noise + repel + G - attract - avoid
	
	direction = np.ones((swarm.size, 2))

	total +=  G - repel + attract - avoid - info_repel
	

	vecx = total.T[0]
	vecy = total.T[1]
	angles = np.arctan2(vecy, vecx)

	swarm.headings = (2*np.pi + angles) * (angles < 0) + angles*(angles > 0)

	swarm.headings = states*angles + np.logical_not(states)*swarm.headings 

	Wx = swarm.speed*np.cos(swarm.headings)
	Wy = swarm.speed*np.sin(swarm.headings)


	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W


