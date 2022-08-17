import simulation.map_gen as map_gen
import simulation.environment
import simulation.asim as asim
import simulation.faulty_swarm as faulty_swarm
import random
import numpy as np
import pickle
import sys
import time 


import matplotlib.pyplot as plt
from matplotlib import animation, rc, rcParams
rcParams['animation.embed_limit'] = 2**128

from matplotlib import collections  as mc
import matplotlib as mpl
from matplotlib import image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


from scipy.spatial.distance import cdist, pdist, euclidean





############################################################################################
def getImage(path):
   return OffsetImage(plt.imread(path, format="png"), zoom=.03)

#   Add image as custom marker

# constant
# dpi = 72
path = 'warning.png'
# # read in our png file
# im = image.imread(path)
# image_size = im.shape[1], im.shape[0]


# # plot our line with transparent markers, and markersize the size of our image
# line, = ax.plot((1,2,3,4),(1,2,3,4),"bo",mfc="None",mec="None",markersize=image_size[0] * (dpi/ 96))
# # we need to make the frame transparent so the image can be seen
# # only in trunk can you put the image on top of the plot, see this link:
# # http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg14534.html
# ax.patch.set_alpha(0)
# ax.set_xlim((0,5))
# ax.set_ylim((0,5))

# # translate point positions to pixel positions
# # figimage needs pixels not points
# line._transform_path()
# path, affine = line._transformed_path.get_transformed_points_and_affine()
# path = affine.transform_path(path)
# for pixelPoint in path.vertices:
#     # place image at point, centering it
#     fig.figimage(im,pixelPoint[0]-image_size[0]/2,pixelPoint[1]-image_size[1]/2,origin="upper")

# plt.show()
##########################################################################################################


#env_map = map_gen.convert_to_env_object(polys)
# First set up the figure, the axis, and the plot element we want to animate
fig, ax1 = plt.subplots( figsize=(10,10), dpi=80, facecolor='w', edgecolor='k')
# fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')
#plt.close()

# dim = 45
# ax1.set_xlim((-dim, dim))
# ax1.set_ylim((-dim, dim))

# Set how data is plotted within animation loop
global line, line1
# Agent plotting 
robot_size = 10

random_pos = [(0,0),(10,10),(20,34)]

# for entry in random_pos:
# 	ab = AnnotationBbox(getImage(path), entry, frameon=False)
# 	ax1.add_artist(ab)

# Type of agent plotting
agent_pos, = ax1.plot([], [], 'rh', markersize = 8, markeredgecolor="black", alpha = 1, zorder=10)

happy_agents, = ax1.plot([], [], 'gh', markersize = 8, markeredgecolor="green", alpha = 1, zorder=10)
unhappy_agents, = ax1.plot([], [], 'rh', markersize = 8, markeredgecolor="black", alpha = 1, zorder=10)

faulty_pos, = ax1.plot([], [], 'r*', markersize = 8, alpha = 1, zorder=10)

repellents, = ax1.plot([], [], 'ro', markersize=70, alpha=0.3)

trails, = ax1.plot([], [], 'bh', markersize = 6, alpha = 0.2)
malicious_trails, = ax1.plot([], [], 'bh', markersize = 6, alpha = 0.2)



# shadow plotting
line1, = ax1.plot([], [], 'bh', markersize = 6, markeredgecolor="black", alpha = 0.1)
line2, = ax1.plot([], [], 'bh', markersize = 6, markeredgecolor="black", alpha = 0.1)
line3, = ax1.plot([], [], 'bh', markersize = 6, markeredgecolor="black", alpha = 0.1)

low_emp, = ax1.plot([], [], 'ro', markersize = 8, markeredgecolor="red", alpha = 0.05)
high_emp, = ax1.plot([], [], 'go', markersize = 8, markeredgecolor="green", alpha = 0.05)

agent_headings, = ax1.plot([], [], 'm*', markersize = 4, alpha = 0.9)

fsize = 12

time_text = ax1.text(-20, 42, '', fontsize = fsize)
box_text = ax1.text(3, 42, '', color = 'red', fontsize = fsize)
cov_text = ax1.text(20, 42, '', color = 'blue', fontsize = fsize)
# cov_line, = ax2.plot([],[], 'r-', markersize = 5, label='Social')

behaviour_text = ax1.text(3, 45, '', color = 'purple', fontsize = fsize)


def init():
    
    line1.set_data([], [])
    return ( line1,)

seed = random.randrange((2**32) - 1)
seed = 99999

random.seed(seed)
np.random.seed(seed)

print('\nChosen seed: ', seed)

env_map = asim.map()
env_map.map1()
env_map.swarm_origin = np.array([44,15])
env_map.gen()


ax1.set_ylim([-44,44])
ax1.set_xlim([-44,44])

# ax2.set_ylim([0,1])

plt.legend(loc="upper left")


# [ax1.plot([env_map.obsticles[a].start[0], env_map.obsticles[a].end[0]], 
# 	[env_map.obsticles[a].start[1], env_map.obsticles[a].end[1]], '-', 
# 			color = 'black', lw=3, markeredgecolor = 'black', markeredgewidth = 3) for a in range(len(env_map.obsticles))]

# ----------- Plot walls except the opening entry point ---------
for a in range(len(env_map.obsticles)):

	if a != 0:
		ax1.plot([env_map.obsticles[a].start[0], env_map.obsticles[a].end[0]], [env_map.obsticles[a].start[1], env_map.obsticles[a].end[1]], '-', color = 'black', lw=3, markeredgecolor = 'black', markeredgewidth = 3) 


timesteps = 1000


# ===================== Swarm Faults/Malicious behaviours =========================


totSwarm_size = 20

# Swarm faults

# positive sensor error added to distance measurement between agents
num_sensorfault = 0
# Channels of communication between certain robots is completely lost
num_robotblind = 0
# Motor error causing agents to move half speed with a degree of fluctuation
num_motorslow = 5
# agents have a persistent heading error 
num_headingerror = 0

# Malicious behaviours
malicious_blockers = 0
'''
Malicious broadcasting agents always communicate that they have maximum happiness.
Agents which have significantly lower happiness will copy broadcasting agents.
Broadcasters do not attempt to find good behaviours, creating sinks where clusters of 
agents form copying the behaviour of the malicious agent.
'''
num_maliciousBroadcast = 0

if malicious_blockers <= 1:
	blockers_active = False
else:
	blockers_active = True


# =================================================================================
base_swarm = map_gen.swarm()
base_swarm.size = totSwarm_size
base_swarm.speed = 0.5
base_swarm.origin = env_map.swarm_origin[:]
base_swarm.map = env_map
base_swarm.gen_agents()

base_swarm.motor_speeds = np.ones(totSwarm_size)

field, grid = asim.potentialField_map(base_swarm.map)
base_swarm.field = field
base_swarm.grid = grid

# Set target positions
targets = asim.target_set()
targets.radius = 2.5
targets.set_state('4x4')
targets.reset()


score = 0


# Declare agent motion noise
noise = np.random.uniform(-.1,.1,(timesteps, base_swarm.size, 2))
coverage_data = list()
time_data = list()

agent_set = np.arange(0, totSwarm_size, 1)
base_swarm.behaviour = np.random.randint(1,9, totSwarm_size)
for i in range(10):

	if i%7 == 0 and len(agent_set) >= 1:

		# At each step pick a random agent to spawn
		pick = np.random.choice(agent_set)
		agent_set = np.delete(agent_set, np.where(agent_set == pick)[0])
	
		base_swarm.agents[pick] = np.array([base_swarm.map.swarm_origin[0], base_swarm.map.swarm_origin[1]])
		base_swarm.previous_state[pick] = np.array([base_swarm.map.swarm_origin[0], base_swarm.map.swarm_origin[1]])
		base_swarm.opinion_timelimit[pick] = 50
		base_swarm.behaviour[pick] = 4

	base_swarm.time = i

	base_swarm.iterate(noise[i-1])

	#print('swarm beh: ', swarmy.behaviour)

	base_swarm.get_state_opinion()
	#swarmy.died += asim.boundary_death(swarmy, swarmy.map)
	score += targets.get_state(base_swarm, i, timesteps)

	coverage_data.append(targets.coverage)
	time_data.append(i)

normal_perf = np.trapz(coverage_data, dx=1)

# ax2.plot(time_data, coverage_data, 'g-', markersize = 5, label='Disperse')

# ax2.set_ylabel('Coverage', fontsize = 15)
# ax2.set_xlabel('Time (seconds)', fontsize = 15)


# ax2.legend(loc='upper left', fontsize = 14)


swarmy = faulty_swarm.swarm()
swarmy.size = totSwarm_size - malicious_blockers
swarmy.speed = 0.3
swarmy.origin = env_map.swarm_origin[:]
swarmy.map = env_map
swarmy.gen_agents()

if blockers_active == True:
	malicious_swarm = faulty_swarm.malicious_swarm()
	malicious_swarm.size = malicious_blockers
	malicious_swarm.speed = 0.3
	malicious_swarm.origin = env_map.swarm_origin[:]
	malicious_swarm.map = env_map
	malicious_swarm.gen_agents()
	malicious_swarm.behaviour = 10*np.ones(malicious_swarm.size)
else:
	malicious_swarm = faulty_swarm.malicious_swarm()
	malicious_swarm.size = 10
	malicious_swarm.speed = 0.3
	malicious_swarm.origin = env_map.swarm_origin[:]
	malicious_swarm.map = env_map
	malicious_swarm.gen_agents()
	# Initilaize swarm positions outside environment
	malicious_swarm.agents = 1000*np.ones((10,2))


# Generate potential field map of environment
field, grid = asim.potentialField_map(swarmy.map)
swarmy.field = field
swarmy.grid = grid

targets = asim.target_set()
targets.radius = 2.5
targets.set_state('4x4')
targets.reset()


# ===================== Setting fault intermittance ======================

swarmy.fault_rate = 100
'''
   Fault intermittance sets the proportion of time that
   the fault is active. i.e 0 means the fault is never active 
   and 1 would mean the fault is always active.

   The fault rate defines the period over which the fault can
   switch between active and inactive.
'''
swarmy.fault_intermittance = 0.5
swarmy.fault_limit = np.random.randint(0, swarmy.fault_rate, swarmy.size)


coverage_data = []
time_data = []
happy_data = []

# ax1.plot(targets.targets.T[0], targets.targets.T[1], 'bo')



# Set the length of agent trails in simulation
max_length = 10*swarmy.size
#max_length = 200000000000000*swarmy.size
agent_trails = 1000*np.ones((swarmy.size, 2))
maliciousAgent_trails = 1000*np.ones((swarmy.size, 2))


# Generate swarm motion noise for entire simulation
noise = np.random.uniform(-.1,.1,(timesteps, swarmy.size, 2))
malicious_noise = np.random.uniform(-.1,.1,(timesteps, malicious_swarm.size, 2))

score = 0 

# ***** Initially randomise the starting behaviour of agents
swarmy.behaviour = np.random.randint(1, 9, swarmy.size)
swarmy.behaviour = 1*np.ones(swarmy.size)
# swarmy.param = 2


#====================== Assign robots which will have faults =================================


#  ----------  Agents with sensor error fault ----------- 


swarmy.sensor_mean = 10
swarmy.sensor_dev = 2

malicious_swarm.sensor_mean = 10
malicious_swarm.sensor_dev = 2

agent_set = np.arange(0, swarmy.size, 1)

for n in range(0, num_sensorfault):
	print(agent_set)
	pick = np.random.choice(agent_set)
	print(pick)
	agent_set = np.delete(agent_set, np.where(agent_set == pick)[0])
	# Chosen agent has error added to sensors
	swarmy.sensor_fault[pick] = 1


for n in range(0, num_robotblind):

	pick = np.random.randint(0, swarmy.size - 1)

	swarmy.sensor_fault[pick] = 1


# # -------------- Agents with slow motors ---------------



# # Define default motor speeds for agents
swarmy.motor_error = np.zeros(swarmy.size)
swarmy.motor_speeds = np.ones(swarmy.size)

malicious_swarm.motor_error = np.zeros(malicious_swarm.size)
malicious_swarm.motor_speeds = np.ones(malicious_swarm.size)
swarmy.motor_mean = 0.5
swarmy.motor_dev = 0.3
for n in range(0, num_motorslow):

	print(agent_set)
	pick = np.random.choice(agent_set)
	print(pick)
	agent_set = np.delete(agent_set, np.where(agent_set == pick)[0])
	# Chosen agent has error added to sensors
	
	swarmy.motor_error[pick] = 1

# ------------- Agents with heading error --------------


swarmy.heading_error = np.zeros(swarmy.size)
malicious_swarm.heading_error = np.zeros(malicious_swarm.size)
# swarmy.motor_speeds = np.ones(swarmy.size)
swarmy.heading_mean = 1.2
swarmy.heading_dev = 0.2
for n in range(0, num_headingerror):

	print(agent_set)
	pick = np.random.choice(agent_set)
	print(pick)
	agent_set = np.delete(agent_set, np.where(agent_set == pick)[0])
	# Chosen agent has error added to sensors
	
	swarmy.heading_error[pick] = 1

# ------------- Malicious broadcasting agents ----------------------------

swarmy.malicious_broadcasters = np.zeros(swarmy.size)
for n in range(0, num_maliciousBroadcast):
	print(agent_set)
	pick = np.random.choice(agent_set)
	print(pick)
	agent_set = np.delete(agent_set, np.where(agent_set == pick)[0])
	# pick = np.random.randint(0, swarmy.size - 1)
	# Chosen agent performs the malicious swarm behaviour
	swarmy.malicious_broadcasters[pick] = 1

# happiness_plot = input("Plot happiness?")
# if happiness_plot == 'yes' or happiness_plot == 'y':
# 	print('plotting happiness')
# 	happiness_plot = True

spawned_state = np.zeros((swarmy.size + malicious_swarm.size))	

# Agents are initially not in view
swarmy.agents = 1000*np.ones((swarmy.size, 2))
malicious_swarm.agents = 1000*np.ones((malicious_swarm.size, 2))
agent_set = np.arange(0, swarmy.size + malicious_blockers, 1)

global plot_happiness
global plot_faulty

plot_happiness = True
plot_faulty = False

time_data = list()
coverage_data = list()

def on_press(event):
	# print('press', event.key)
	# sys.stdout.flush()
	if event.key == 'up':
		swarmy.behaviour = 1*np.ones(swarmy.size)
	if event.key == 'left':
		swarmy.behaviour = 4*np.ones(swarmy.size)
	if event.key == 'down':
		swarmy.behaviour = 2*np.ones(swarmy.size)
	if event.key == 'right':
		swarmy.behaviour = 3*np.ones(swarmy.size)
	swarmy.opinion_timer = 1*np.ones(swarmy.size)
	swarmy.opinion_timelimit = 50*np.ones(swarmy.size)
	# sys.stdout.flush()

	global plot_happiness
	global plot_faulty

	if event.key == 'h':
		# turn on/off agent happiness display
		plot_happiness = np.logical_not(plot_happiness)
	if event.key == 'j':
		# turn on/off agent happiness display
		plot_faulty = np.logical_not(plot_faulty)



# pygame stuff

# pygame.init()




fig.canvas.mpl_connect('key_press_event', on_press)
swarmy.param = 10

agents_withFaults = np.logical_or(swarmy.malicious_broadcasters, swarmy.heading_error)

agents_withFaults = np.logical_or(agents_withFaults, swarmy.motor_error)

agents_withFaults = np.logical_or(agents_withFaults, swarmy.sensor_fault)

faulty_indicies = np.where(agents_withFaults == 1)[0]
print(faulty_indicies)

sim_speed = list()

def animate(i):

	# Check gird intersection
	#grid_check(swarmy)
	global score
	global happy_agents
	global unhappy_agents
	global agent_trails
	global maliciousAgent_trails

	global agent_set
	global plot_happiness

	start = time.time()

	print('swarmy fault active states: ', swarmy.fault_active)

	# ---------------------------- Spawn agents from edge of environment over time -------------------------------------

	total_agentSize = swarmy.size + malicious_blockers
	# input()
	pos_variance = 4
	if i%7 == 0 and len(agent_set) >= 1:

		# At each step pick a random agent to spawn
		pick = np.random.choice(agent_set)
		agent_set = np.delete(agent_set, np.where(agent_set == pick)[0])
		if pick < malicious_blockers:
			# spawn a malicious agent
			malicious_swarm.agents[pick] = np.array([swarmy.map.swarm_origin[0], swarmy.map.swarm_origin[1]])
		else:
			swarmy.agents[pick-malicious_swarm.size] = np.array([swarmy.map.swarm_origin[0], swarmy.map.swarm_origin[1] + np.random.uniform(-pos_variance, pos_variance)])
			swarmy.previous_state[pick-malicious_swarm.size] = np.array([swarmy.map.swarm_origin[0], swarmy.map.swarm_origin[1] + np.random.uniform(-pos_variance ,pos_variance)])
			swarmy.opinion_timelimit[pick-malicious_swarm.size] = 50
			swarmy.behaviour[pick-malicious_swarm.size] = 4
	if i <= 150:
		swarmy.behaviour = 4*np.ones(swarmy.size)
	else:
		swarmy.param = 60


	# ------------------------------------------------------------------------------------------------------------------

	swarmy.time = i

	# swarmy.happiness = np.random.normal(0.9,.01, swarmy.size)
	if i >= 50:
		faulty_swarm.collision_check(swarmy, malicious_swarm)
	swarmy.iterate(malicious_swarm, noise[i-1])


	# print('Swarm happiness: ', swarmy.happiness)
	# print('Swarm opinion timer: ', swarmy.opinion_timer)
	# print('swarm previous states:', swarmy.previous_state)

	# swarmy.behaviour = np.random.randint(1,9, swarmy.size)
	if blockers_active == True:
		malicious_swarm.iterate(malicious_noise[i-1])
		malicious_swarm.get_state()
	#print('swarm beh: ', swarmy.behaviour)
	# print(swarmy.behaviour)

	swarmy.get_state_opinion()
	swarmy.get_state()
	#swarmy.died += asim.boundary_death(swarmy, swarmy.map)
	score += targets.get_state(swarmy, i, timesteps)


	#repellents.set_data(swarmy.beacon_rep.T[0],swarmy.beacon_rep.T[1])

	time_data.append(i)
	coverage_data.append(targets.coverage)

	agents = np.concatenate((swarmy.agents, malicious_swarm.agents), axis = 0)

	x = agents.T[0]
	y = agents.T[1]

	

	time_text.set_text('Time: (%d/%d)' % (i, timesteps))

	#box_text.set_text('Fitness: %.2f' % (score/len(targets.targets)))
	cov_text.set_text('Coverage: %.2f' % (targets.coverage))

	#mass.set_data(swarmy.centermass[0], swarmy.centermass[1])

	agent_trails = np.concatenate((agent_trails, swarmy.agents), axis =0)

	if len(agent_trails) > max_length:

		agent_trails = agent_trails[swarmy.size:]

	trails.set_data(agent_trails.T[0], agent_trails.T[1])


	maliciousAgent_trails = np.concatenate((maliciousAgent_trails, malicious_swarm.agents), axis = 0)

	if len(maliciousAgent_trails) > 10*malicious_swarm.size:

		maliciousAgent_trails = maliciousAgent_trails[malicious_swarm.size:]

	malicious_trails.set_data(maliciousAgent_trails.T[0], maliciousAgent_trails.T[1])

	

	# Plot with or without agent happiness
	if plot_happiness == True:

		setA = swarmy.agents[swarmy.happiness <= 0.5]
		setB = swarmy.agents[swarmy.happiness > 0.5]

		happy_agents.set_data(setB.T[0], setB.T[1])
		unhappy_agents.set_data(setA.T[0], setA.T[1])
		agent_pos.set_data([],[])

	else:

		agent_pos.set_data(x,y)
		happy_agents.set_data([],[])
		unhappy_agents.set_data([],[])


	# Highlight agents which are faulty
	if plot_faulty == True and len(faulty_indicies) != 0:

		positions = swarmy.agents[agents_withFaults == 1]
		# print('fault position data: ', positions)
		faulty_pos.set_data(positions.T[0] + 1, positions.T[1] + 1)
	else:
		faulty_pos.set_data([],[])


	# collided_agents = swarmy.agents[swarmy.collision_state == 1]
	# print(collided_agents)
	# for entry in collided_agents:
	# 	# ab = AnnotationBbox(getImage(path), (entry[0], entry[1]), frameon=False)
	# 	# ax1.add_artist(ab)
	# 	ax1.plot(entry, marker = getImage(path))
	# fig.canvas.draw()

	# happy.set_data(setB.T[0], setB.T[1])
	# unhappy.set_data(setA.T[0], setA.T[1])

	# happy_agents = np.concatenate((happy_agents, setB), axis = 0)
	# unhappy_agents = np.concatenate((unhappy_agents, setA), axis = 0)

	# if len(happy_agents) > max_length:

	# 	happy_agents = happy_agents[swarmy.size:]
		

	# if len(unhappy_agents) > .5*max_length:

	# 	unhappy_agents = unhappy_agents[swarmy.size:]



	# high_emp.set_data(happy_agents.T[0], happy_agents.T[1])
	# low_emp.set_data(unhappy_agents.T[0], unhappy_agents.T[1])

	
	# time_data.append(i)



	# cov_line.set_data(time_data, coverage_data)

	taken = 1000*(time.time() - start)
	sim_speed.append(taken)
	
	# return (line1, line2, line3, time_text, box_text,cov_text, 
	# 		agent_headings, trails, malicious_trails,agent_pos, behaviour_text, cov_line, 
	# 		happy_agents, unhappy_agents, faulty_pos)
	return (time_text,cov_text, 
			agent_headings, trails, malicious_trails,agent_pos, 
			happy_agents, unhappy_agents, faulty_pos)




anim = animation.FuncAnimation(fig, animate, init_func=init,
                            frames=timesteps, interval=30, blit=True, repeat = False)


# Note: below is the part which makes it work on Colab
#rc('animation', html='jshtml')
# anim.save('outputs/malicious/advPres/happiness.mp4', fps=25, dpi=100)
anim


plt.show()

# print('Integral of Coverage over time of normal behaviour = %.2f' % (normal_perf/timesteps))
# print('Integral of Coverage over time of with faults = %.2f' % (np.trapz(coverage_data, dx=1)/timesteps))

# plt.plot(sim_speed)
# plt.ylabel('Step speed (ms)')
# plt.show()