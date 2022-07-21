#!/usr/bin/env python



import time
import pydot
import math
import tty
import termios
import sys
import select
import pickle
import subprocess
import py_trees
import argparse
import functools
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import numpy as np

import behtree.treegen as tg
import simulation.asim as asim

from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist


def render_tree(ind, swarm, targets, name):


	bt = tg.tree().decode(ind, swarm, targets)
	py_trees.display.render_dot_tree(bt.root, name = name)
	print(py_trees.display.ascii_tree(bt.root))

def justsim(ind, oldswarm, targets, timesteps, trials = 10):

	# Decode genome into executable behaviour tree
	
	fits = []

	record = np.zeros(trials)

	dur = 0
	score = 0

	
	field, grid = asim.potentialField_map(oldswarm.map)
	

	for k in range(trials):
		
		swarm = asim.swarm()
		swarm.size = oldswarm.size
		swarm.behaviour = 'none'
		swarm.speed = 0.5
		swarm.origin = np.array([0,0])
		swarm.gen_agents()
		
		targets = asim.target_set()
		targets.set_state('uniform')
		targets.radius = 5
		targets.reset()

		env = asim.map()
		'''
		The map has to be set to the same as passed into the evolution!!!
		'''
		env.brl_simp([0,0,0,0,0,0,0,0])
		env.gen()
		swarm.map = env

		swarm.field = field
		swarm.grid = grid

		fitness = 0; score = 0 ;t = 0
		found = False
		bt = tg.tree().decode(ind, swarm, targets)
		swarm.beacon_set = []

		swarm.behaviour = 'disperse'
		swarm.param = 3

		noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))

		

		#asim.plot_field(swarm)


		now = time.time()
		while t <= timesteps and found == False:
			
			#bt.tick()
			swarm.iterate(noise[t-1])
			swarm.get_state()
			#score = targets.get_state(swarm, t)
			if targets.found == len(targets.targets):
				found = True

			t += 1

		maxsize = 300
		fitness = 0
		fitness = score/len(targets.targets)
		#fitness = fitness - (len(ind.genome)*0.0001)
		if fitness < 0: fitness = 0
		print ('fitness: ', fitness)

		coverage = score
		record[k] = fitness
		print('Trail ', k)
		print('Completion time: ', 1000*(time.time()-now))

		targets.reset()
		swarm.reset()

	print('Set of coverages: ', record)

	print('The mean = ', np.mean(record))
	print('The Standard deviation = ', np.std(record))

	print('length of ind = ', len(ind.genome))

	maxsize = 300
	fitness = 0
	fitness = score/(trials*len(targets.targets))
	fitness = fitness - (len(ind.genome)*0.0001)
	if fitness < 0: fitness = 0
	print ('Average fitness: ', fitness)
	print(ind.tree)

	return fits

def batch(ind, oldswarm, targets, timesteps, trials = 10):

	batch = 1
	batchname = 'outputs/brl_environment/batch11'
	mapname = 'brlsimp_State4_'

	sols = ['outputs/aamas_environments/'+ batchname + '/' + mapname + '_10agents'+ str(batch),
			'outputs/aamas_environments/'+ batchname + '/' + mapname + '_20agents'+ str(batch),
			'outputs/aamas_environments/'+ batchname + '/' + mapname + '_30agents'+ str(batch),
			'outputs/aamas_environments/'+ batchname + '/' + mapname + '_40agents'+ str(batch),
			'outputs/aamas_environments/'+ batchname + '/' + mapname + '_50agents'+ str(batch)]

	sols = [batchname + '/' + mapname + str(1),
			batchname + '/' + mapname + str(2),
			batchname + '/' + mapname + str(3),
			batchname + '/' + mapname + str(4),
			batchname + '/' + mapname + str(5)]


	results = []		
	size = []

	agents = 100

	for solution in sols:

		with open(solution + '_hallfame', 'rb') as input:
			hall = pickle.load(input)
		
		ind = hall[0].copy()

		fits = []

		record = np.zeros(trials)

		# states = [[0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0]]
		states = [[0,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0],[0,0,0,0,0,0,0,0]]

		env = asim.map()
		env.brl_mod(states[0])
		env.gen()

		field1, grid = asim.potentialField_map(env)

		env.brl_mod(states[1])
		env.gen()
		field2, grid = asim.potentialField_map(env)

		fields = [field1, field2, field1]
	
		score = 0

		state_counter = 0
		for k in range(trials):
			
			swarm = asim.swarm()
			swarm.size = agents
			swarm.behaviour = 'none'
			swarm.speed = 0.5
			swarm.origin = np.array([-15,-35])
			swarm.gen_agents()
			
			targets = asim.target_set()
			targets.set_state('brlset')
			targets.radius = 2.5
			targets.reset()

			env = asim.map()
			'''
			The map has to be set to the same as passed into the evolution!!!
			'''
			env.brl_mod(states[state_counter])
			env.gen()
			swarm.map = env

			noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))

			swarm.field = fields[state_counter]
			swarm.grid = grid

			state_counter += 1
			if state_counter > 2:
				state_counter = 0


			fitness = 0; score = 0 ;t = 0
			found = False
			bt = tg.tree().decode(ind, swarm, targets)
			swarm.beacon_set = []
			
			now = time.time()
			# swarm.behaviour = 'disperse'
			# swarm.param = 60
			while t <= timesteps and found == False:
				
				bt.tick()
				swarm.iterate(noise[t-1])
				swarm.get_state()
				score = targets.get_state(swarm, t)
				if targets.found == len(targets.targets):
					found = True
				t += 1

			maxsize = 300
			fitness = 0
			fitness = score/len(targets.targets)
		
			if fitness < 0: fitness = 0
			
			coverage = score
			record[k] = fitness
			

			sys.stdout.write("Test progress %s: %.2f%%   \r" % (solution, 100*k/trials) )
			sys.stdout.flush()

			targets.reset()
			swarm.reset()

		print('Average coverage = ', np.mean(record))

		results.append(record)
		size.append(len(ind.genome))
		# agents += 10


	results_disp = list()
	agents = 100
	# Repeat but just for dispersion benchmark
	for solution in sols:

		with open(solution + '_hallfame', 'rb') as input:
			hall = pickle.load(input)
		
		ind = hall[0].copy()

		fits = []

		record_disp = np.zeros(trials)

		dur = 0
		score = 0
		for k in range(1):
			
			swarm = asim.swarm()
			swarm.size = agents
			swarm.behaviour = 'none'
			swarm.speed = 0.5
			swarm.origin = np.array([-15,-35])
			swarm.gen_agents()
			
			targets = asim.target_set()
			targets.set_state('brlset')
			targets.radius = 2.5
			targets.reset()

			env = asim.map()
			'''
			The map has to be set to the same as passed into the evolution!!!
			'''
			env.brl_mod(states[state_counter])
			env.gen()
			swarm.map = env

			noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))

			field, grid = asim.potentialField_map(swarm.map)
			swarm.field = field
			swarm.grid = grid


			state_counter += 1
			if state_counter > 2:
				state_counter = 0

			fitness = 0; score = 0 ;t = 0
			found = False
			bt = tg.tree().decode(ind, swarm, targets)
			swarm.beacon_set = []
			
			now = time.time()
			swarm.behaviour = 'disperse'
			swarm.param = 60
			while t <= timesteps and found == False:
				
				swarm.iterate(noise[t-1])
				swarm.get_state()
				score = targets.get_state(swarm, t)
				if targets.found == len(targets.targets):
					found = True
				t += 1

			fitness = 0
			fitness = score/len(targets.targets)

			coverage = score
			record_disp[k] = fitness

			sys.stdout.write("Test progress %s: %.2f%%   \r" % (solution, 100*k/trials) )
			sys.stdout.flush()

			targets.reset()
			swarm.reset()

		print('Average coverage = ', np.mean(record))

		results_disp.append(record_disp)
		size.append(len(ind.genome))
		# agents += 10


	# Multiple box plots on one Axes
	fig, ax = plt.subplots(figsize=(7.5,7), dpi=100)
	#ax.boxplot(results)
	labels = ['1','2','3','4','5']
	ax.set_ylim([0,1])


	ax.set_xlabel('Swarm Size', fontsize = 15)
	ax.set_ylabel('Coverage', fontsize = 15)

	bp = ax.boxplot(results, patch_artist = True)
	bp_disp = ax.boxplot(results_disp, patch_artist = True)
	
	#set boxplot colours

	colors = ['pink','pink','pink','pink','pink']
	colors_a = ['white','white','white','white','white']

	for patch, color in zip(bp['boxes'], colors_a):
		patch.set_facecolor(color)

	for patch, color in zip(bp_disp['boxes'], colors):
		patch.set_facecolor(color)

	ax.set_xticklabels(labels, fontsize=12)

	ax.legend([bp_disp["boxes"][0], bp["boxes"][0], ], ['Dispersion only', 'With supervision'], loc='lower right')

	

	plt.grid()
	

	fig.savefig('batch'+str(batch)+'_' + mapname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

def trail_formation(ind, swarm, targets, timesteps, name):


	# Decode genome into executable behaviour tree
	bt = tg.tree().decode(ind, swarm, targets)
	bt.setup(timeout=15)

	# Setup plot
	dimx = swarm.map.dimensions[0]
	dimy = swarm.map.dimensions[1]
	# Setup plot
	xmin = -dimx/2; xmax = dimx/2
	ymin = -dimy/2; ymax = dimy/2
	plotscale = 10
	fig, ax = plt.subplots( num=None, figsize=(plotscale, plotscale*(dimy/dimx)), dpi=100, facecolor='w', edgecolor='k')	
	ax.set_xlim([xmin,xmax])
	ax.set_ylim([ymin,ymax])
	
	plt.ion()
	plt.grid()
	fig.canvas.draw()
	[ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
		[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=2) for a in range(len(swarm.map.obsticles))]


	dur = 0; score = 0; fitness = 0; t = 0
	fontsize = 12
	found = False
	swarm.beacon_set = []

	ba = []; br = []
	rgb = [0, 0, 1.0]

	color_shifts = [[0,0,1],[1,0,0]]

	shifts = 1
	steps = int((timesteps)/shifts)+ 1

	# # blue = np.flip(np.arange(0, 1, 1/steps))

	# # blue = np.concatenate((blue, np.zeros(steps)))

	# blue = np.ones(steps)

	# blue = np.concatenate((np.ones(steps), np.flip(np.arange(0, 1, 1/steps))))

	# #green = np.concatenate((np.arange(0, 0.5, 0.5/steps), np.flip(np.arange(0,0.5, 0.5/steps))))
	# green = np.zeros(2*steps)

	# red = np.concatenate((np.arange(0, 1, 1/steps), np.ones(steps)))

	green = np.zeros(timesteps)

	red = np.arange(0, 1, 1/(timesteps))

	blue = np.flip(np.arange(0, 1, 1/(timesteps)))

	noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))

	swarm.behaviour = 'random'
	swarm.param = 0.02

	sw = asim.swarm()
	sw.size = 1000
	sw.speed = 0.5
	sw.origin = np.array([-15, -35])
	sw.gen_agents()
	sw.map = swarm.map 

	fitmap = targets.fitness_map(swarm.map, sw, 1)
	targets.fitmap = fitmap

	while t <= timesteps and found == False:
			
		
		#bt.tick()
		swarm.iterate(noise[t-1] )
		swarm.get_state()
		score += targets.get_state(swarm, t, timesteps)	
		
		now = time.time()	
		#ax.clear()
		ax.set_xlim([xmin,xmax])
		ax.set_ylim([ymin,ymax])
		
		x = swarm.agents.T[0]
		y = swarm.agents.T[1]
		ax.plot(x,y,'bo', markersize = 3, color=(red[t-1],green[t-1],blue[t-1]), alpha=0.4)

		t += 1

        # Remove data
		#agents.remove()
		[b[0].remove() for b in ba]
		[b[0].remove() for b in br]
		
	
		sys.stdout.write("Test progress: %.2f%%   \r" % (100*t/timesteps) )
		sys.stdout.flush()

		rgb[0] += 1/(timesteps+1)
		rgb[2] -= 1/(timesteps+1)
	
	alpha = 0.5

	if swarm.beacon_att.size != 0:
		ba = [ax.plot(swarm.beacon_att[a][0],swarm.beacon_att[a][1], 'go', markersize=70, alpha=alpha) for a in range(len(swarm.beacon_att))]
				
	if swarm.beacon_rep.size != 0:

		br = [ax.plot(swarm.beacon_rep[a][0],swarm.beacon_rep[a][1], 'ro', markersize=70, alpha=alpha) for a in range(len(swarm.beacon_rep))]

	# for n in range(0, len(targets.targets)):
	# 	if targets.old_state[n] == False:
	# 		ax.plot(targets.targets[n][0],targets.targets[n][1], 'ro', markersize=10, alpha=0.4)
	# 	else:
	# 		ax.plot(targets.targets[n][0],targets.targets[n][1], 'go', markersize=10, alpha=0.4)
	#ax.text(-20, 45, 'Coverage: %.2f' % targets.coverage, fontsize=fontsize, color='red')

	print('\n\nScore: ', score)
	print('len targets: ', len(targets.targets))
	
	fitness = 0
	fitness = score/len(targets.targets)
	if fitness < 0: fitness = 0
	print ('\n\nCoverage: ', targets.coverage)


	ax.plot(x,y, 'rh', markersize = 6, markeredgecolor="black", alpha = 0.9)
	# Save output figure.

	fig.savefig('outputs/robust/' + name + '_trails', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


def prob_map(ind, swarm, targets, timesteps, name):


	# Track the number of times each target is detected


	# Decode genome into executable behaviour tree
	bt = tg.tree().decode(ind, swarm, targets)
	bt.setup(timeout=15)

	
	dur = 0; score = 0; fitness = 0; t = 0
	found = False
	swarm.beacon_set = []

	pos = np.zeros((len(targets.targets), 2))

	granularity = 2.5

	x = np.arange(-72.5,74.9,granularity)
	y = np.flip(np.arange(-37.5,39.9,granularity))

	pos = np.zeros((len(y),len(x)))

	noise = np.random.uniform(-.1,.1,(20*timesteps, swarm.size, 2))


	swarm.behaviour = 'random'
	swarm.param = 0.01

	covered = False
	total_nodes = len(x)*len(y)
	# timesteps = 100
	trials = 10

	while covered == False and t <= trials*timesteps:

		if t%timesteps == 0 and t != 0:
			print('\nresetting agents')
			swarm.gen_agents()

		swarm.iterate(noise[t-1] )
		swarm.get_state()
		
		# Check intersection of agents with targets
		mag = cdist(targets.targets, swarm.agents)
		dist = mag <= granularity

		# For each target sum the number of detections
		total = np.sum(dist, axis = 1)

		# Add the new detections to an array of the positions
		for n in range(len(targets.targets)):

			row = int((targets.targets[n][1]+39)/granularity)
			col = int((targets.targets[n][0]+74)/granularity)

			if total[n] >= 1:
				pos[row][col] += 1
		
		t += 1

		# Check if all points are visited
		vis = np.count_nonzero(pos)
		#print('visited, ', vis)
		if vis >= 0.9*total_nodes:
			covered = True

		sys.stdout.write("Test progress: %.2f%%   \r" % (100*t/(trials*timesteps)) )
		#sys.stdout.write("Visited: %.2f%%   \r" % (100*(vis/total_nodes)) )
		sys.stdout.flush()

	pos = pos/(trials*timesteps)

	# Count number of zeros
	out = np.reshape(pos, (total_nodes))

	nonzero = np.count_nonzero(out)

	numzeros = total_nodes - nonzero

	print('Proportion of zeros = %.2f' % (numzeros/total_nodes))


	print('Cover time is ', t)

	print('\nmax value is ', np.max(pos))
	print('\nAverage probabality of occupation is ', np.mean(pos))

	plt.imshow(pos, origin='lower')
	plt.colorbar()
	plt.show()
	
	
	
def shadows(ind, swarm, targets, timesteps, filename, shadows=15):

	# Generate array to store old positions
	paths = np.zeros((shadows, swarm.size, 2))

	# Decode genome into executable behaviour tree
	bt = tg.tree().decode(ind, swarm, targets)
	bt.setup(timeout=15)

	# Setup post tick handlers for tree animation
	snapshot_visitor = py_trees.visitors.SnapshotVisitor()
	bt.add_post_tick_handler(functools.partial(post_tick_handler, snapshot_visitor))
	bt.visitors.append(snapshot_visitor)

	dimx = swarm.map.dimensions[0]/2
	dimy = swarm.map.dimensions[1]/2
	# Setup plot
	#lim = 40
	xmin = -dimx; xmax = dimx
	ymin = -dimy; ymax = dimy
	plotscale = 12
	fig, ax = plt.subplots(num=None, figsize=(plotscale, plotscale*(dimy/dimx)), dpi=100, facecolor='w', edgecolor='k')	
	ax.set_xlim([xmin,xmax])
	ax.set_ylim([ymin,ymax])
	
	plt.ion()
	plt.grid(which='both', alpha = 0.5)
	fig.canvas.draw()
	#plt.show()
	[ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
		[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=2) for a in range(len(swarm.map.obsticles))]

	# [ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
	# 	[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=3) for a in range(len(swarm.map.obsticles))]
	# [ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
	# 	[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], '-', lw=1, color = "0.5") for a in range(len(swarm.map.obsticles))]
	
	#ax.set_facecolor((200/255, 196/255, 255/255))
	
	dur = 0; score = 0; fitness = 0; t = 0
	fontsize = 12
	found = False
	swarm.beacon_set = []

	total = 0
	oldscore = 0

	noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))

	field, grid = asim.potentialField_map(swarm.map)
	swarm.field = field
	swarm.grid = grid
	
	sw = asim.swarm()
	sw.size = 1000
	sw.speed = 0.5
	sw.origin = np.array([0, 0])
	sw.gen_agents()
	sw.map = swarm.map 

	fitmap = targets.fitness_map(swarm.map, sw, 15)
	targets.fitmap = fitmap

	# Decode genome into executable behaviour tree
	bt = tg.tree().decode(ind, swarm, targets)
	bt.setup(timeout=15)

	#asim.plot_field(swarm)

	# swarm.behaviour = 'flocking'
	# swarm.param = 50

	while t <= timesteps and found == False:
			
		# if t%25 == 0:
		# 	swarm.gen_agents()
		print('Time: ', t)
		bt.tick()

		#print(py_trees.display.ascii_tree(bt.root))

		now = time.time()
		swarm.iterate(noise[t-1])
		
		swarm.get_state()
		t += 1 

		score += targets.get_state(swarm, t, timesteps)
		print(score)
		
		now = time.time()	
	
		ax.set_xlim([xmin,xmax])
		ax.set_ylim([ymin,ymax])
		
		# Shift out old data				
		for n in range(len(paths)-1):
			paths[n] = paths[n+1]

		# Add in new agent positions.
		paths[len(paths)-1] = swarm.agents

		shadow_pos = []
		size = 1
		for n in range(len(paths)-1):
			x = paths[n].T[0]
			y = paths[n].T[1]

			# a = 1 / math.pow(float(shadows - n ), 1.5)
			a = 1/(shadows-n)
			size = 5 + 1*(1/(shadows - n)) 
			shadow_pos.append(ax.plot(x,y, 'bh', markersize = size, alpha = a))
		shadow_pos.append(ax.plot(x,y, 'rh', markersize = 6, markeredgecolor="black", alpha = 0.9))

		metrics = []
		upper = np.quantile(swarm.agents, 0.75, axis = 0)
		lower = np.quantile(swarm.agents, 0.25, axis = 0)
		median = np.quantile(swarm.agents, 0.5, axis = 0)

		c = list()
		c.append(np.quantile(swarm.agents,0.5,axis=0))

		# for point in c:
		# 	metrics.append(ax.plot(point[0], point[1], 'ro', markersize = swarm.spread*10, alpha = 0.2))
		# 	metrics.append(ax.plot(point[0], point[1], 'gs', markersize = 10, markeredgecolor="black"))
		# metrics.append(ax.plot(upper[0], upper[1], 'g^', markersize = 10, markeredgecolor="black"))
		# metrics.append(ax.plot(lower[0], lower[1], 'gv', markersize = 10, markeredgecolor="black"))
		# metrics.append(ax.plot(median[0], median[1], 'gs', markersize = 10, markeredgecolor="black"))
	
		# Plot stats
		text = []
		fontsize = 14
		# text.append(ax.text(10, 42, 'Swarm behviour: ' + swarm.behaviour + ', ' + str(swarm.param), fontsize=fontsize, color='green'))
		# text.append(ax.text(10, 47, 'Time: %d/%d' %(t, timesteps), fontsize=fontsize, color='purple'))
		# text.append(ax.text(-65, 42, 'Swarm Median Position: %.2f, %.2f' % (swarm.median[0], swarm.median[1]), fontsize=fontsize, color='green'))
		# text.append(ax.text(-65, 47, 'Spread: %.2f' % swarm.spread, fontsize=fontsize, color='red'))
		# text.append(ax.text(-35, 47, 'Coverage: %.2f' % targets.coverage, fontsize=fontsize, color='blue'))
		# text.append(ax.text(40, 47, 'Fitness: %.2f' % (score/len(targets.targets)), fontsize=fontsize, color='red'))


		text.append(ax.text(0, dimy+5, 'Swarm behviour: ' + swarm.behaviour + ', ' + str(swarm.param), fontsize=fontsize, color='green'))
		text.append(ax.text(0, dimy+2, 'Time: %d/%d' %(t, timesteps), fontsize=fontsize, color='purple'))
		#text.append(ax.text(-65, 42, 'Swarm Median Position: %.2f, %.2f' % (swarm.median[0], swarm.median[1]), fontsize=fontsize, color='green'))
		#text.append(ax.text(-65, 47, 'Spread: %.2f' % swarm.spread, fontsize=fontsize, color='red'))
		text.append(ax.text(-30, dimy+2, 'Coverage: %.2f' % targets.coverage, fontsize=fontsize, color='blue'))
		text.append(ax.text(-30, dimy+5, 'Fitness: %.2f' % (score/len(targets.targets)), fontsize=fontsize, color='red'))

		ba = []; br = []
		if swarm.beacon_att.size != 0:
			for a in range(0, len(swarm.beacon_att)):
				ba.append( ax.plot(swarm.beacon_att[a][0],swarm.beacon_att[a][1], 'go', markersize=70, alpha=0.3))
		
					#ax.text(swarm.beacon_set[a].pos[0],swarm.beacon_set[a].pos[1], 'A', fontsize=15, color='green')
		if swarm.beacon_rep.size != 0:
			for a in range(0, len(swarm.beacon_rep)):
				br.append( ax.plot(swarm.beacon_rep[a][0],swarm.beacon_rep[a][1], 'ro', markersize=70, alpha=0.3))
					#ax.text(swarm.beacon_set[a].pos[0],swarm.beacon_set[a].pos[1], 'R', fontsize=15, color='red')

		# targs = []
		# for n in range(0, len(targets.targets)):
		# 	targs, = ax.plot(targets.targets[n][0],targets.targets[n][1], 'ro', markersize=10, alpha=0.3)
			# if targets.old_state[n] == False:
			# 	targs, = ax.plot(targets.targets[n][0],targets.targets[n][1], 'ro', markersize=10, alpha=0.5)
			# else:
			# 	targs, = ax.plot(targets.targets[n][0],targets.targets[n][1], 'go', markersize=10, alpha=0.5)


		room_center = np.array([[-2, -12],[40,-12],[14, -12],
								[3, 20],[3,12],[3,4],[-8,20],[-8,12],[-8,4],
								[-25,20],[-25,12],[-25,4],[-36,20],[-36,12],[-36,4],
								[32.5, 20],[32.5, 6],[-3.5,-32.5],[10, -32.5], [27.5, -32.5],
								[49, -28], [49, -36],[67, -8],[67, -16],[67, -26],[67, -36],
								[-64, -26],[-52, -26],[-38, -26],[-24, -26],
								[-34, -11],[-24, -11],[-50,12],[-21, 32],[47, 5],[47, 27],[47, 16],
								[67, 18],[67, 30],[67, 9],
								[7.5,32],[-61,-11],[15,12],
								[10,0]])

		# shadow_pos.append(ax.plot(room_center.T[0], room_center.T[1], 'rx', markersize = 12))
		

		fig.canvas.draw()

		fig.savefig('frames/'+filename+str(t), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

		print('\n\ntime per loop: %f ms' % (1000*(time.time()-now)))

		for shadow in shadow_pos:
			ax.lines.remove(shadow[0])

		for m in metrics:
			ax.lines.remove(m[0])
		#[a.pop(0).remove() for a in shadow_pos]
		# if swarm.beacon_rep.size != 0:
		# 	br.remove()
		# if swarm.beacon_att.size != 0:
		# 	ba.remove()
		for b in ba:
			ax.lines.remove(b[0])
		for b in br:
			ax.lines.remove(b[0])
		# br.remove()
		
		# ba.remove()
		#targs.remove()
		[t.remove() for t in text]

	command = 'ffmpeg -i frames/'+filename+'%01d.png -vf scale=1000:1000 -filter:v fps=30 ' + 'outputs/robust/' + filename + '_shadowstest.mp4'

	process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	
	
	

def post_tick_handler(snapshot_visitor, behaviour_tree):
    print(
        py_trees.display.unicode_tree(
            behaviour_tree.root,
            visited=snapshot_visitor.visited,
            previously_visited=snapshot_visitor.visited
        )
    )


def default(ind, swarm, targets, timesteps):
		
	# define the name of the directory to be created
	path = "frames"

	try:
		os.mkdir(path)
	except OSError:
		print ("Creation of the directory %s failed" % path)
	else:
		print ("Successfully created the directory %s " % path)

	# Decode genome into executable behaviour tree
	bt = tg.tree().decode(ind, swarm, targets)
	bt.setup(timeout=15)

	# Setup post tick handlers for tree animation
	snapshot_visitor = py_trees.visitors.SnapshotVisitor()
	bt.add_post_tick_handler(functools.partial(post_tick_handler, snapshot_visitor))
	bt.visitors.append(snapshot_visitor)

	dimx = swarm.map.dimensions[0]
	dimy = swarm.map.dimensions[1]
	# Setup plot
	#lim = 40
	xmin = -dimx/2; xmax = dimx/2
	ymin = -dimy/2; ymax = dimy/2
	plotscale = 12
	fig, ax = plt.subplots( num=None, figsize=(plotscale, plotscale*(dimy/dimx)), dpi=100, facecolor='w', edgecolor='k')	
	ax.set_xlim([xmin,xmax])
	ax.set_ylim([ymin,ymax])
	
	plt.ion()
	plt.grid()
	fig.canvas.draw()
	plt.show()
	[ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
		[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=2) for a in range(len(swarm.map.obsticles))]

	
	dur = 0; score = 0; fitness = 0; t = 0
	fontsize = 12
	found = False
	swarm.beacon_set = []

	total = 0
	oldscore = 0

	noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))

	while t <= timesteps and found == False:
			
		t += 1
		bt.tick()
		swarm.iterate(noise[t-1])
		swarm.get_state()
		score = targets.get_state(swarm, t, timesteps)
		total += (score - oldscore)*((-t/(timesteps*2))+1)

		oldscore = score

		now = time.time()	
		#ax.clear()
		ax.set_xlim([xmin,xmax])
		ax.set_ylim([ymin,ymax])
		
		x = swarm.agents.T[0]
		y = swarm.agents.T[1]
		agents, = ax.plot(x,y,'bo',markersize = 3)# targs= []
		# for n in range(0, len(targets.targets)):
		# 	if targets.old_state[n] == False:
		# 		targs, = ax.plot(targets.targets[n][0],targets.targets[n][1], 'ro', markersize=10, alpha=0.5)
		# 	else:
		# 		targs, = ax.plot(targets.targets[n][0],targets.targets[n][1], 'go', markersize=10, alpha=0.5)
	
		# Plot stats
		text = []
		text.append(ax.text(5, 41, 'Swarm behviour: ' + swarm.behaviour + ', ' + str(swarm.param), fontsize=fontsize, color='green'))
		text.append(ax.text(5, 45, 'Time: %d/%d' %(t, timesteps), fontsize=fontsize, color='purple'))
		text.append(ax.text(-40, 41, 'Center of Mass: %.2f, %.2f' % (swarm.centermass[0], swarm.centermass[1]), fontsize=fontsize, color='green'))
		text.append(ax.text(-40, 45, 'Spread: %.2f' % swarm.spread, fontsize=fontsize, color='red'))
		text.append(ax.text(-20, 45, 'Coverage: %.2f' % targets.coverage, fontsize=fontsize, color='blue'))

		text.append(ax.text(swarm.agents[0][0], swarm.agents[0][1], 'Agent 1', fontsize=fontsize, color='blue'))

		# ba = []; br = []
		# if swarm.beacon_att.size != 0:
		# 	for a in range(0, len(swarm.beacon_att)):
		# 		ba, = ax.plot(swarm.beacon_att[a][0],swarm.beacon_att[a][1], 'go', markersize=70, alpha=0.3)
		
		# 			#ax.text(swarm.beacon_set[a].pos[0],swarm.beacon_set[a].pos[1], 'A', fontsize=15, color='green')
		# if swarm.beacon_rep.size != 0:
		# 	for a in range(0, len(swarm.beacon_rep)):
		# 		br, = ax.plot(swarm.beacon_rep[a][0],swarm.beacon_rep[a][1], 'ro', markersize=70, alpha=0.3)
		# 			#ax.text(swarm.beacon_set[a].pos[0],swarm.beacon_set[a].pos[1], 'R', fontsize=15, color='red')

		# targs= []
		# for n in range(0, len(targets.targets)):
		# 	if targets.old_state[n] == False:
		# 		targs, = ax.plot(targets.targets[n][0],targets.targets[n][1], 'ro', markersize=10, alpha=0.5)
		# 	else:
		# 		targs, = ax.plot(targets.targets[n][0],targets.targets[n][1], 'go', markersize=10, alpha=0.5)

		fig.canvas.draw()

		print('\n\ntime per loop: %f ms' % (1000*(time.time()-now)))

		# Remove data
		agents.remove()
		# ba.remove()
		# br.remove()
		#targs.remove()
		[t.remove() for t in text]
	
	print('\n\nScore: ', score)
	print('len targets: ', len(targets.targets))
	maxsize = 300
	fitness = 0
	fitness = score/len(targets.targets)
	print('fitness pre cost: ', fitness)
	fitness = fitness - (len(ind.genome)*0.001)
	if fitness < 0: fitness = 0
	print ('Individual fitness: ', fitness)
	
	return fitness


def isData():

        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def read_input(data, swarm):

	if data == 'w':
		swarm.behaviour = 'north'
	if data == 'x':
		swarm.behaviour = 'south'
	if data == 'a':
		swarm.behaviour = 'west'
	if data == 'd':
		swarm.behaviour = 'east'
	if data == 'q':
		swarm.behaviour = 'northwest'
	if data == 'e':
		swarm.behaviour = 'northeast'
	if data == 'z':
		swarm.behaviour = 'southwest'
	if data == 'c':
		swarm.behaviour = 'southeast'
	if data == 's':
		swarm.behaviour = 'disperse'
	if data == 'o':
		swarm.behaviour = 'rot_clock'
	if data == 'p':
		swarm.behaviour = 'rot_anti'

	if data == '1':
		swarm.param = 1
	if data == '2':
		swarm.param = 30
	if data == '3':
		swarm.param = 60

	if data == '8':
		swarm.param = 0.01
	if data == '9':
		swarm.param = 0.05
	if data == '0':
		swarm.param = 0.09     

def game(ind, swarm, targets, timesteps):
		
	# define the name of the directory to be created
	path = "frames"

	try:
		os.mkdir(path)
	except OSError:
		print ("Creation of the directory %s failed" % path)
	else:
		print ("Successfully created the directory %s " % path)


	dimx = swarm.map.dimensions[0]
	dimy = swarm.map.dimensions[1]
	# Setup plot
	#lim = 40
	xmin = -dimx/2; xmax = dimx/2
	ymin = -dimy/2; ymax = dimy/2
	plotscale = 12
	fig, ax = plt.subplots( num=None, figsize=(plotscale, plotscale*(dimy/dimx)), dpi=100, facecolor='w', edgecolor='k')	
	ax.set_xlim([xmin,xmax])
	ax.set_ylim([ymin,ymax])
	
	plt.ion()
	plt.grid()
	fig.canvas.draw()
	plt.show()
	[ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
		[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=2) for a in range(len(swarm.map.obsticles))]

	
	dur = 0; score = 0; fitness = 0; t = 0
	fontsize = 12
	found = False
	swarm.beacon_set = []

	total = 0
	oldscore = 0

	while t <= timesteps and found == False:


		# Read user input
		if isData():
			c = sys.stdin.read(1)
			read_input(c, swarm)
			# print(c)
			if c == '\x1b':         # x1b is ESC
				break
			
		t += 1
		#bt.tick()
		swarm.iterate()
		swarm.get_state()
		score = targets.get_state(swarm, t)
		total += (score - oldscore)*((-t/(timesteps*2))+1)

		oldscore = score

		now = time.time()	
		#ax.clear()
		ax.set_xlim([xmin,xmax])
		ax.set_ylim([ymin,ymax])
		
		x = swarm.agents.T[0]
		y = swarm.agents.T[1]
		#agents, = ax.plot(x,y,'bo',markersize = 3)

		center, = ax.plot(swarm.centermass[0], swarm.centermass[1],'rx',markersize = 5)

		median, = ax.plot(np.median(x), np.median(y),'kx',markersize = 5)

		bubble, = ax.plot(swarm.centermass[0], swarm.centermass[1],'go', alpha = 0.2, markersize = 10*swarm.spread)

		

		agents, = ax.plot(swarm.agents.T[0], swarm.agents.T[1], 'bo', alpha = 0.5, markersize = 4)
	
		# Plot stats
		text = []
		text.append(ax.text(5, 41, 'Swarm behviour: ' + swarm.behaviour + ', ' + str(swarm.param), fontsize=fontsize, color='green'))
		text.append(ax.text(15, 45, 'Time: %d/%d' %(t, timesteps), fontsize=fontsize, color='purple'))
		text.append(ax.text(-40, 41, 'Center of Mass: %.2f, %.2f' % (swarm.centermass[0], swarm.centermass[1]), fontsize=fontsize, color='green'))
		text.append(ax.text(-40, 45, 'Spread: %.2f' % swarm.spread, fontsize=fontsize, color='red'))
		text.append(ax.text(-20, 45, 'Coverage: %.2f' % targets.coverage, fontsize=fontsize, color='blue'))

		# ba = []; br = []
		# if swarm.beacon_att.size != 0:
		# 	for a in range(0, len(swarm.beacon_att)):
		# 		ba, = ax.plot(swarm.beacon_att[a][0],swarm.beacon_att[a][1], 'go', markersize=70, alpha=0.3)
		
		# 			#ax.text(swarm.beacon_set[a].pos[0],swarm.beacon_set[a].pos[1], 'A', fontsize=15, color='green')
		# if swarm.beacon_rep.size != 0:
		# 	for a in range(0, len(swarm.beacon_rep)):
		# 		br, = ax.plot(swarm.beacon_rep[a][0],swarm.beacon_rep[a][1], 'ro', markersize=70, alpha=0.3)
		# 			#ax.text(swarm.beacon_set[a].pos[0],swarm.beacon_set[a].pos[1], 'R', fontsize=15, color='red')

		# targs= []
		# for n in range(0, len(targets.targets)):
		# 	if targets.old_state[n] == False:
		# 		targs, = ax.plot(targets.targets[n][0],targets.targets[n][1], 'ro', markersize=10, alpha=0.5)
		# 	else:
		# 		targs, = ax.plot(targets.targets[n][0],targets.targets[n][1], 'go', markersize=10, alpha=0.5)

		fig.canvas.draw()

		fig.savefig('frames/image'+str(t), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

        # Remove data
		agents.remove()
		center.remove()
		median.remove()
		bubble.remove()
		
		# [b[0].remove() for b in ba]
		# [b[0].remove() for b in br]
		
		[t.remove() for t in text]

		sys.stdout.write("Test progress: %.2f%%   \r" % (100*t/timesteps) )
		sys.stdout.flush()
	
	print('\n\nScore: ', score)
	print('len targets: ', len(targets.targets))
	maxsize = 300
	fitness = 0
	fitness = score/len(targets.targets)
	print('fitness pre cost: ', fitness)
	fitness = fitness - (len(ind.genome)*0.001)
	if fitness < 0: fitness = 0
	print ('Individual fitness: ', fitness)
	
	return fitness

def record(ind, swarm, targets, timesteps):
		
	# define the name of the directory to be created
	path = "frames"

	try:
		os.mkdir(path)
	except OSError:
		print ("Creation of the directory %s failed" % path)
	else:
		print ("Successfully created the directory %s " % path)

	# Decode genome into executable behaviour tree
	bt = tg.tree().decode(ind, swarm, targets)
	bt.setup(timeout=15)

	# # Setup post tick handlers for tree animation
	# snapshot_visitor = py_trees.visitors.SnapshotVisitor()
	# bt.add_post_tick_handler(functools.partial(post_tick_handler, snapshot_visitor))
	# bt.visitors.append(snapshot_visitor)

	# Setup plot
	dimx = swarm.map.dimensions[0]
	dimy = swarm.map.dimensions[1]
	# Setup plot
	#lim = 40
	xmin = -dimx/2; xmax = dimx/2
	ymin = -dimy/2; ymax = dimy/2
	plotscale = 12
	fig, ax = plt.subplots( num=None, figsize=(plotscale, plotscale*(dimy/dimx)), dpi=100, facecolor='w', edgecolor='k')	
	ax.set_xlim([xmin,xmax])
	ax.set_ylim([ymin,ymax])
	
	
	plt.ion()
	plt.grid()
	fig.canvas.draw()
	#plt.show()
	[ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
		[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=2) for a in range(len(swarm.map.obsticles))]


	dur = 0; score = 0; fitness = 0; t = 0
	fontsize = 12
	found = False
	swarm.beacon_set = []

	noise = np.random.uniform(-.1,.1,(timesteps, swarm.size, 2))

	ba = []; br = []
	while t <= timesteps and found == False:
			
		t += 1
		bt.tick()
		# swarm.behaviour = 'south'
		# swarm.param = 60
		swarm.iterate(noise[t-1])
		swarm.get_state()
		score = targets.get_state(swarm, t, timesteps)	

		now = time.time()	
		#ax.clear()
		ax.set_xlim([xmin,xmax])
		ax.set_ylim([ymin,ymax])
		
		x = swarm.agents.T[0]
		y = swarm.agents.T[1]
		agents, = ax.plot(x,y,'bo',markersize = 3)

		# Plot stats
		text = []
		text.append(ax.text(5, 41, 'Swarm behviour: ' + swarm.behaviour + ', ' + str(swarm.param), fontsize=fontsize, color='green'))
		text.append(ax.text(5, 45, 'Time: %d/%d' %(t, timesteps), fontsize=fontsize, color='purple'))
		text.append(ax.text(-40, 41, 'Center of Mass: %.2f, %.2f' % (swarm.centermass[0], swarm.centermass[1]), fontsize=fontsize, color='green'))
		text.append(ax.text(-40, 45, 'Spread: %.2f' % swarm.spread, fontsize=fontsize, color='red'))
		text.append(ax.text(-20, 45, 'Coverage: %.2f' % targets.coverage, fontsize=fontsize, color='blue'))

		#[ax.plot(swarm.beacon_set[a].pos[0],swarm.beacon_set[a].pos[1], 'ro', markersize=70, alpha=0.3) for a in range(len(swarm.beacon_set))]
		if swarm.beacon_att.size != 0:
			# for a in range(0, len(swarm.beacon_att)):
			# 	ba, = ax.plot(swarm.beacon_att[a][0],swarm.beacon_att[a][1], 'go', markersize=70, alpha=0.3)
			ba = [ax.plot(swarm.beacon_att[a][0],swarm.beacon_att[a][1], 'go', markersize=70, alpha=0.3) for a in range(len(swarm.beacon_att))]
					#ax.text(swarm.beacon_set[a].pos[0],swarm.beacon_set[a].pos[1], 'A', fontsize=15, color='green')
		if swarm.beacon_rep.size != 0:
			#for a in range(0, len(swarm.beacon_rep)):
			#	br, = ax.plot(swarm.beacon_rep[a][0],swarm.beacon_rep[a][1], 'ro', markersize=70, alpha=0.3)
					#ax.text(swarm.beacon_set[a].pos[0],swarm.beacon_set[a].pos[1], 'R', fontsize=15, color='red')

			br = [ax.plot(swarm.beacon_rep[a][0],swarm.beacon_rep[a][1], 'ro', markersize=70, alpha=0.3) for a in range(len(swarm.beacon_rep))]

		# for n in range(0, len(targets.targets)):
		# 	if targets.old_state[n] == False:
		# 		ax.plot(targets.targets[n][0],targets.targets[n][1], 'ro', markersize=10, alpha=0.5)
		# 	else:
		# 		ax.plot(targets.targets[n][0],targets.targets[n][1], 'go', markersize=10, alpha=0.5)

		

		


		fig.savefig('frames/image'+str(t), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

        # Remove data
		agents.remove()
		[b[0].remove() for b in ba]
		[b[0].remove() for b in br]
		
		[t.remove() for t in text]

		sys.stdout.write("Test progress: %.2f%%   \r" % (100*t/timesteps) )
		sys.stdout.flush()
	
	print('\n\nScore: ', score)
	print('len targets: ', len(targets.targets))
	maxsize = 300
	fitness = 0
	fitness = score/len(targets.targets)
	print('fitness pre cost: ', fitness)
	fitness = fitness - (len(ind.genome)*0.001)
	if fitness < 0: fitness = 0
	print ('Individual fitness: ', fitness)
	
	return fitness



def default_ad(inda, indb, swarma, swarmb, targets, timesteps):
		
	# Decode genome into executable behaviour tree
	bta = tg.tree().decode(inda, swarma, targets)
	bta.setup(timeout=15)

	btb = tg.tree().decode(indb, swarmb, targets)
	btb.setup(timeout=15)

	# Setup plot
	lim = 40
	xmin = -lim; xmax = lim
	ymin = -lim; ymax = lim
	fig, ax = plt.subplots(facecolor=(.99, .99, .99))	
	ax.set_xlim([xmin,xmax])
	ax.set_ylim([ymin,ymax])
	plt.ion()
	plt.grid()
	fig.canvas.draw()
	plt.show()

	dur = 0; scorea = 0; scoreb = 0; fitness = 0; t = 0
	fontsize = 12
	found = False
	
	while t <= timesteps and found == False:
		t += 1
		# input()
		bta.tick()
		btb.tick()
		swarma.iterate()
		swarma.get_state()
		swarmb.iterate()
		swarmb.get_state()

		scorea += targets.ad_state(swarma, t)
		scoreb += targets.ad_state(swarmb, t)		

		ax.clear()
		ax.set_xlim([xmin,xmax])
		ax.set_ylim([ymin,ymax])
		plt.show()
		plt.grid()

		[ax.plot(swarma.agents[a][0],swarma.agents[a][1], 'bo') for a in range(swarma.size)]
		[ax.plot(swarmb.agents[a][0],swarmb.agents[a][1], 'ro') for a in range(swarmb.size)]
		[ax.plot([swarma.map.obsticles[a].start[0], swarma.map.obsticles[a].end[0]], [swarma.map.obsticles[a].start[1], swarma.map.obsticles[a].end[1]], 'k-', lw=2) for a in range(len(swarma.map.obsticles))]
		
		ax.text(5, 41, 'Swarm behviour A: ' + swarma.behaviour + ', ' + str(swarma.param), fontsize=fontsize, color='green')
		ax.text(5, 45, 'Swarm behviour B: ' + swarmb.behaviour + ', ' + str(swarmb.param), fontsize=fontsize, color='green')
		#ax.text(-40, 41, 'Center of Mass: %.2f, %.2f' % (swarm.centermass[0], swarm.centermass[1]), fontsize=fontsize, color='green')
		#ax.text(-40, 45, 'Spread: %.2f' % swarm.spread, fontsize=fontsize, color='red')
		ax.text(-20, 45, 'Coverage: %.2f' % targets.coverage, fontsize=fontsize, color='blue')
		
		for n in range(0, len(targets.targets)):
			if targets.old_state[n] == False:
				ax.plot(targets.targets[n][0],targets.targets[n][1], 'ro', markersize=10, alpha=0.5)
			else:
				ax.plot(targets.targets[n][0],targets.targets[n][1], 'go', markersize=10, alpha=0.5)

		fig.canvas.draw()
		print('Time: ', t, '/',timesteps,  end='\r')

	
	print('\n\nScore A: ', scorea)
	print('\nScore B: ', scoreb)
	print('len targets: ', len(targets.targets))
	maxsize = 300
	fitness = 0
	fitness = score/len(targets.targets)
	print('fitness pre cost: ', fitness)
	fitness = fitness - (len(ind.genome)*0.001)
	if fitness < 0: fitness = 0
	print ('Individual fitness: ', fitness)
	input()
	return fitness



# Reworked py-trees code to enable animated dot graphs of solutions

def generate_pydot_graph(root, visibility_level, collapse_decorators=False):

	def get_node_attributes(node, visibility_level):
		blackbox_font_colours = {py_trees.common.BlackBoxLevel.DETAIL: "dodgerblue",
								py_trees.common.BlackBoxLevel.COMPONENT: "lawngreen",
								py_trees.common.BlackBoxLevel.BIG_PICTURE: "white"
								}

		coldict = {py_trees.Status.SUCCESS: 'green', py_trees.Status.FAILURE: 'red', py_trees.Status.INVALID: 'white' , py_trees.Status.RUNNING: 'white' }

		# if isinstance(node, py_trees.composites.Chooser):
		# 	attributes = ('doubleoctagon', col, 'black')  # octagon
		if isinstance(node, py_trees.composites.Selector):
			attributes = ('octagon', coldict[node.status], 'black')  # octagon
		elif isinstance(node, py_trees.composites.Sequence):
			attributes = ('box', coldict[node.status], 'black')
		# elif isinstance(node, py_trees.composites.Parallel):
		# 	attributes = ('parallelogram', col, 'black')
		# elif isinstance(node, py_trees.decorators.Decorator):
		# 	attributes = ('ellipse', 'ghostwhite', 'black')
		else:
			attributes = ('ellipse', coldict[node.status], 'black')
		if node.blackbox_level != py_trees.common.BlackBoxLevel.NOT_A_BLACKBOX:
			attributes = (attributes[0], coldict[node.status], blackbox_font_colours[node.blackbox_level])
		return attributes

	fontsize = 11
	graph = pydot.Dot(graph_type='digraph')
	graph.set_name(root.name.lower().replace(" ", "_"))
	# fonts: helvetica, times-bold, arial (times-roman is the default, but this helps some viewers, like kgraphviewer)
	graph.set_graph_defaults(fontname='times-roman')
	graph.set_node_defaults(fontname='times-roman')
	graph.set_edge_defaults(fontname='times-roman')
	(node_shape, node_colour, node_font_colour) = get_node_attributes(root, visibility_level)
	node_root = pydot.Node(root.name, shape=node_shape, style="filled", fillcolor=node_colour, fontsize=fontsize, fontcolor=node_font_colour)
	graph.add_node(node_root)
	names = [root.name]

	def add_edges(root, root_dot_name, visibility_level, collapse_decorators):
		# if isinstance(root, py_trees.decorators.Decorator) and collapse_decorators:
		# 	return
		if visibility_level < root.blackbox_level:
			for c in root.children:
				(node_shape, node_colour, node_font_colour) = get_node_attributes(c, visibility_level)
				proposed_dot_name = c.name
				while proposed_dot_name in names:
					proposed_dot_name = proposed_dot_name + "*"
				names.append(proposed_dot_name)
				node = pydot.Node(proposed_dot_name, shape=node_shape, style="filled", fillcolor=node_colour, fontsize=fontsize, fontcolor=node_font_colour)
				graph.add_node(node)
				edge = pydot.Edge(root_dot_name, proposed_dot_name)
				graph.add_edge(edge)
				if c.children != []:
					add_edges(c, proposed_dot_name, visibility_level, collapse_decorators)

	add_edges(root, root.name, visibility_level, collapse_decorators)
	return graph

def stringify_dot_tree(root):
	
	graph = generate_pydot_graph(root, visibility_level=common.VisibilityLevel.DETAIL)
	return graph.to_string()

def render_dot_tree(root, visibility_level=py_trees.common.VisibilityLevel.DETAIL, collapse_decorators=False, name=None):

	graph = generate_pydot_graph(root, visibility_level, collapse_decorators)
	filename_wo_extension = root.name.lower().replace(" ", "_") if name is None else name
	#print("Writing %s.dot/svg/png" % filename_wo_extension)
	#graph.write('treeanim' + '.dot')
	# graph.write_png(filename_wo_extension + '.png')
	graph.write_svg(filename_wo_extension + '.svg')