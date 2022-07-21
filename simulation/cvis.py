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
import matplotlib
import matplotlib.pyplot 
import numpy as np

import behtree.treegen as tg
import simulation.csim as csim
import simulation.asim as asim

from matplotlib.animation import FuncAnimation


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
 
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Evaluation')
    ax.set_ylabel('Trained')

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["white", "white"],
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



def benchmark(swarm, grid, timesteps, filename, shadows=15):

	
	dur = 0; score = 0; fitness = 0; t = 0
	fontsize = 12
	found = False
	swarm.beacon_set = []

	total = 0
	oldscore = 0

	swarm.behaviour = 'random'
	swarm.param = 0.01

	# Range of tests to perform
	swarm_size = np.arange(50, 201, 50)
	grid_weight = np.arange(0.51, 0.71, 0.01)

	map_data = np.zeros((len(swarm_size), len(grid_weight)))

	timesteps = 1000

	with open('outputs/aamas_environments/batch19/map1_50agents3_hallfame', 'rb') as input:
		hall = pickle.load(input)
	
	ind = hall[0].copy()

	for n in range(len(swarm_size)):

		for i in range(len(grid_weight)):

			trials = 25
			data = np.zeros(trials)
			for k in range(trials):
				# Reset sim params
				swarm = csim.cswarm()
				swarm.size = swarm_size[n]
				swarm.speed = 0.5
				swarm.origin = np.array([0, 0])
				swarm.gen_agents()

				grid = csim.gridset()
				grid.distribution = grid_weight[i]
				grid.gen()

				env = asim.map()
				env.empty()
				env.gen()
				swarm.map = env

				targets = asim.target_set()
				targets.set_state('set1')
				targets.radius = 5
				targets.reset()

				bt = tg.tree().decode(ind, swarm, targets)
				
				# Generate array to store old positions
				
				
				dur = 0; score = 0; fitness = 0; t = 0
				fontsize = 12
				found = False
				swarm.beacon_set = []

				total = 0
				oldscore = 0

				swarm.behaviour = 'random'
				swarm.param = 0.01

				while t <= timesteps and found == False:
						
					t += 1
					#bt.tick()
					swarm.iterate()
					swarm.check_grid(grid)
					swarm.get_state()

					

				# Get the proportion of black opinions at finish
				data[k] = np.sum(swarm.opinions)/swarm.size

			map_data[n][i] = np.mean(data)

			print('\n At point: size = %d, dist = %.2f' %(swarm_size[n], grid_weight[i]))

	size_labels = [str(x) for x in swarm_size]
	weight_labels = [str(np.around(x, 2)) for x in grid_weight]

	fig, ax = plt.subplots()

	im, cbar = heatmap(map_data, size_labels, weight_labels, ax=ax,
	                   cmap="plasma", cbarlabel="Solution Fitness")
	texts = annotate_heatmap(im, valfmt="{x:.2f}")


	fig.tight_layout()
	plt.show()


		


def shadows(swarm, grid, timesteps, filename, shadows=15):

	# Generate array to store old positions
	paths = np.zeros((shadows, swarm.size, 2))

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
	

	# [ax.plot(grid.grid_pos[a][0], grid.grid_pos[a][1], 'gs', alpha = 0.3, markersize = 30) for a in range(len(grid.grid_pos))]

	squaresize = 30
	[ax.plot(grid.grid_black[a][0], grid.grid_black[a][1], 'ks', alpha = 0.3, markersize = squaresize) for a in range(len(grid.grid_black))]
	[ax.plot(grid.grid_white[a][0], grid.grid_white[a][1], 'bs', alpha = 0.3, markersize = squaresize) for a in range(len(grid.grid_white))]

	[ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
		[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=4) for a in range(len(swarm.map.obsticles))]
	[ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
		[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], '-', lw=1.5, color = "0.5") for a in range(len(swarm.map.obsticles))]
	
	dur = 0; score = 0; fitness = 0; t = 0
	fontsize = 12
	found = False
	swarm.beacon_set = []

	total = 0
	oldscore = 0

	swarm.behaviour = 'random'
	swarm.param = 0.01

	while t <= timesteps and found == False:
			
		t += 1
		#bt.tick()
		swarm.iterate()
		swarm.check_grid(grid)
		swarm.get_state()

		print('Exploring agents: ', swarm.exploring)
		print('\nAgent qualities: ', swarm.quality)
		print('\nAgent Opinions: ', swarm.opinions)
		print('\nExplore counter: ', swarm.explore_counter)

		ax.set_xlim([xmin,xmax])
		ax.set_ylim([ymin,ymax])
		
		# Shift out old data				
		for n in range(len(paths)-1):
			paths[n] = paths[n+1]

		# Add in new agent positions.
		paths[len(paths)-1] = swarm.agents

		shadow_pos = []
		size = 2
		for n in range(len(paths)-1):
			x = paths[n].T[0]
			y = paths[n].T[1]

			# a = 1 / math.pow(float(shadows - n ), 1.5)
			a = 1/(shadows-n)
			size = 5 + 1*(1/(shadows - n)) 
			shadow_pos.append(ax.plot(x,y, 'bh', markersize = size, alpha = a))
		

		for a in range(swarm.size):

			if swarm.opinions[a] == 1:
				shadow_pos.append(ax.plot(swarm.agents[a][0],swarm.agents[a][1], 'rh', markersize = 8, markeredgecolor="black"))
			else:
				shadow_pos.append(ax.plot(swarm.agents[a][0],swarm.agents[a][1], 'wh', markersize = 8, markeredgecolor="black"))

		metrics = []
		

		text = []
		text.append(ax.text(5, 41, 'Swarm behviour: ' + swarm.behaviour + ', ' + str(swarm.param), fontsize=fontsize, color='green'))
		text.append(ax.text(5, 45, 'Time: %d/%d' %(t, timesteps), fontsize=fontsize, color='purple'))

		text.append(ax.text(-30, 45, 'White opinions: %.1f Percent' %((swarm.size - np.sum(swarm.opinions))/swarm.size*100), fontsize=fontsize, color='purple'))
		text.append(ax.text(-30, 41, 'Black opinions: %.1f Percent' %(np.sum(swarm.opinions)/swarm.size*100), fontsize=fontsize, color='purple'))


		fig.canvas.draw()

		fig.savefig('frames/image'+str(t), dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

		for shadow in shadow_pos:
			ax.lines.remove(shadow[0])

		for m in metrics:
			ax.lines.remove(m[0])
		
		[t.remove() for t in text]

	command = 'ffmpeg -i frames/image%01d.png -vf scale=1000:1000 -filter:v fps=30 ' + filename + '_shadows.mp4'

	process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()




def supervisor(ind, swarm, grid, timesteps, filename, shadows=15):

	# Generate array to store old positions
	paths = np.zeros((shadows, swarm.size, 2))

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
	

	# [ax.plot(grid.grid_pos[a][0], grid.grid_pos[a][1], 'gs', alpha = 0.3, markersize = 30) for a in range(len(grid.grid_pos))]

	squaresize = 30
	[ax.plot(grid.grid_black[a][0], grid.grid_black[a][1], 'ks', alpha = 0.3, markersize = squaresize) for a in range(len(grid.grid_black))]
	[ax.plot(grid.grid_white[a][0], grid.grid_white[a][1], 'bs', alpha = 0.3, markersize = squaresize) for a in range(len(grid.grid_white))]

	[ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
		[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], 'k-', lw=4) for a in range(len(swarm.map.obsticles))]
	[ax.plot([swarm.map.obsticles[a].start[0], swarm.map.obsticles[a].end[0]], 
		[swarm.map.obsticles[a].start[1], swarm.map.obsticles[a].end[1]], '-', lw=1.5, color = "0.5") for a in range(len(swarm.map.obsticles))]
	
	dur = 0; score = 0; fitness = 0; t = 0
	fontsize = 12
	found = False
	swarm.beacon_set = []

	total = 0
	oldscore = 0

	targets = asim.target_set()
	targets.radius = 5
	targets.set_state('uniform')
	targets.reset()

	fitness = 0
	t = 0
	found = False
	# IMPORTANT! need to reset behaviours after each run 
	swarm.beacon_set = []
	bt = tg.tree().decode(ind, swarm, targets)

	while t <= timesteps and found == False:
			
		t += 1
		bt.tick()
		swarm.iterate()
		swarm.check_grid(grid)
		swarm.get_state()

		print('Exploring agents: ', swarm.exploring)
		print('\nAgent qualities: ', swarm.quality)
		print('\nAgent Opinions: ', swarm.opinions)
		print('\nExplore counter: ', swarm.explore_counter)

		ax.set_xlim([xmin,xmax])
		ax.set_ylim([ymin,ymax])

		ba = []; br = []
		if swarm.beacon_att.size != 0:
			for a in range(0, len(swarm.beacon_att)):
				ba.append( ax.plot(swarm.beacon_att[a][0],swarm.beacon_att[a][1], 'go', markersize=70, alpha=0.3))
		
					#ax.text(swarm.beacon_set[a].pos[0],swarm.beacon_set[a].pos[1], 'A', fontsize=15, color='green')
		if swarm.beacon_rep.size != 0:
			for a in range(0, len(swarm.beacon_rep)):
				br.append( ax.plot(swarm.beacon_rep[a][0],swarm.beacon_rep[a][1], 'ro', markersize=70, alpha=0.3))
					#ax.text(swarm.beacon_set[a].pos[0],swarm.beacon_set[a].pos[1], 'R', fontsize=15, color='red')
		
		# Shift out old data				
		for n in range(len(paths)-1):
			paths[n] = paths[n+1]

		# Add in new agent positions.
		paths[len(paths)-1] = swarm.agents

		shadow_pos = []
		size = 2
		for n in range(len(paths)-1):
			x = paths[n].T[0]
			y = paths[n].T[1]

			# a = 1 / math.pow(float(shadows - n ), 1.5)
			a = 1/(shadows-n)
			size = 5 + 1*(1/(shadows - n)) 
			shadow_pos.append(ax.plot(x,y, 'bh', markersize = size, alpha = a))
		

		for a in range(swarm.size):

			if swarm.opinions[a] == 1:
				shadow_pos.append(ax.plot(swarm.agents[a][0],swarm.agents[a][1], 'rh', markersize = 8, markeredgecolor="black"))
			else:
				shadow_pos.append(ax.plot(swarm.agents[a][0],swarm.agents[a][1], 'wh', markersize = 8, markeredgecolor="black"))

		metrics = []
		

		text = []
		text.append(ax.text(5, 41, 'Swarm behviour: ' + swarm.behaviour + ', ' + str(swarm.param), fontsize=fontsize, color='green'))
		text.append(ax.text(5, 45, 'Time: %d/%d' %(t, timesteps), fontsize=fontsize, color='purple'))

		text.append(ax.text(-30, 45, 'White opinions: %.1f Percent' %((swarm.size - np.sum(swarm.opinions))/swarm.size*100), fontsize=fontsize, color='purple'))
		text.append(ax.text(-30, 41, 'Black opinions: %.1f Percent' %(np.sum(swarm.opinions)/swarm.size*100), fontsize=fontsize, color='purple'))

		

		fig.canvas.draw()

		# fig.savefig('frames/image'+str(t), dpi=None, facecolor='w', edgecolor='w',
  #       orientation='portrait', papertype=None, format=None,
  #       transparent=False, bbox_inches=None, pad_inches=0.1,
  #       frameon=None, metadata=None)

		for b in ba:
			ax.lines.remove(b[0])
		for b in br:
			ax.lines.remove(b[0])

		for shadow in shadow_pos:
			ax.lines.remove(shadow[0])

		for m in metrics:
			ax.lines.remove(m[0])
		
		[t.remove() for t in text]

	command = 'ffmpeg -i frames/image%01d.png -vf scale=1000:1000 -filter:v fps=30 ' + filename + '_shadows.mp4'

	process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()