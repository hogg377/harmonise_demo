
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


def sbend_gen(start_point, direction):

	# Generate an S bend corridor polygon

	lengthA = random.randint(10,15)
	lengthB = random.randint(10,15)
	lengthC = random.randint(10,15)
	
	




	return poly

def grow_corridor(dimx, dimy, cor_width, turn_rate, max_length, 
							min_seg_len, max_seg_len, subtree_prob, grow_steps, beacons, rules, cor_proportion):


	# Set corridor direction
	cor_dir = [0,0]

	poly_list = []

	edge_buffer = cor_width

	# Define bounds of the environment
	bottom = -dimy/2
	top = dimy/2
	left = -dimx/2
	right = dimx/2

	poly_bound = Polygon([(left, bottom), (left, top), (right, top), (right, bottom)])

	# Pick random growth point on perimeter

	if np.random.choice([True, False]) == True:

		# Start on horizontal border
		y = np.random.choice([-dimy/2, dimy/2])
		x = np.random.randint(-dimx/2 + edge_buffer, dimx/2 - + edge_buffer)

		if y < 0:
			cor_dir = np.array([0,1])
		else:
			cor_dir = np.array([0,-1])

	else:
		x = np.random.choice([-dimx/2, dimx/2])
		y = np.random.randint(-dimy/2 + edge_buffer, dimy/2 - edge_buffer)

		if x < 0:
			cor_dir = np.array([1,0])
		else:
			cor_dir = np.array([-1,0])

	# Track position of growth
	current_pos = np.array([x,y])

	swarm_origin = current_pos[:] + 5*cor_dir

	################################# Create first beacon at starting position ###################################


	beacons.positions = np.array([[swarm_origin[0], swarm_origin[1]]])
	# Set type to disperse
	beacons.behaviour_type = np.array([rules['start']])
	beacons.radius = np.array([0.5*cor_width])

	##############################################################################################################



	# Create root segment 

	segment_length = np.random.randint(min_seg_len, max_seg_len)

	# update new position
	new_pos = current_pos + segment_length*cor_dir

	start_pointA = current_pos - 0.5*cor_width*np.logical_not(cor_dir)
	start_pointB = current_pos + 0.5*cor_width*np.logical_not(cor_dir)

	end_pointA = current_pos - 0.5*cor_width*np.logical_not(cor_dir) + segment_length*cor_dir
	end_pointB = current_pos + 0.5*cor_width*np.logical_not(cor_dir) + segment_length*cor_dir

	seg_poly = Polygon([(end_pointA),(end_pointB),(start_pointB),(start_pointA)])

	current_pos = new_pos[:]

	poly_list.append(seg_poly)

	total_poly = seg_poly

	#################################################

	# Add in a doorway at the end of the corridor

	doorway_size = 5

	doorway_set = []

	door_lineA = np.array([end_pointA - 0.5*cor_width*cor_dir, new_pos - (doorway_size/2)*np.logical_not(cor_dir) - 0.5*cor_width*cor_dir])
	door_lineB = np.array([end_pointB- 0.5*cor_width*cor_dir, new_pos + (doorway_size/2)*np.logical_not(cor_dir) - 0.5*cor_width*cor_dir])

	doorway_set.append(door_lineA)
	doorway_set.append(door_lineB)


	#################################################



	# 180 rotation matrix
	angle = 0.5*math.pi

	rot90 = np.array([[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]])

	angle = math.pi
	rot180 = np.array([[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]])

	# The max number of turns that can be made in a row
	max_turns = 2
	turn_counter = 0
	last_turn = 0

	growth_complete = False

	parent_pos = None

	grow_counter = 0

	stack = []
	stack.append(grow_steps)

	subtree_active = False

	# Define corridor modules for selection

	# Loop through growth process adding further corridors

	growth_sequence = []

	area_lim = cor_proportion*dimx*dimy

	while total_poly.area <= area_lim:

		# Probability to turn growth direction 90 degrees

		# Rotation matrix = [cos(theta)   sin(theta)]
		# 	              [-sin(theta)    cos(theta)]


		# //////////////////// Probability to add an S-shaped bend /////////////////////


		# if np.random.uniform(0,1) <= 0.1:

		# 	start = current_pos
			

		# 	seg_dir = cor_dir

		# 	length = np.random.randint(15,25)


		# 	start_pointA = start - 0.5*cor_width*np.matmul(rot90,seg_dir) + 0.5*cor_width*np.matmul(rot180,seg_dir)
		# 	start_pointB = start + 0.5*cor_width*np.matmul(rot90,seg_dir)+ 0.5*cor_width*np.matmul(rot180,seg_dir)

		# 	end_pointA = (start + seg_dir*length)  - 0.5*cor_width*np.matmul(rot90,seg_dir) 
		# 	end_pointB = (start + seg_dir*length) + 0.5*cor_width*np.matmul(rot90,seg_dir)

		# 	seg_poly = Polygon([np.round(end_pointA),np.round(end_pointB),np.round(start_pointB),np.round(start_pointA)])

		# 	print('start points: (%s, %s)' % (start_pointA, start_pointB))
		# 	print('end points: (%s, %s)' % (end_pointA, end_pointB))

		# 	# Choose angles to creat two opposite turns
		# 	first_angle = np.random.choice([-0.5*math.pi, 0.5*math.pi])
		# 	angles = [first_angle, -1*first_angle]

		# 	start = start + seg_dir*length

		# 	print(' new start: ', start)

		# 	for j in range(2):

		# 		# Get random length of first segment

		# 		length = np.random.randint(15,25)

		# 		rotation = np.array([[math.cos(angles[j]), math.sin(angles[j])],[-math.sin(angles[j]), math.cos(angles[j])]])


		# 		# Rotate direction of growth
		# 		seg_dir = np.matmul(rotation,seg_dir)

		# 		print('segment direction: ', seg_dir)

		# 		start_pointA = start - 0.5*cor_width*np.matmul(rot90,seg_dir) + 0.5*cor_width*np.matmul(rot180,seg_dir)
		# 		start_pointB = start + 0.5*cor_width*np.matmul(rot90,seg_dir)+ 0.5*cor_width*np.matmul(rot180,seg_dir)

		# 		end_pointA = (start + seg_dir*length)  - 0.5*cor_width*np.matmul(rot90,seg_dir) 
		# 		end_pointB = (start + seg_dir*length) + 0.5*cor_width*np.matmul(rot90,seg_dir)

		# 		print('start points: (%s, %s)' % (start_pointA, start_pointB))
		# 		print('end points: (%s, %s)' % (end_pointA, end_pointB))

		# 		new_poly = Polygon([np.round(end_pointA),np.round(end_pointB),np.round(start_pointB),np.round(start_pointA)])

		# 		start = start + seg_dir*length


		# 		seg_poly = seg_poly.union(new_poly)

		# 	# Attach new bend with random rotation
		# 	# rot = np.random.choice([-90,0,90])

		# 	# seg_poly = affinity.rotate(seg_poly, rot)

		# 	# update current position!!!
		# 	current_pos = start

		# 	cor_dir = seg_dir

		# 	# Add poly to corridor poly
		# 	total_poly = total_poly.union(seg_poly)


		# ///////////////////////////////////////////////////////////////

		if np.random.uniform(0,1) <= subtree_prob and subtree_active == False:

			# Start growth of a sub corridor
			subtree_steps = np.random.randint(1, 3)

			stack.append(subtree_steps)

			# Save starting point
			parent_pos = current_pos

			# Add bool to stop subtrees upon subtrees
			subtree_active = True



		if turn_counter >= max_turns:
			angle = 0
			turn_counter = 0
		else:
			# Randomly switch direction of the corridor growth

			if np.random.uniform(0,1) <= 0.4:

				angle = 0
			else:
				angle = np.random.choice([-0.5*math.pi , 0.5*math.pi])

			if angle != last_turn:
				turn_counter += 1

		x_dir = cor_dir[0]*math.cos(angle) + cor_dir[1]*math.sin(angle)
		y_dir = -cor_dir[0]*math.sin(angle) + cor_dir[1]*math.cos(angle)

		cor_dir = np.array([x_dir, y_dir])

		# If angle is 0, combine poly with last

		# Detect if a turn is made
		if angle != 0:

			# Has there been a turn here already?

			# check if current pos has already been added as a beacon

			out = np.isin(beacons.positions, current_pos)

			exists = np.sum(np.logical_and(out.T[0], out.T[1]))

			# Check if no co-ordinates match
			if np.sum(exists) != 0:

				# Beacon position already exists, change to junction beacon

				index = np.where(exists == True)

				# Update beacon behaviour 
				beacons.behaviour_type[index[0]] = 3


			else:
				# if left turn add beacon

				scale = 1.
				if angle == 0.5*math.pi:
			
					beacons.positions = np.concatenate((beacons.positions, np.array([current_pos])), axis = 0)
					beacons.behaviour_type = np.concatenate((beacons.behaviour_type, np.array([rules['left']])), axis = 0)

				# if right turn add beacon
				if angle == -0.5*math.pi:
			
					beacons.positions = np.concatenate((beacons.positions, np.array([current_pos])), axis = 0)
					beacons.behaviour_type = np.concatenate((beacons.behaviour_type, np.array([rules['right']])), axis = 0)
				beacons.radius = np.concatenate((beacons.radius, np.array([scale*(cor_width/2)])), axis = 0)


		# Grow further segments

		segment_length = np.random.randint(min_seg_len, max_seg_len)


		# update new position
		new_pos = current_pos + segment_length*cor_dir

		# Check proximity to boundaries

		'''
			If corridors grow but stop close to the environment boundaries,
			extend growth to reach the boundary. Fixes issues with strangely
			shaped rooms which haven't been fully seperated by short corridors.

		'''
		boundary_buffer = 15
		if abs((dimx/2) - max(abs(new_pos[0]), abs(new_pos[1]))) <= boundary_buffer:

			# Round corridor to the edge of the boundary
			new_pos = current_pos + (segment_length + boundary_buffer)*cor_dir



		radius = beacons.radius[0]
		overlap = 0.1

		# Check if direction is horizontal
		if cor_dir[0] != 0:

			minx = min(current_pos[0], new_pos[0])
			maxx = max(current_pos[0], new_pos[0])

			x = np.arange(minx, maxx-radius, (1-overlap)*2*radius)
			y = new_pos[1]*np.ones(len(x))

		# Check if direction is vertical
		if cor_dir[1] != 0:

			miny = min(current_pos[1], new_pos[1])
			maxy = max(current_pos[1], new_pos[1])

			y = np.arange(miny, maxy-radius, (1-overlap)*2*radius)
			x = new_pos[0]*np.ones(len(y))
	
		points = np.zeros((len(x)*len(y), 2))
		
		count = 0
		for k in x:
			for j in y:
				points[count][0] = k 
				points[count][1] = j
				count += 1

		

		mag = cdist(points, beacons.positions) <= 2*radius
		mag = np.sum(mag, axis = 1)



		# Determine vertices of added poly

		start_pointA = current_pos - 0.5*cor_width*np.matmul(rot90,cor_dir) + 0.5*cor_width*np.matmul(rot180,cor_dir)
		start_pointB = current_pos + 0.5*cor_width*np.matmul(rot90,cor_dir)+ 0.5*cor_width*np.matmul(rot180,cor_dir)

		end_pointA = new_pos - 0.5*cor_width*np.matmul(rot90,cor_dir) 
		end_pointB = new_pos + 0.5*cor_width*np.matmul(rot90,cor_dir)

		seg_poly = Polygon([np.round(end_pointA),np.round(end_pointB),np.round(start_pointB),np.round(start_pointA)])

		# Add new poly to total poly
		total_poly = total_poly.union(seg_poly)


	
		# Check if grown outside of bounds
		
		if Point(new_pos[0], new_pos[1]).within(poly_bound) == True:

			# Continue to new point if within bounds, otherwise stay at current position
			current_pos = new_pos[:]
		# else: 

		# 	# Move to edge of perimeter

		# 	current_pos = current_pos + (35*cor_dir - current_pos)


		# Countdown remaining steps
		stack[len(stack)-1] -= 1

		if stack[len(stack)-1] == 0:

			if len(stack) == 1 and stack[0] == 0:

				# Reached max number of steps. End growth.
				growth_complete = True
			else:

				del stack[-1]

				# Return to original position before sub-tree growth
				current_pos = parent_pos

				subtree_active = False

	#print('The corridor poly is .... ', total_poly)


	return total_poly, swarm_origin, beacons, doorway_set


def simplifiy_polygon(poly):


	# Loop through vertices until redundant vertice found

	#print('This is the poly you were looking for: ', poly)

	if type(poly) is MultiPolygon:

		

		poly = poly[0].union(poly[1])

	vertices = list(poly.exterior.coords)
	
	redundant = True

	while redundant == True:

		vertex_count = 0

		redundant = False

		new_vertices = list()

		# Keep looping through vertices until no redundancy found
		while redundant == False and vertex_count < len(vertices) - 2:

			'''
			for 3 points, if the mid-point lies on the line between the
			start and end point, the midpoint is redundant
			'''
			start = np.round(vertices[vertex_count])
			mid = np.round(vertices[vertex_count+1])
			end = np.round(vertices[vertex_count+2])

			line = LineString([start,end])

			if Point(mid).within(line) == True:
				# midpoint is redundant
				new_vertices = vertices[:vertex_count+1] + vertices[vertex_count+2:]
				#print('New vertices ', new_vertices)
				redundant = True
			else:
				vertex_count += 1

		if redundant == True:

			vertices = new_vertices[:]
		

	simple_poly = Polygon(vertices)



	return simple_poly
		

def rand_mapgen(dimx, dimy, cor_num, cor_width, min_seg_len, max_seg_len, subtree_prob, grow_steps, cor_proportion):

	# Define bounds of the environment
	bottom = -dimy/2
	top = dimy/2
	left = -dimx/2
	right = dimx/2

	# Create behaviour beacons to build in parallel to map gen

	beacons = beh_beacons(1)
	poly_bound = Polygon([(left, bottom), (left, top), (right, top), (right, bottom)])
	poly_list = []
	# Add the bounding walls

	# polyA, swarm_origin, beacons = grow_corridor(dimx, dimy, cor_width, turn_rate=0, max_length=10, min_seg_len=40, 
	# 											max_seg_len=60, subtree_prob = 0.8, grow_steps = 5, beacons=beacons)

	'''
	Behaviours:

	1 - clockwise
	2 - anti-clockwise
	3 - disperse
	4 - flock
	5 - sweep horizontal
	6 - sweep vertical
	7 - roundabout

	'''

	#Rule set A
	rule_set = {'vertical_room': 1, 'horizontal_room': 2, 'start': 3, 'left': 1, 'right': 2}

	# Rule set B
	# rule_set = {'vertical_room': 1, 'horizontal_room': 2, 'start': 3, 'left': 1, 'right': 2}

	# # Rule set C: Add sweeps in rooms	
	# rule_set = {'vertical_room': 6, 'horizontal_room': 5, 'start': 3, 'left': 7, 'right': 7}
	
	# polyA, swarm_origin, beacons = grow_corridor(dimx, dimy, cor_width, turn_rate=0, max_length=10, min_seg_len=40, 
	# 											max_seg_len=60, subtree_prob = 0.4, grow_steps = 6, beacons=beacons, rules= rule_set)


	 ##### GOOD LEVEL OF COMPLEXITY FOR TESTING ###################
	polyA, swarm_origin, beacons, doorway_set = grow_corridor(dimx, dimy, cor_width, turn_rate=0, max_length=10, min_seg_len=min_seg_len, 
	 											max_seg_len=max_seg_len, subtree_prob = subtree_prob, grow_steps = grow_steps, 
	 											beacons=beacons, rules= rule_set, cor_proportion=cor_proportion)


	# Cut off sections out of bounds

	polyA = polyA.intersection(poly_bound)


	minx, miny, maxx, maxy = poly_bound.bounds

	# Generate beacon points over these bounds

	radius = beacons.radius[0]
	overlap = 0.9

	x = np.arange(minx, maxx-radius, (1-overlap)*2*radius)
	y = np.arange(miny, maxy-radius, (1-overlap)*2*radius)
	points = np.zeros((len(x)*len(y), 2))
	
	count = 0
	for k in x:
		for j in y:
			points[count][0] = k 
			points[count][1] = j
			count += 1


	# Remove redundant vertices
	polyA = simplifiy_polygon(polyA)
	
	# Split environment into different segments
	poly_segments = poly_bound.difference(polyA)


	if type(poly_segments) is MultiPolygon:

		for poly in poly_segments:
			poly_list.append(poly)
	else:
		poly_list.append(poly_segments)
		

	# RULE: If segment is very large split into two segments




	################################# Add doorways into segments #########################################
	
	new_poly_list = []

	env_map = asim.map()

	for poly in poly_list:

		# Check edges which don't touch boundaries
		x, y = poly.exterior.coords.xy

		edge_num = len(x) - 1

		'''	
		Rules for picking doors:
			1. rooms which are very small have no doors (obstacles)
			2. Can't be a boundary wall
		 	3. Doorway on the longest wall
		 	4. Some rooms have two doorways
		'''

		size_lim = 50

		# check edge lengths

		minx,miny,maxx,maxy = poly.bounds

		too_small = False

		edge_lim = 10

		if maxx-minx <= edge_lim or maxy-miny <= edge_lim:
			too_small = True

		# If poly area is too small, don't add doorway
		if poly.area >= size_lim and too_small == False: 


			########## add beacons to rooms #############



			# Centroid approach

			# find min length of room for beacon size

		
			

			# Fill room

			# Find the bounds of the room

			# minx, miny, maxx, maxy = poly.bounds

			# # Generate beacon points over these bounds

			# radius = beacons.radius[0]
			# overlap = 0.4

			# x = np.arange(minx + radius, maxx-radius, (1-overlap)*2*radius)
			# y = np.arange(miny + radius, maxy-radius, (1-overlap)*2*radius)
			# points = np.zeros((len(x)*len(y), 2))

			# shiftx = (maxx - minx) - (max(x) - min(x) + 2*radius)

			# x = x + 0.5*shiftx*np.ones(len(x))

			# shifty = (maxy - miny) - (max(y) - min(y) + 2*radius)

			# y = y + 0.5*shifty*np.ones(len(y))
			
			# count = 0
			# for k in x:
			# 	for j in y:
			# 		points[count][0] = k 
			# 		points[count][1] = j
			# 		count += 1


			# # Add beacon if point within room shape


			# for point in points:

			# 	mag = cdist(points, beacons.positions) <= 2*radius
			# 	mag = np.sum(mag, axis = 1)


			# 	if Point(point[0], point[1]).within(poly):
			# 		beacons.positions = np.concatenate((beacons.positions, np.array([[point[0], point[1]]])), axis = 0)
			# 		beacons.behaviour_type = np.concatenate((beacons.behaviour_type, np.array([2])), axis = 0)



			#############################################

			# Separate edges into viable options

			poly_lines = poly_to_lines(poly)
			# Remove lines which intersect with bounds

			valid_lines = list()
			unchanged_lines = list()

			for line in poly_lines:

				# Check boundary intersection with bound walls
				intersect = line.intersection(poly_bound.boundary)

				if type(intersect) is not LineString or intersect.length == 0:
	
					# keep if line intersects with bounds at one point
					valid_lines.append(line)
				
			
			# Pick line with highest length

			max_length = 0

			choice = None
			for n in range(len(valid_lines)):

				#print(v)

				if valid_lines[n].length > max_length:
					doorway_choice = n
					max_length = valid_lines[n].length


			##################### START CONVERSION INTO MAP OBJECT ################################

			for n in range(len(valid_lines)):
				if n == doorway_choice:
			
					x, y = valid_lines[n].coords.xy 

					# proportion of edge size to make doorsize
					door_size = 0.4

					grad = np.array([x[1],y[1]]) - np.array([x[0],y[0]])

					wall = environment.wall(); wall.start = [x[0], y[0]]; wall.end = wall.start + 0.5*(1-door_size)*grad
					env_map.obsticles.append(wall)

					wall = environment.wall(); wall.start = [x[1], y[1]]; wall.end = wall.start - 0.5*(1-door_size)*grad
					env_map.obsticles.append(wall)

					# If doorway is horizontal
					scale = 1.
					if y[0] == y[1]:
						# Add vertical sweep
						size = min(maxy-miny, maxx-minx)*0.5
			
						beacons.positions = np.concatenate((beacons.positions, np.array([[poly.centroid.xy[0][0], poly.centroid.xy[1][0]]])), axis = 0)
						beacons.behaviour_type = np.concatenate((beacons.behaviour_type, np.array([rule_set['vertical_room']])), axis = 0)
						beacons.radius = np.concatenate((beacons.radius, np.array([scale*size])), axis = 0)
					else:
						size = min(maxy-miny, maxx-minx)*0.5
			
						beacons.positions = np.concatenate((beacons.positions, np.array([[poly.centroid.xy[0][0], poly.centroid.xy[1][0]]])), axis = 0)
						beacons.behaviour_type = np.concatenate((beacons.behaviour_type, np.array([rule_set['horizontal_room']])), axis = 0)
						beacons.radius = np.concatenate((beacons.radius, np.array([scale*size])), axis = 0)


				else:	
					x, y = valid_lines[n].coords.xy 
					if y[0] == y[1]:
						wall = environment.wall(); wall.start = [x[0], min(y[0],y[1])]; wall.end = [x[1],max(y[0],y[1])]
						env_map.obsticles.append(wall)
					else:
						wall = environment.wall(); wall.start = [min(x[0],x[1]), y[0]]; wall.end = [max(x[0],x[1]),y[1]]
						env_map.obsticles.append(wall)

		else:

			# room is too small for door
			for n in range(len(x)-1):
				# convert into wall object

				# Only add some of the walls to make an open obstacle, not closed box

				if np.random.choice([True, False]) == True:
					wall = environment.wall(); wall.start = [x[n], y[n]]; wall.end = [x[n+1],y[n+1]]; env_map.obsticles.append(wall)


	######## Convert doorway set into wall objects ###########


	# for line in doorway_set:

	# 	wall = environment.wall(); wall.start = line[0]; wall.end = line[1]; env_map.obsticles.append(wall)


	################### ADD ADDITIONAL DOORWAYS TO CORRIDORS ############################


	doorway_prob = 0.15

	for m in range(len(poly_list)):

		for k in range(len(poly_list)):

			if m != k:

				# Pairwise check of possible doorways

				x1, y1 = poly_list[m].exterior.coords.xy
				x2, y2 = poly_list[k].exterior.coords.xy

				# stack co-ordinates, don't add repeat starting point
				pointsA = np.stack([x1,y1], axis = 1)[1:]
				pointsB = np.stack([x2,y2], axis = 1)[1:]

				mag = cdist(pointsA, pointsB)

				out = (mag == cor_width)*(mag > 0)

				# get indices of valid points

				indA, indB = np.where(out == True)

				for z in range(len(indA)):

					# probability to add doorway
					if np.random.uniform(0,1) <= doorway_prob:

						# return valid points for doorway
						point1 = pointsA[indA[z]]
						point2 = pointsB[indB[z]]

						diff = point1 - point2

						wall = environment.wall(); wall.start = point1; wall.end = point1 - 0.25*diff; env_map.obsticles.append(wall)
						wall = environment.wall(); wall.start = point2; wall.end = point2 + 0.25*diff; env_map.obsticles.append(wall)




	# Set swarm origin point for map
	env_map.swarm_origin = swarm_origin[:]

	# Add bounding walls
	wall = environment.wall(); wall.start = [left,bottom]; wall.end = [left,top]; env_map.obsticles.append(wall)
	wall = environment.wall(); wall.start = [left,top]; wall.end = [right,top]; env_map.obsticles.append(wall)
	wall = environment.wall(); wall.start = [right,top]; wall.end = [right,bottom]; env_map.obsticles.append(wall)
	wall = environment.wall(); wall.start = [right,bottom]; wall.end = [left,bottom]; env_map.obsticles.append(wall)

	env_map.dimensions = [dimx,dimy]

	return env_map, beacons
	

def poly_to_lines(poly):

	coords = list(poly.exterior.coords)

	lines = list()

	for n in range(len(coords)-1):

		# Check if 

		lines.append(LineString([coords[n], coords[n+1]]))

	return lines


class beh_beacons(object):

	def __init__(self, size):

		self.positions = np.zeros((size, 2))
		self.behaviour_type = np.zeros(size)
		self.radius = None
		self.fitness = 0

	def default_set(self):

		'''
		Beacon types:

		0 - Random walk
		1 - clockwise
		2 - Anti-clockwise
		3 - dispersion
		'''

		# Bench1 map
		self.positions = np.array([[15,-5], [20,15], [10,-30], [30,-30], [-30,30], [-30,-30], [-20,-20], [-30,-10] , [-20,0], [0, -5]])
		self.radius = np.array([10,10,10,10,10,10,10,10,10,10])
		self.behaviour_type = np.array([2, 1, 2, 1, 2, 1, 2, 1, 2, 1])

		# # Bench2 map
		# self.beacons = np.array([[-35,35], [10,-30], [35,35], [35,5], [15,5], [-20,-10], [30,-30], [20,20], [-20,15]])
		# self.radius = np.array([10,10,10,10,10,10,10,10,10,10])
		# self.behaviour_type = np.array([1, 2, 1, 1, 2, 1, 2, 1, 1])

	def beacon_check(self, swarm):

		mag = cdist(self.positions, swarm.agents)

		# Reset agent behaviours before detecting intersection with beacons
		swarm.behaviour = 0*np.ones(swarm.size)

		for i in range(len(self.positions)):

			intersection = mag <= self.radius[i]

			for n in range(swarm.size):

				if intersection[i][n] == True:

					swarm.behaviour[n] = self.behaviour_type[i]
					#swarm.behaviour[n] = 0

					# print('Bhaviour set')
					# print('Swarm behaviour: ', swarm.behaviour)

				# else:
				# 	# Set default behaviour to random walk.
				# 	swarm.behaviour[n] = 0

				# # If agent intersects with a beacon
				# beacon_intersect = np.sum(intersection[n])
				
				# if beacon_intersect == 1:

				# 	beacon_num = np.where(intersection[n] == 1)[0] 

				# 	swarm.behaviour[n] = self.behaviour_type[beacon_num]

				# elif beacon_intersect == 0:
				# 	# Set default behaviour to random walk.
				# 	swarm.behaviour[n] = 0

		

class target_set(object):

	def __init__(self):
		self.targets = []
		self.radius = 0
		self.found = 0
		self.coverage = 0
		self.old_state = np.zeros(len(self.targets))
		self.fitmap = []

	def set_state(self, state):

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

		now = time.time()

		score = 0
		# adjacency matrix of agents and targets
		mag = cdist(swarm.agents, self.targets)

		# Check which distances are less than detection range
		a = mag < self.radius
		# Sum over agent axis 
		detected = np.sum(a, axis = 0)
		# convert to boolean, targets with 0 detections set to false.
		detected = detected > 0
		# Check detection against previous state. If a target is already found return false.
		updated = np.logical_or(detected, self.old_state) 





		# Do mem update

		# Shifts mem data by one step
		#swarm.agent_mem = np.roll(swarm.agent_mem, 1, axis = 0)

		# Write in new state
		# for n in range(swarm.size):

		# 	index = np.where(a[n] == 1)[0][0]
			
		# 	swarm.agent_mem[0][n] = updated[index]


		
		# Accumulate scores for each target found
		# Tracks the total targets found so far. Not this iteration.
		score = np.sum(updated)
		self.coverage = score/len(self.targets)	

		# How many targets were found this iteration.
		found = np.logical_xor(detected, self.old_state)*detected

		score = 0
		# Determine score based on decay of target rewards.

		# Get indices of found targets
		found = np.where(found == True)
		
		for ind in found[0]:

			# row = int((self.targets[ind][1]+39)/2.5)
			# col = int((self.targets[ind][0]+74)/2.5)
			row = int((self.targets[ind][1]+ (swarm.map.dimensions[1]/2) -1)/1)
			col = int((self.targets[ind][0]+ (swarm.map.dimensions[0]/2) -1)/1)

			# Find the decay constant for the target.
			decay = self.fitmap[row][col]
			score += np.exp(3*((-t*decay)/timesteps))

		self.old_state = updated
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


class swarm(object):

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

		self.happiness = None # Binary list indicating un-happy or happy
		self.prev_happiness = 0
		self.happiness_threshold = 0
		self.update_rate = 20 # Opinion sharing frequency 
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
		self.previous_state = self.agents
		self.longterm_state = self.agents
		self.longterm_counter = np.zeros(self.size)
		self.longterm_timelimit = np.random.randint(3*self.update_rate-2, 3*self.update_rate+2, self.size)

		self.happiness_noise = np.zeros(self.size)


		self.collision_count = np.zeros(self.size)

		self.objective_count = np.zeros(self.size)
		self.agent_objective_states = [None for x in range(self.size)]
		# mag = 0.4
		# self.happiness_noise = np.random.uniform(-mag, mag, self.size)

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

		#collision_detection(self)

		dispersion(self, np.array([0,0]), 60, noise)
		# sweep_LR(self, self.time, 10, noise)
		# sweep_UD(self, self.time, 10, noise)
		# roundabout(self, 0.1, 0.02, 5, 0.3, noise, self.time)
		random_walk(self, 0.01)
		rotate_clock(self, [-2,2], 0.1)
		rotate_anti(self, [-1,3], 0.1)
		flocking(self, 0.1, 0.04, 5, 0.5, noise)
		# # flocking_infoGain(self, 0.1, 0.05, 5, 0.9, noise)

		force = 20

		# east(self, np.array([1,0]), force, noise)
		# west(self, np.array([-1,0]), force, noise)
		# north(self, np.array([0,1]), force, noise)
		# south(self, np.array([0,-1]), force, noise)
		# northeast(self, np.array([1,1]), force, noise)
		# northwest(self, np.array([-1,1]), force, noise)
		# southeast(self, np.array([1,1]), force, noise)
		# southwest(self, np.array([-1,-1]), force, noise)

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

		for n in range(0,self.size):
			# Loop through all agents and update their decision making states


			# update agent happiness each timestep over 5 steps

			if self.opinion_timer[n] >= 5:
			# have far have they moved from previous position?
				self.happiness[n] = cdist([self.previous_state[n]], [self.agents[n]])[0][0]/self.opinion_timer[n]

				max_val = self.speed

				inrange = np.where(prox[n] == True)[0]

				neighbour_num = len(inrange)
				max_neighbour = 10
				if neighbour_num > max_neighbour:
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

						# *** Ignore agents with the same opnion 
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
				index = 0
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
				if neighbour_num > max_neighbour:
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

					if max_opinion >= opinion_buffer*self.happiness[n] and self.behaviour[n] != self.behaviour[index]:

						# **** SWITCH TO BEHAVIOUR OF NEIGHBOUR WITH HIGHER PERFORMANCE

						self.behaviour[n] = self.behaviour[index]
					else:

						# If no one else is performing better than you, try random behaviour


						options = [0,1,2,3,4,5,6,7,8]

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
							options = [0,1,2,3,4,5,6,7,8]
							#options = [0,10,11,9]
							self.behaviour[n] = np.random.choice(options)

					# All neighbours take on chosen behaviour
					for k in range(len(inrange)):
						if np.random.uniform(0,1) <= 0.5 and np.random.uniform(0,1) <= self.comm_prob:
							self.behaviour[inrange[k]] = self.behaviour[n]
							self.opinion_timer[inrange[k]] = 0

				self.collision_count[n] = 0

				self.previous_state[n] = self.agents[n]

				# Reset agent timer
				# *** If happiness is low check opinion again more quickly
				self.opinion_timelimit[n] = (1+self.happiness[n])*np.random.randint(self.update_rate-2 , self.update_rate+2)

		
		

					# Increment longterm timer
		self.longterm_counter = self.longterm_counter + np.ones(self.size)

		#np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

		# print('\n\nSwarm happiness: ' , self.happiness)

		# print('\n\nSwarm update timers: ', self.opinion_timer)

		# print('\n\nSwarm update LIMITS: ', self.opinion_timelimit)

		# print('\n\nSwarm behaviours: ' , self.behaviour)

		# print('\n\n\nlongterm happiness: ', self.long_happiness)

	def get_state_opinionB(self):

		# --------------------------------------------------------------
		# Removed broadcasting from happy agents!
		# --------------------------------------------------------------


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

		for n in range(0,self.size):
			# Loop through all agents and update their decision making states


			# update agent happiness each timestep over 5 steps

			if self.opinion_timer[n] >= 5:
			# have far have they moved from previous position?
				self.happiness[n] = cdist([self.previous_state[n]], [self.agents[n]])[0][0]/self.opinion_timer[n]

				max_val = self.speed

				inrange = np.where(prox[n] == True)[0]

				neighbour_num = len(inrange)
				max_neighbour = 10
				if neighbour_num > max_neighbour:
					neighbour_num = 9

				# Normalize between 0 to 1
				self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
				#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2

				############# Happiness with collision detection

				# collision_prop = (self.opinion_timer[n] - (self.collision_count[n]+1))/self.opinion_timer[n]
				# if collision_prop <= 0: collision_prop = 0.01

				# self.happiness[n] = collision_prop*((max_neighbour - neighbour_num)/max_neighbour)



				# Add noise to happiness measures

				self.happiness[n] += self.happiness_noise[n]



			

			#Check whether it's time to update opinion

			if self.opinion_timer[n] >= self.opinion_timelimit[n]:
				# Update opinion

				self.opinion_timer[n] = 0
			

				opinion_buffer = 1.1 # percent difference

				# Compare opinion with neighbours

				inrange = np.where(prox[n] == True)[0]

				max_opinion = 0
				index = 0
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
				if neighbour_num > max_neighbour:
					neighbour_num = 9

				# Normalize between 0 to 1
				self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
				#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2

				# Add noise to happiness measures

				############# Happiness with collision detection

				# collision_prop = (self.opinion_timelimit[n] - (self.collision_count[n]+1))/self.opinion_timelimit[n]
				# if collision_prop <= 0: collision_prop = 0.01

				# self.happiness[n] = collision_prop*((max_neighbour - neighbour_num)/max_neighbour)

				# print('\n\n RESET COLLISION COUNTER FROM %d' % (self.collision_count[n]))


				# print('Agent %d collision count is %d' % (n, self.collision_count[n]))

				# self.collision_count[n] = 0

				# print('Agent %d collision count is %d AFTER RESET' % (n, self.collision_count[n]))


				self.happiness[n] += self.happiness_noise[n]

				

				# first check own performance

				if self.happiness[n] - self.prev_happiness[n] <= 0.2 and self.happiness[n] <= 0.5:

					#Something needs to change!

					self.prev_happiness[n] = self.happiness[n]

					# Compare best opinion with own opinion

					# Only change if they're opinion is different to yours

					if max_opinion >= opinion_buffer*self.happiness[n] and self.behaviour[n] != self.behaviour[index]:

						# **** SWITCH TO BEHAVIOUR OF NEIGHBOUR WITH HIGHER PERFORMANCE

						self.behaviour[n] = self.behaviour[index]
					else:

						# If no one else is performing better than you, try random behaviour


						options = [0,1,2,3,4,5,6,7,8]

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
							options = [0,1,2,3,4,5,6,7,8]
							#options = [0,10,11,9]
							self.behaviour[n] = np.random.choice(options)

					# All neighbours take on chosen behaviour
					for k in range(len(inrange)):
						if np.random.uniform(0,1) <= 0.5 and np.random.uniform(0,1) <= self.comm_prob:
							self.behaviour[inrange[k]] = self.behaviour[n]
							self.opinion_timer[inrange[k]] = 0

				self.collision_count[n] = 0

				self.previous_state[n] = self.agents[n]

				# Reset agent timer
				# *** If happiness is low check opinion again more quickly
				self.opinion_timelimit[n] = (1+self.happiness[n])*np.random.randint(self.update_rate-2 , self.update_rate+2)

		
		

					# Increment longterm timer
		self.longterm_counter = self.longterm_counter + np.ones(self.size)

		
	def get_state_opinionC(self):

		# --------------------------------------------------------------
		# No happy agent broadcasting and no sharing evidence with neighbours
		# --------------------------------------------------------------


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

		for n in range(0,self.size):
			# Loop through all agents and update their decision making states


			# update agent happiness each timestep over 5 steps

			if self.opinion_timer[n] >= 5:
			# have far have they moved from previous position?
				self.happiness[n] = cdist([self.previous_state[n]], [self.agents[n]])[0][0]/self.opinion_timer[n]

				max_val = self.speed

				inrange = np.where(prox[n] == True)[0]

				neighbour_num = len(inrange)
				max_neighbour = 10
				if neighbour_num > max_neighbour:
					neighbour_num = 9

				# Normalize between 0 to 1
				self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
				#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2

				############# Happiness with collision detection

				# collision_prop = (self.opinion_timer[n] - (self.collision_count[n]+1))/self.opinion_timer[n]
				# if collision_prop <= 0: collision_prop = 0.01

				# self.happiness[n] = collision_prop*((max_neighbour - neighbour_num)/max_neighbour)



				# Add noise to happiness measures

				self.happiness[n] += self.happiness_noise[n]





			#Check whether it's time to update opinion

			if self.opinion_timer[n] >= self.opinion_timelimit[n]:
				# Update opinion

				self.opinion_timer[n] = 0
			

				opinion_buffer = 1.1 # percent difference

				# Compare opinion with neighbours

				inrange = np.where(prox[n] == True)[0]

				max_opinion = 0
				index = 0
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
				if neighbour_num > max_neighbour:
					neighbour_num = 9

				# Normalize between 0 to 1
				self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
				#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2

				# Add noise to happiness measures

				############# Happiness with collision detection

				# collision_prop = (self.opinion_timelimit[n] - (self.collision_count[n]+1))/self.opinion_timelimit[n]
				# if collision_prop <= 0: collision_prop = 0.01

				# self.happiness[n] = collision_prop*((max_neighbour - neighbour_num)/max_neighbour)

				# print('\n\n RESET COLLISION COUNTER FROM %d' % (self.collision_count[n]))


				# print('Agent %d collision count is %d' % (n, self.collision_count[n]))

				# self.collision_count[n] = 0

				# print('Agent %d collision count is %d AFTER RESET' % (n, self.collision_count[n]))


				self.happiness[n] += self.happiness_noise[n]

				

				# first check own performance

				if self.happiness[n] - self.prev_happiness[n] <= 0.2 and self.happiness[n] <= 0.5:

					#Something needs to change!

					self.prev_happiness[n] = self.happiness[n]

					# Compare best opinion with own opinion

					# Only change if they're opinion is different to yours

					if max_opinion >= opinion_buffer*self.happiness[n] and self.behaviour[n] != self.behaviour[index]:

						# **** SWITCH TO BEHAVIOUR OF NEIGHBOUR WITH HIGHER PERFORMANCE

						self.behaviour[n] = self.behaviour[index]
						#pass
					else:

						# If no one else is performing better than you, try random behaviour


						# Reset options if all neighbours are doing different behaviours
						options = [0,1,2,3,4,5,6,7,8]
						#options = [0,10,11,9]
						self.behaviour[n] = np.random.choice(options)

					# # All neighbours take on chosen behaviour
					# for k in range(len(inrange)):
					# 	if np.random.uniform(0,1) <= 0.5 and np.random.uniform(0,1) <= self.comm_prob:
					# 		self.behaviour[inrange[k]] = self.behaviour[n]
					# 		self.opinion_timer[inrange[k]] = 0

				self.collision_count[n] = 0

				self.previous_state[n] = self.agents[n]

				# Reset agent timer
				# *** If happiness is low check opinion again more quickly
				self.opinion_timelimit[n] = (1+self.happiness[n])*np.random.randint(self.update_rate-2 , self.update_rate+2)

		
		

					# Increment longterm timer
		self.longterm_counter = self.longterm_counter + np.ones(self.size)

		#np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

		# print('\n\nSwarm happiness: ' , self.happiness)

		# print('\n\nSwarm update timers: ', self.opinion_timer)

		# print('\n\nSwarm update LIMITS: ', self.opinion_timelimit)

		# print('\n\nSwarm behaviours: ' , self.behaviour)

		# print('\n\n\nlongterm happiness: ', self.long_happiness)


	def get_state_avoidance(self):

		# Function to update opinions and compare with neighbours

		#### -----------------------------------------------

		collision_detection(self)

		### -------------------------------------------------

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

		for n in range(0,self.size):
			# Loop through all agents and update their decision making states


			# update agent happiness each timestep over 5 steps

			if self.opinion_timer[n] >= 5:
			# have far have they moved from previous position?
				self.happiness[n] = cdist([self.previous_state[n]], [self.agents[n]])[0][0]/self.opinion_timer[n]

				max_val = self.speed

				inrange = np.where(prox[n] == True)[0]

				neighbour_num = len(inrange)
				max_neighbour = 10
				if neighbour_num > max_neighbour:
					neighbour_num = 9

				# Normalize between 0 to 1
				self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
				#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2

				############# Happiness with collision detection

				collision_prop = (self.opinion_timer[n] - (self.collision_count[n]+1))/self.opinion_timer[n]
				if collision_prop <= 0: collision_prop = 0.01

				self.happiness[n] = collision_prop*((max_neighbour - neighbour_num)/max_neighbour)



				# Add noise to happiness measures

				self.happiness[n] += self.happiness_noise[n]




			#High empowerment individuals continue to broadcast opinion

			if self.happiness[n] - self.prev_happiness[n] >= 0.2 or self.happiness[n] >= 0.5:

				inrange = np.where(prox[n] == True)[0]

				for k in range(len(inrange)):

						# Loop through neighbours and broadcast opinion

						# *** Ignore agents with the same opnion 
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
				index = 0
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
				if neighbour_num > max_neighbour:
					neighbour_num = 9

				# Normalize between 0 to 1
				self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
				#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2

				# Add noise to happiness measures

				############# Happiness with collision detection

				collision_prop = (self.opinion_timelimit[n] - (self.collision_count[n]+1))/self.opinion_timelimit[n]
				if collision_prop <= 0: collision_prop = 0.01

				self.happiness[n] = collision_prop*((max_neighbour - neighbour_num)/max_neighbour)

				# print('\n\n RESET COLLISION COUNTER FROM %d' % (self.collision_count[n]))


				# print('Agent %d collision count is %d' % (n, self.collision_count[n]))

				# self.collision_count[n] = 0

				# print('Agent %d collision count is %d AFTER RESET' % (n, self.collision_count[n]))


				self.happiness[n] += self.happiness_noise[n]

				

				# first check own performance

				if self.happiness[n] - self.prev_happiness[n] <= 0.2 and self.happiness[n] <= 0.5:

					#Something needs to change!

					self.prev_happiness[n] = self.happiness[n]

					# Compare best opinion with own opinion

					# Only change if they're opinion is different to yours

					if max_opinion >= opinion_buffer*self.happiness[n] and self.behaviour[n] != self.behaviour[index]:

						# **** SWITCH TO BEHAVIOUR OF NEIGHBOUR WITH HIGHER PERFORMANCE

						self.behaviour[n] = self.behaviour[index]
					else:

						# If no one else is performing better than you, try random behaviour


						options = [0,1,2,3,4,5,6,7,8]

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
							options = [0,1,2,3,4,5,6,7,8]
							#options = [0,10,11,9]
							self.behaviour[n] = np.random.choice(options)

					# All neighbours take on chosen behaviour
					for k in range(len(inrange)):
						if np.random.uniform(0,1) <= 0.5 and np.random.uniform(0,1) <= self.comm_prob:
							self.behaviour[inrange[k]] = self.behaviour[n]
							self.opinion_timer[inrange[k]] = 0

				self.collision_count[n] = 0

				self.previous_state[n] = self.agents[n]

				# Reset agent timer
				# *** If happiness is low check opinion again more quickly
				self.opinion_timelimit[n] = (1+self.happiness[n])*np.random.randint(self.update_rate-2 , self.update_rate+2)

		
		

					# Increment longterm timer
		self.longterm_counter = self.longterm_counter + np.ones(self.size)


	def get_state_coverage(self):

		# Function to update opinions and compare with neighbours

		#### -----------------------------------------------

		

		### -------------------------------------------------

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

		print('Swarm objective count: ' ,self.objective_count)
		# Check update status of agents

		for n in range(0,self.size):
			# Loop through all agents and update their decision making states


			# update agent happiness each timestep over 5 steps

			if self.opinion_timer[n] >= 5:
			# have far have they moved from previous position?
				self.happiness[n] = cdist([self.previous_state[n]], [self.agents[n]])[0][0]/self.opinion_timer[n]

				max_val = self.speed

				inrange = np.where(prox[n] == True)[0]

				neighbour_num = len(inrange)
				max_neighbour = 10
				if neighbour_num > max_neighbour:
					neighbour_num = 9

				# Normalize between 0 to 1
				self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
				#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2

				############# Happiness with collision detection

				coverage_prop = (self.objective_count[n]+1)/self.opinion_timer[n]
				if coverage_prop <= 0: coverage_prop = 0.01

				self.happiness[n] = coverage_prop*((max_neighbour - neighbour_num)/max_neighbour)



				# Add noise to happiness measures

				self.happiness[n] += self.happiness_noise[n]




			#High empowerment individuals continue to broadcast opinion

			if self.happiness[n] - self.prev_happiness[n] >= 0.2 or self.happiness[n] >= 0.5:

				inrange = np.where(prox[n] == True)[0]

				for k in range(len(inrange)):

						# Loop through neighbours and broadcast opinion

						# *** Ignore agents with the same opnion 
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
				index = 0
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
				if neighbour_num > max_neighbour:
					neighbour_num = 9

				# Normalize between 0 to 1
				self.happiness[n] = (self.happiness[n]/max_val)*((max_neighbour - neighbour_num)/max_neighbour)
				#self.happiness[n] = (self.happiness[n] + self.long_happiness[n])/2

				# Add noise to happiness measures

				############# Happiness with collision detection

				coverage_prop = (self.objective_count[n]+1)/self.opinion_timelimit[n]
				if coverage_prop <= 0: coverage_prop = 0.01

				self.happiness[n] = coverage_prop*((max_neighbour - neighbour_num)/max_neighbour)

				# print('\n\n RESET COLLISION COUNTER FROM %d' % (self.collision_count[n]))


				# print('Agent %d collision count is %d' % (n, self.collision_count[n]))

				# self.collision_count[n] = 0

				# print('Agent %d collision count is %d AFTER RESET' % (n, self.collision_count[n]))


				self.happiness[n] += self.happiness_noise[n]

				

				# first check own performance

				if self.happiness[n] - self.prev_happiness[n] <= 0.2 and self.happiness[n] <= 0.5:

					#Something needs to change!

					self.prev_happiness[n] = self.happiness[n]

					# Compare best opinion with own opinion

					# Only change if they're opinion is different to yours

					if max_opinion >= opinion_buffer*self.happiness[n] and self.behaviour[n] != self.behaviour[index]:

						# **** SWITCH TO BEHAVIOUR OF NEIGHBOUR WITH HIGHER PERFORMANCE

						self.behaviour[n] = self.behaviour[index]
					else:

						# If no one else is performing better than you, try random behaviour


						options = [0,1,2,3,4,5,6,7,8]

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
							options = [0,1,2,3,4,5,6,7,8]
							#options = [0,10,11,9]
							self.behaviour[n] = np.random.choice(options)

					# All neighbours take on chosen behaviour
					for k in range(len(inrange)):
						if np.random.uniform(0,1) <= 0.5 and np.random.uniform(0,1) <= self.comm_prob:
							self.behaviour[inrange[k]] = self.behaviour[n]
							self.opinion_timer[inrange[k]] = 0

				self.objective_count[n] = 0

				self.previous_state[n] = self.agents[n]

				# Reset agent timer
				# *** If happiness is low check opinion again more quickly
				self.opinion_timelimit[n] = (1+self.happiness[n])*np.random.randint(self.update_rate-2 , self.update_rate+2)

		
		

					# Increment longterm timer
		self.longterm_counter = self.longterm_counter + np.ones(self.size)

		
	def get_state_covdist(self):

		# Function to update opinions and compare with neighbours

		#### -----------------------------------------------

		

		### -------------------------------------------------

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

		print('Swarm objective count: ' ,self.objective_count)
		# Check update status of agents

		for n in range(0,self.size):
			# Loop through all agents and update their decision making states


			# update agent happiness each timestep over 5 steps

			if self.opinion_timer[n] >= 5:
			# have far have they moved from previous position?
				dist_happy = cdist([self.previous_state[n]], [self.agents[n]])[0][0]/self.opinion_timer[n]

				max_val = self.speed

				dist_happy = dist_happy/max_val

				inrange = np.where(prox[n] == True)[0]

				neighbour_num = len(inrange)
				max_neighbour = 10
				if neighbour_num > max_neighbour:
					neighbour_num = 9

			

				############# Happiness with collision detection

				coverage_prop = (self.objective_count[n]+1)/self.opinion_timer[n]
				if coverage_prop <= 0: coverage_prop = 0.01


				# Take the average value of coverage and istance happiness
				self.happiness[n] = ((coverage_prop + 2*dist_happy)/3)*((max_neighbour - neighbour_num)/max_neighbour)



				# Add noise to happiness measures

				self.happiness[n] += self.happiness_noise[n]




			#High empowerment individuals continue to broadcast opinion

			if self.happiness[n] - self.prev_happiness[n] >= 0.2 or self.happiness[n] >= 0.5:

				inrange = np.where(prox[n] == True)[0]

				for k in range(len(inrange)):

						# Loop through neighbours and broadcast opinion

						# *** Ignore agents with the same opnion 
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
				index = 0
				for i in range(len(inrange)):
					if self.happiness[inrange[i]] > max_opinion and np.random.uniform(0,1) <= self.comm_prob:

						index = inrange[i]
						max_opinion = self.happiness[inrange[i]]


				############# Update measure based on distance and time ##############

				# have far have they moved from previous position?
				dist_happy = cdist([self.previous_state[n]], [self.agents[n]])[0][0]/self.opinion_timelimit[n]

				max_val = self.speed

				dist_happy = dist_happy/max_val

				neighbour_num = len(inrange)
				max_neighbour = 10
				if neighbour_num > max_neighbour:
					neighbour_num = 9

				
				# Add noise to happiness measures

				############# Happiness with collision detection

				coverage_prop = (self.objective_count[n]+1)/self.opinion_timelimit[n]
				if coverage_prop <= 0: coverage_prop = 0.01

				self.happiness[n] = ((coverage_prop + 2*dist_happy)/3)*((max_neighbour - neighbour_num)/max_neighbour)

				# print('\n\n RESET COLLISION COUNTER FROM %d' % (self.collision_count[n]))


				# print('Agent %d collision count is %d' % (n, self.collision_count[n]))

				# self.collision_count[n] = 0

				# print('Agent %d collision count is %d AFTER RESET' % (n, self.collision_count[n]))


				self.happiness[n] += self.happiness_noise[n]

				

				# first check own performance

				if self.happiness[n] - self.prev_happiness[n] <= 0.2 and self.happiness[n] <= 0.5:

					#Something needs to change!

					self.prev_happiness[n] = self.happiness[n]

					# Compare best opinion with own opinion

					# Only change if they're opinion is different to yours

					if max_opinion >= opinion_buffer*self.happiness[n] and self.behaviour[n] != self.behaviour[index]:

						# **** SWITCH TO BEHAVIOUR OF NEIGHBOUR WITH HIGHER PERFORMANCE

						self.behaviour[n] = self.behaviour[index]
					else:

						# If no one else is performing better than you, try random behaviour


						options = [0,1,2,3,4,5,6,7,8]

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
							options = [0,1,2,3,4,5,6,7,8]
							#options = [0,10,11,9]
							self.behaviour[n] = np.random.choice(options)

					# All neighbours take on chosen behaviour
					for k in range(len(inrange)):
						if np.random.uniform(0,1) <= 0.5 and np.random.uniform(0,1) <= self.comm_prob:
							self.behaviour[inrange[k]] = self.behaviour[n]
							self.opinion_timer[inrange[k]] = 0

				self.objective_count[n] = 0

				self.previous_state[n] = self.agents[n]

				# Reset agent timer
				# *** If happiness is low check opinion again more quickly
				self.opinion_timelimit[n] = (1+self.happiness[n])*np.random.randint(self.update_rate-2 , self.update_rate+2)

		
					# Increment longterm timer
		self.longterm_counter = self.longterm_counter + np.ones(self.size)
	

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



def north(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	states = swarm.behaviour == 1

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#A = asim.fieldmap_avoidance(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	

	#B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	a += A - vector + noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W

def south(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	states = swarm.behaviour == 2

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#A = asim.fieldmap_avoidance(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	

	#B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	a += A - vector + noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W

def east(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	states = swarm.behaviour == 3

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#A = asim.fieldmap_avoidance(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	

	#B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	a += A - vector + noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W

def west(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	states = swarm.behaviour == 4

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#A = asim.fieldmap_avoidance(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	

	#B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	a += A - vector + noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W

def northwest(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	states = swarm.behaviour == 5

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	
	#A = asim.fieldmap_avoidance(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	

	#B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	a += A - vector + noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W

def northeast(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	states = swarm.behaviour == 6

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#A = asim.fieldmap_avoidance(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	

	#B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	a += A - vector + noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W

def southeast(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	states = swarm.behaviour == 7

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#A = asim.fieldmap_avoidance(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	

	#B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	a += A - vector + noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W

def southwest(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	states = swarm.behaviour == 8

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#A = asim.fieldmap_avoidance(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	

	#B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	a += A - vector + noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W

def dispersion(swarm, vector, param, noise):

	R = param; r = 2; A = 1; a = 20

	states = swarm.behaviour == 3

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	#A = asim.fieldmap_avoidance(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	

	#B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	a += A - vector + noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W

def sweep_LR(swarm, time, param, noise):

	R = param; r = 2; A = 1; a = 20

	# Create a sinusoidal direction vector which shifts over time

	periodx = 200
	periody = 20

	magx = np.sin( ((time - periodx)/periodx)*2*math.pi )
	magy = np.sin( ((time - (periody))/periody)*2*math.pi )

	vector = np.array([magx, magy])
	#print('Sweep vector = ', vector)

	states = swarm.behaviour == 5

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	A = asim.avoidance(swarm.agents, swarm.map)
	#now = time.time()
	#A = asim.fieldmap_avoidance(swarm)
	
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	#print('\nDispersion force on agent 1: (%.2f, %.2f)' % (a[0][0], a[0][1]))
	
	a += A - vector + 3*noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#swarm.headings = states*angles + np.logical_not(states)*swarm.headings 
	swarm.headings = states*np.arctan2(vector[1], vector[0]) + np.logical_not(states)*swarm.headings


	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W 

def sweep_UD(swarm, time, param, noise):

	R = param; r = 2; A = 1; a = 20

	# Create a sinusoidal direction vector which shifts over time

	periodx = 20
	periody = 200

	magx = np.sin( ((time - periodx)/periodx)*2*math.pi )
	magy = np.sin( ((time - (periody))/periody)*2*math.pi )

	vector = np.array([magx, magy])
	#print('Sweep vector = ', vector)

	states = swarm.behaviour == 6

	# Compute euclidean distance between agents
	mag = cdist(swarm.agents, swarm.agents)

	# Compute vectors between agents
	diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:] 

	A = asim.avoidance(swarm.agents, swarm.map)
	#A = asim.fieldmap_avoidance(swarm)
	#B = beacon(swarm)
	
	a = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)	
	a = np.sum(a, axis = 0).T

	a += A - vector + 3*noise
	
	vecx = a.T[0]
	vecy = a.T[1]
	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	#swarm.headings = (2*np.pi + angles) * (angles < 0) + angles*(angles > 0)
	# swarm.headings = states*np.random.uniform(-3.14, 3.14, swarm.size) + np.logical_not(states)*swarm.headings 

	swarm.headings = states*np.arctan2(vector[1], vector[0]) + np.logical_not(states)*swarm.headings



	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = -np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W




def roundabout(swarm, repel, attract, comm_range, align, noise, time):

	argname = ['repel', 'attract', 'align']
	args = [repel, attract, align]
	for n in range(len(args)):
		if args[n] > 1 or args[n] < 0:
			raise ValueError("Value %s must be within the range of 0 to 1." % (argname[n]))

	R = repel; r = 3; A = attract; a = 3


	period = 100

	magx = np.cos( ((time - period)/period)*2*math.pi )
	magy = np.sin( ((time - period)/period)*2*math.pi )

	vector = np.array([magx, magy])

	states = swarm.behaviour == 7

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

	# Add attraction to beacon
	# mag = cdist(swarm.beh_beacons.positions, swarm.agents)

	# for i in range(swarm.beh_beacons.positions):

	# 	if swarm.beh_beacons.behaviour_type == 7:




	total = 0
	# total +=  noise + repel + G - attract - avoid
	
	direction = np.ones((swarm.size, 2))

	total +=  G - repel + attract - avoid + vector
	

	vecx = total.T[0]
	vecy = total.T[1]
	angles = np.arctan2(vecy, vecx)

	swarm.headings = states*((2*np.pi + angles) * (angles < 0) + angles*(angles > 0)) + np.logical_not(states)*swarm.headings 

	swarm.headings = states*angles + np.logical_not(states)*swarm.headings 

	Wx = swarm.speed*np.cos(swarm.headings)
	Wy = swarm.speed*np.sin(swarm.headings)


	#W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])
	W = np.stack((Wx, Wy), axis = 1)
	swarm.agents += states[:,np.newaxis]*W
	
	

def random_walk(swarm, param):

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
	A = asim.avoidance(swarm.agents, swarm.map)
	a += A + G 

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.stack((Wx, Wy), axis = 1)

	# Move only agents with random behaviour set
	states = swarm.behaviour == 0
	swarm.agents += states[:,np.newaxis]*W

	swarm.agents = asim.continuous_boundary(swarm.agents, swarm.map)

def rotate_clock(swarm, direction, param):

	noise = param*np.random.randint(-2, 2, swarm.size)
	
	states = swarm.behaviour == 1

	swarm.headings += states*noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	a = np.zeros((swarm.size,2))
	B = np.zeros((swarm.size, 2))
	#B = asim.beacon(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	a += G + A + B

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])

	# Move only agents with random behaviour set
	states = swarm.behaviour == 1

	swarm.agents += states[:,np.newaxis]*W 
	swarm.agents = asim.continuous_boundary(swarm.agents, swarm.map)

def rotate_anti(swarm, direction, param):

	noise = param*np.random.randint(-1, 3, swarm.size)
	
	states = swarm.behaviour == 2

	swarm.headings += states*noise

	# Calculate new heading vector
	gx = 1*np.cos(swarm.headings)
	gy = 1*np.sin(swarm.headings)
	G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

	a = np.zeros((swarm.size,2))
	B = np.zeros((swarm.size, 2))
	#B = asim.beacon(swarm)
	A = asim.avoidance(swarm.agents, swarm.map)
	a += G + A + B

	vecx = a.T[0]
	vecy = a.T[1]

	angles = np.arctan2(vecy, vecx)
	Wx = swarm.speed*np.cos(angles)
	Wy = swarm.speed*np.sin(angles)

	W = -np.array([[Wx[n], Wy[n]] for n in range(0, swarm.size)])

	# Move only agents with random behaviour set
	states = swarm.behaviour == 2

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


