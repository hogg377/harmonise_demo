from operator import truediv

from pygame.math import enable_swizzling
from model.Agent import Agent
import numpy as np
import logging, pygame, sys
import colours as colours
from model.GridEmpowerment import createTreeOfMoves

class Dog(Agent):

    last_tick = -1
    voronoi_partitions = None

    def __init__(self, position, type) -> None:
        super().__init__(position, type)

        #initialise vectors to hold the features and actions
        self.m_feature_vec = np.array([np.nan, np.nan])
        self.m_state_vec = 'searching'
        self.m_direction_vector = np.zeros((3,2))
        self.m_empowerment = 0

        #initialise a count of how many ticks the dog hasn't moved
        self.m_n_stationary_tick = 0

        #important features related to targetting
        self.m_target_sheep_position_vec = np.empty((1,2))
        self.m_target_sheep_value = 0
        self.m_target_id = 0
        self.m_target_starting_disitance_2_com = 0

        #parameters to be set by the config
        self.m_endgoal_position_vec = np.array([0,0])
        self.m_home_position = np.array([0,0])
        self.m_nn_input_layer_size = 1
        self.m_nn_hidden_layer_size = 1
        self.m_nn_output_layer_size = 1
        self.m_nn_input_scaling = 1
        self.m_n_targeting_params = 1
        self.m_n_stationary_tick_limit = 10
        self.m_empowerment_sum_method = 'transition'
        self.m_empowerement_task_weighted_b = False
        self.m_empowerment_horizon = 2
        self.m_calculate_empowerment_b = True
        self.m_show_empowerment_b = True

        #assumes a feedforward network and create a dummy genome
        genome_length = self.m_n_targeting_params + (self.m_nn_input_layer_size + 1) * self.m_nn_hidden_layer_size + (self.m_nn_hidden_layer_size + 1) * self.m_nn_output_layer_size
        self.m_genome = np.ones((1,genome_length))

        #used for setting the images
        self.m_sprite_images = ['images\green_boat.png', 'images\yellow_boat.png', 'images\orange_boat.png', r'images\red_boat.png']
        self.m_empowerment_thresholds = [5,10,15,20]
    #end function


    def setConfig(self, cfg):
        super().setConfig(cfg)
        #TODO: convert the input cfg from a structure used by matlab into a python friendly form
        # and update this code
        self.m_endgoal_position_vec = cfg['endgoal_position']
        self.m_home_position = cfg['home_position']
        self.m_nn_input_layer_size = cfg['nn']['input_layer_size']
        self.m_nn_hidden_layer_size = cfg['nn']['hidden_layer_size']
        self.m_nn_output_layer_size = cfg['nn']['output_layer_size']
        self.m_n_stationary_tick_limit = cfg['stationary_tick_limit']
        self.m_n_targeting_params = cfg['n_valueweights']
        #scaling needs to be 2d because it multiplies elementwise with a 2d matrix
        self.m_nn_input_scaling = np.asarray(cfg['nn']['input_scaling'])
        self.setGenome(cfg['genome'])
        self.m_empowerment_sum_method = cfg['empowerment_method']
        self.m_empowerment_horizon = cfg['empowerment_horizon']
        self.m_calculate_empowerment_b = cfg['calculate_empowerment_b']
        self.m_show_empowerment_b = cfg['show_empowerment_b']
        self.m_empowerement_task_weighted_b = cfg['use_taskweighted_empowerment_b']
    #end function


    def update(self, world):
        '''
        update: main funciton to run once per tick.
        Causes the agent to update its state and take actions
        '''

        #update sensors (note this is usually more efficient to do once at a population level and pass the relevant results into update())
        xy_positions_vec, agent_ids, agent_types = world.entitiesInRange(self.m_position, self.m_sensor_range, b_include_position = False)

        #update internal belief
        #TODO: add functions for internal belief, only needed in fullfat version

        #calculate the dog's current empowerement at the start of it's turn
        if self.m_calculate_empowerment_b:
            self.m_empowerment = self.calcEmpowerment(world, use_task_weighted =self.m_empowerement_task_weighted_b)
        else:
            self.m_empowerment = 0

        #if the dog hasn't moved for a while, take an action to become unstuck
        if self.m_n_stationary_tick > self.m_n_stationary_tick_limit:
            #make a random move
            vec = self.randomMove(world, True)

        #elseif there are agents visible
        elif np.any(xy_positions_vec):

            # react to other dogs regardless of my state
            idx_dog = world.isDogId(agent_ids)
            if np.any(idx_dog):
                # contribution from avoiding dogs (always do this regardless of state)
                #TODO add code if needed
                dog_positions = xy_positions_vec[idx_dog,:]
                accel_dogs = np.array((0,0))
            else:
                dog_positions =[]
                accel_dogs = np.array((0,0))

            # react to sheep (reaction is state dependent)
            idx_sheep = np.logical_not(idx_dog)
            if np.any(idx_sheep):
                sheep_positions = xy_positions_vec[idx_sheep, :]
                sheep_ids = agent_ids[idx_sheep]

                #calculate features
                # returns: my_state, flock_com, weighted_flock_com, furthest_sheep_position, distance_2_closest_sheep
                my_state, flock_global_com, weight_flock_com, furthest_sheep_position, distance_2_closest_sheep, positions_sheep_in_my_partition, ids_sheep_in_partition  = self.calculateFlockFeatures(world, sheep_positions, sheep_ids, dog_positions)
                self.m_state_vec = my_state

                #identify which sheep to target
                target_sheep_pos, target_id, target_dist2com = self.selectTargetSheep(sheep_positions, sheep_ids, ids_sheep_in_partition, flock_global_com)
                self.m_target_id = target_id
                self.m_target_starting_disitance_2_com = target_dist2com

                #calculate the steering point as the grid square needed to target the sheep and move it in the correct direction for driving/collecting
                p_steer = self.selectSteeringPoint(world, self.m_position, flock_global_com, self.m_endgoal_position_vec, my_state, target_sheep_pos)

                #convert the steering point to a waypoint (the next actual square the dog will try to reach)
                #the waypoint is esssentially the steering point but with hysteresis to prevent rapid switching
                if my_state == 'collecting':
                    p_waypoint = self.selectWayPoint(p_steer)
                else:
                    p_waypoint = p_steer

                #decide the next grid square to move to with the aim of reaching the waypoint
                if my_state == 'searching':
                    #TODO: don't think this is ever reachable?  Since if the dog is searching it can't see a sheep which means it wouldn't have reached here
                    accel_herding = self.randomMove(world)
                else:
                    #state is 'collecting', 'driving' or 'go_home'

                    #test if the dog is too close to the sheep
                    #TODO: add code if needed, with a grid world the dog is adjacent to a sheep to move it so there isn't a notion of "too close"

                    #calculate the contribution from herding the sheep
                    accel_herding = self.herdingBehaviour(self.m_position, sheep_positions,  flock_global_com, weight_flock_com, target_sheep_pos, p_waypoint)


            #else dog can't see any sheep so move randomly
            else:
                accel_herding = self.randomMove(world)

            #add noise to heading
            #TODO: add code if needed, grid world may not need noise although need to ensure the "random step if stuck" function still runs in this new heuristic version

            #do the vector sum for the behaviours to: avoid dogs, avoid the wrong sheep, approach the waypoint and move radially around the flock
            vec = accel_dogs + accel_herding

        #can't see any other agents so move randomly
        else:
            vec = self.randomMove(world)

        #quantise the movement direction into a compass direction and update where to move next
        move = self.vector2CompassDirection(vec)[1]

        #try to move and test if succesful
        desired_next_position = self.m_position + move
        last_position = self.m_position
        new_position = self.move(world, desired_next_position)

        #if move wasn't succesful then count the ticks spent stationary
        if np.array_equal(last_position, new_position):
            self.m_n_stationary_tick += 1
        else:
            self.m_n_stationary_tick = 0

        #record the new position in the agents internal state
        # TODO: this may be better set outside the agent if any further checks are needed before the agent updates its state
        self.m_position = new_position

        #operations to tidy up at the end of an update
        self.m_adj_free_squares_is_current_b = False

        return new_position
    #end function


    def calcEmpowerment(self, world, sum_method = 'none', use_task_weighted = False):
        '''
        calcEmpowerment: returns an agents total empowerment.
        '''
        import model.GridEmpowerment as em
        from treelib import Tree
        move_horizon = self.m_empowerment_horizon
        #if no method provided for calculating empowerment then use the dog's internal parameter
        if sum_method=='none':
            sum_method = self.m_empowerment_sum_method
        #Perform an exhasutive search and return a tree structure of all possible 8 connected moves from the dog's current position
        #For each move, model the action of the sheep and store the predicted state
        tree = em.calcMovementTree(world, self.m_position, move_horizon)
        total_empowerment = em.sumEmpowerment(tree, sum_method, task_weight_empowerment_b=use_task_weighted, goal_location=self.m_endgoal_position_vec)
        return total_empowerment
    #end function


    # #genome not needed in the heuristic version
    # def parseGenome(self,genome):
    #     #segments the genome into parameters used in various parts of the controller.
    #     #TODO: set the limits from the config rather than hardcoding them here
    #     targeting_weights = genome[0:4]
    #     nn_weights = genome[4:]
    #     parsed_genome = np.concatenate((targeting_weights, nn_weights))
    #     return (targeting_weights, nn_weights)
    #  #end function


    #not sure what this was being used for.....
    # def steeringPoint(self, goal_position_vec, sheep_position_vec):
    #     #calculate the vector describing the line collinear with the goal and sheep
    #     #TODO: Add code
    #     #calculate vector from the sheep to the goal
    #     steering_direction_vec = goal_position_vec - sheep_position_vec
    #     #convert to a 8 point compass direction
    #     #[~, steering_direction] = vector2CompassDirection(-steering_direction);

    #     #return the resultant square relative to the sheep
    #     steering_point_vec = sheep_position_vec + steering_direction_vec
    #     return steering_direction_vec
    # #end function

    # # nn not needed in the heuristic version
    # def nnOutput(self, nn_inputs):
    #     #round the nn_inputs to 2 dp, runs them through network and returns the output
    #     import model.feedForwardNn as ff
    #     nn_inputs = np.round(nn_inputs,decimals=2)
    #     nn_weights = self.parseGenome(self.m_genome)[1]
    #     #scale the inputs
    #     nn_inputs = self.m_nn_input_scaling * nn_inputs
    #     K = ff.nnForwardPropergation(nn_weights, self.m_nn_input_layer_size, self.m_nn_hidden_layer_size,
    #     self.m_nn_output_layer_size, nn_inputs)
    #     return K
    # #end function


    def setGenome(self, genome):
        self.m_genome = np.asarray(genome)
    #end function


    #List of actions are:
    #update the sheep target ->returns the sheep target ID, position, target_dist2com and whether it's changed
    #choose a steering point based on the sheep target and dog state
    #pick a waypoint (could be the steering point)
    #calculate the accelerations:
    #   Acceleration towards waypoint
    #   Acceleration in response to sheep
    #   Acceleration due to dogs
    #


    def selectTargetSheep(self, sheep_positions, sheep_ids, ids_sheep_in_partition, flock_global_com):
        #Based on a test condition, this function checks to see if the
        #curret sheep should continue to be a target.  If it shouldn't a
        #new turtle is returned, if it should then the current ID is
        #returned.
        #TODO pass in all sheep positions and the indicies for partition sheep, then if there's none available we can handle it!!

        #make sure inputs are arrays
        sheep_positions = np.atleast_2d(sheep_positions)
        sheep_ids = np.array(sheep_ids)
        flock_global_com = np.atleast_2d(flock_global_com)

        #handle the edge case where no sheep are visible
        if not np.any(sheep_positions):
            target_pos = []
            target_id = 0
            target_dist2com = 0
            return target_pos, target_id, target_dist2com

        #there are sheep visible
        #decide if a new target is needed
        if self.m_target_id==0:
            #it doesn't have a target so aquire one
            new_target_b = True
        else:
            #it does have a target sheep
            idx_current_target = sheep_ids == self.m_target_id
            #if the current target isn't visible then aquire a new target
            if not np.any(idx_current_target):
                new_target_b = True
            #if current target is visible test if it still meets the condition for being a target
            else:
                b, target_dist2com = self.isNewTargetNeeded(sheep_positions[idx_current_target,:], flock_global_com)
                if b:
                    new_target_b = True
                else:
                    #if no new target is needed then
                    new_target_b = False

        #handle what to do if a new target it needed
        if new_target_b:
            #if there's a sheep in its partition to target
            if np.any(ids_sheep_in_partition):
                #find the indicies representing the sheep in my partition
                idx_sheep_in_partition = np.zeros([sheep_ids.shape[0]], dtype='bool')
                for i, id in enumerate(sheep_ids):
                    if np.any(ids_sheep_in_partition==id):
                        idx_sheep_in_partition[i]=1
                if self.m_state_vec == 'driving':
                    target_id, target_pos, target_dist2com = self.findFurthestSheep(sheep_positions[idx_sheep_in_partition,:], sheep_ids[idx_sheep_in_partition], self.m_endgoal_position_vec)
                else:
                    target_id, target_pos, target_dist2com = self.findFurthestSheep(sheep_positions[idx_sheep_in_partition,:], sheep_ids[idx_sheep_in_partition], flock_global_com)
            #if there's no sheep in partition but it can see a sheep
            elif np.any(sheep_positions):
                #new target becomes the closest sheep
                d = np.linalg.norm(sheep_positions-self.m_position,axis=1)
                idx = np.argmin(d)
                target_pos = sheep_positions[idx,:]
                target_id = sheep_ids[idx]
                target_dist2com = d[idx]
            #if it can't see any sheep
            else:
                #panic, function should have handled this earlier!
                target_pos = []
                target_id = 0
                target_dist2com = 0

        #handle what to do if we can continue to use the current target
        else:
            target_id = self.m_target_id
            target_pos = sheep_positions[idx_current_target,:]
            target_dist2com = self.m_target_starting_disitance_2_com

        #return (target_id, target_pos, target_dist2com, new_target_b)
        return target_pos, target_id, target_dist2com
    #end function


    def findFurthestSheep(self, positions, ids, com):
        #Finds the next sheep to target for collecting

        #ensure the inputs are 2d so the indexing works
        positions = np.atleast_2d(positions)
        ids = np.atleast_2d(ids)
        #if ids is a column vector then it needs converting to row vector to index ids correctly
        if ids.shape[1]>ids.shape[0]:
            ids = ids.T

        #target sheep is the one furthest from the com
        distances2com = np.linalg.norm(positions-com, axis=1, keepdims=True)
        idx_max = np.argmax(distances2com)
        logging.debug(f'indicies for the target ids is {idx_max}')
        target_id = ids[idx_max,:]
        target_pos = positions[idx_max,:]
        target_distance_2_com = distances2com[idx_max,:]

        return (target_id.item(), target_pos, target_distance_2_com.item())
    # end function


    def isNewTargetNeeded(self, target_position, com):
        # if the target sheep has moved over 50% of it's initial
        # distance to the flock CoM or the target sheep is within a threshold distance of the goal
        # then choose a new target

        #TODO: move minimum distance to goal to the config or make it dynamic based on the #of sheep in flock
        MIN_DISTANCE2GOAL = 10
        distance2com = np.linalg.norm(target_position - com)
        distance2goal = np.linalg.norm(target_position - self.m_endgoal_position_vec)
        if (distance2com.item(0) < (0.5 * self.m_target_starting_disitance_2_com)) or (distance2goal.item(0) < MIN_DISTANCE2GOAL):
            b = True
        else:
            b = False
        return (b, distance2com.item(0))
    #end function


    def selectSteeringPoint(self, world, my_position, flock_global_com, flock_goal_position, my_state, target_sheep_position):
        #finds the next location from which to steer the flock
        #returns P_steer and heading_at_P
        #TODO: This needs converting from continous space to grid world
        #TODO: This needs a feature vector or similar structure to pass instead of all these parameters!!!
        #TODO: sheep_positions can be removed as an input if the function to move the driving position to the convex hull isn't needed
        #TODO: this may need to return the heading to decide which direction to approach the steering point from

        N=50
        max_distance_from_goal = 15

        #handle the case where there isn't a target position provided
        if not np.any(target_sheep_position):
            P_steer = []
            return P_steer

        #if too close to sheep then don't move but keep the current heading
        if my_state=="tooclose":
            P_steer = my_position
            #heading_at_P = np.squeeze(my_velocity)
            #heading_at_P = self.unitVec(heading_at_P)

        #if collecting, the steering point is a distance behind the target on a line collinear with the target at the flock's GLOBAL com
        elif my_state == "collecting":
            #H = np.asarray(target_sheep_position) - np.asarray(flock_com)
            #H = self.unitVec(H)
            # P_collecting = target_sheep_position + self.COLLECTING_DISTANCE * H
            p_collecting = self.steeringPoint(flock_global_com, target_sheep_position)
            P_steer = p_collecting
            #heading_at_P = -H

        #if the sheep are close enough to the goal then set a target of the home position
        elif my_state=="go_home":
            #H = np.asarray(self.m_home_position) - my_position
            #H = self.unitVec(H)
            P_steer = self.m_home_position

        #if driving, the steering point is a distance behind the target on a line collinear with the flock COM at the goal position
        elif my_state=="driving":
            #H = flock_com - flock_goal_position
            #H = self.unitVec(H)
            #P_driving = flock_com + self.drivingDistance(N) * H
            p_driving = self.steeringPoint(flock_goal_position, target_sheep_position)

            #potentially, the driving position can be in the middle of the flock.
            #the matlab has a fancy bit of coding which finds the convex hull of the flock and projects the driving position on to it
            #lets see how badly it performs without this for now!!!

            P_steer = p_driving
            #heading_at_P = -H

        #the dog state is searching
        else:
            H = self.randomMove(world)
            P_steer = my_position + H
            #heading_at_P = H

        #quantise the steering point to a whole number (grid coordinate) and heading to a compass direction
        P_steer = np.round(P_steer)
        #heading_at_P = self.vector2CompassDirection(heading_at_P)[1]

        return (P_steer)
    #end function


    def drivingDistance(self, N):
        #returns the distance a dog should be from the flock com when driving it (as a function of the #of sheep)
        return 6*(N**0.5)
    #end function


    def unitVec(self, v):
        #returns a vector with the same direction as v but magnitude 1
        v = np.asarray(v)
        #ensure no devide by zero
        if np.any(v):
            v_hat = v / np.linalg.norm(v)
        else:
            v_hat=v
        return v_hat
    #end function


    def selectWayPoint(self, p_steer):
        #prevents the steering point from rapidly switching
        #TODO: Add code to this if needed
        p_steer_hyst = p_steer
        return p_steer_hyst
    #end function


    def herdingBehaviour(self, my_position, sheep_positions, flock_com, weighted_local_com, target_position, way_point):
        '''
        herdingBehaviour: returns an acceleration resuling from the agents interaction
        with the sheep and the current obj.waypoint
        '''

        PI = 3.14159265359

        #Create a bunch of direction vectors to move along
        ##################################################
        distance2sheep = np.linalg.norm(sheep_positions - self.m_position, axis=1)
        repsulsion_from_sheep_dir = np.sum(distance2sheep)
        repsulsion_from_sheep_dir = self.unitVec(repsulsion_from_sheep_dir)
        #calculate the radial component as the perpendicular vector to the sheep to dog direction
        radial2com_dir = self.rotate2D(self.m_position - target_position, 90)
        radial2com_dir = self.unitVec(radial2com_dir)
        #calculate the component towards the waypoint
        mypos2waypoint_dir = self.unitVec(np.asarray(way_point) - self.m_position)

        #calculate the error between the approach direction the dog wants and what it needs
        ####################################################
        mypos2target =  np.squeeze(np.asarray(target_position) - self.m_position)
        targetpos2com = np.squeeze(np.asarray(flock_com) - np.asarray(target_position))
        #th1,r1 = self.cart2pol(mypos2target[0], mypos2target[1])
        #th2,r2 = self.cart2pol(targetpos2com[0], targetpos2com[1])
        #com_theta_error = th1 - th2
        #pass the angular error values through the cosineStep function to create an error signal in the range[-1, +1]
        #com_theta_error = self.cosineStep(com_theta_error+PI/2)
        # add a fix, if the dog is on the waypont then the error signal is zero (this is needed mainly to account of quantisation error of the grid square vs true waypoint)
        if np.array_equal(self.m_position,way_point):
            com_theta_error = 0
        else:
            com_theta_error = self.cosineStepGrid(mypos2target,targetpos2com)

        #Calculate theta in the range 0 to 180
        import math as m
        logging.debug(f'position set to {my_position} and com set to {weighted_local_com}')
        com2mypos = np.squeeze(np.asarray(my_position) - np.asarray(weighted_local_com))
        com2wp = np.squeeze(np.asarray(way_point) - np.asarray(flock_com))
        #needs a hack to prevent the denominator ever being zero
        a = max([np.linalg.norm(com2mypos), 0.00001])
        b = max([np.linalg.norm(com2wp), 0.00001])
        #hack to prevent rounding errors causing a value greater than 1 or less than -1
        dotab = np.dot(com2mypos,com2wp)
        # dotab = min([dotab, +1])
        # dotab = max([dotab, -1])
        theta = m.acos(max(min(dotab / (a*b), +1), -1))
        #Calculate the contribution of the sheep repulsion
        sheep_pc = 1 - m.exp(-2*abs(theta))
        #TODO: remove disabling the repsulsion to sheep if not needed
        sheep_pc = 0

        #do a dumbed down version of the resultant acceleration which combines behaviours to avoid sheep, move towards way point and move cirularly around the target
        accel_hearding = sheep_pc *  repsulsion_from_sheep_dir + mypos2waypoint_dir + com_theta_error * radial2com_dir

        accel_hearding = self.unitVec(accel_hearding)
        return accel_hearding
    #end function


    # def selectTargetSheep(self, sheep_positions, sheep_ids, sheep_values):
    #     #takes a list of sheep_positions, calculates a value for each and then selects which to target
    #     #returns a tuple (target_sheep, target_sheep_id, max_sheep_value)

    #     #find the maximum sheep value and the element number it appears in
    #     idx = np.argmax(sheep_values)
    #     sheep_positions = np.atleast_2d(sheep_positions)
    #     target_sheep = sheep_positions[idx,:]
    #     target_sheep_id = sheep_ids[idx]
    #     target_sheep_value = sheep_values[idx]

    #     #return the target_sheep and target_sheep_id
    #     return (target_sheep, target_sheep_id, target_sheep_value)
    # #end function


    def distanceToNearestAgent(self, position: list, id_visible_agents: list, pos_visible_agent: list) -> tuple:
        '''
        Calculate the distances between the agent at the given 'position' and each
        agent in visible_agents.

        Args:
            position: position of the agent of concern, e.g., a dog
            visible_agents: list of all agents visible to the current agent,
                e.g., a list of sheep visible to the dog
        '''

        relative_agent_positions = pos_visible_agent - position
        # Calculate the euclidean distance to each agent -- should this be euclidean? Or grid-move based?
        distance_to_agents = np.array(
            [[i, np.sqrt(xy[0]**2 + xy[1]**2)] for i, xy in zip(id_visible_agents, relative_agent_positions)]
        )
        distance_to_agents = distance_to_agents[np.argsort(distance_to_agents[:,1])]
        distance_to_agents = list(zip(map(int, distance_to_agents[:,0]), distance_to_agents[:,1]))

        # Get id and distance of the closest agent.
        closest_agent_id,  closest_agent_dist = distance_to_agents[0]

        return closest_agent_id,  closest_agent_dist, distance_to_agents


    def partitionVisibleAgents(
        self, world,
        position: list, other_positions: list,
        id_visible_agents: list, pos_visible_agents: list
    ) -> list:
        '''
        Split the visible flock up into voronoi partitions based on the locations of the other dogs

        The approach is to generate polygons which represent the voroni partitions based the space each dog will reach before the other dogs.

        Create a bounding box for the voronoi partition which atleast encompasses all the dogs and sheep
        '''
        import matplotlib.pyplot as plt
        from scipy.spatial import Voronoi, voronoi_plot_2d
        from model.VoronoiPolgyons import voronoi_finite_polygons_2d
        from shapely.geometry import Polygon, Point

        # The current dog's position is always at index 0, to retrieve its polygon/sheep later.
        all_positions = np.vstack([position, other_positions])
        try:
            vor = Voronoi(all_positions)
        except:
            #if the voronoi fails (e.g. because points are coplanar, then return an empty partition)
            idx_sheep_in_partition = []
            positions_sheep_in_my_partition = []
            return idx_sheep_in_partition, positions_sheep_in_my_partition

        if world.tick > Dog.last_tick:
            logging.debug("Voronoi partition calc'd on tick {}".format(world.tick))
            Dog.voronoi_partitions = voronoi_finite_polygons_2d(vor)
            Dog.last_tick = world.tick

        regions, vertices = Dog.voronoi_partitions

        # If we want some additional padding of the outer polygons, to ensure that all sheep
        # are included, then we can pad the world boundaries here.

        world_bounds = (world.m_width, world.m_height)

        padding = 0.1
        min_x = 0 - padding
        max_x = world_bounds[0] + padding
        min_y = 0 - padding
        max_y = world_bounds[1] + padding

        mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
        bounded_vertices = np.max((vertices, mins), axis=0)
        maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
        bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

        box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
        polygons = []

        for region in regions:
            polygon = vertices[region]
            # Clipping polygon
            poly = Polygon(polygon)
            poly = poly.intersection(box)
            polygon = [p for p in poly.exterior.coords]

            polygons.append(polygon)

            # plt.fill(*zip(*polygon), alpha=0.4)

        # plt.plot(all_positions[:, 0], all_positions[:, 1], 'ko')
        # plt.axis('equal')
        # plt.xlim(vor.min_bound[0] - padding, vor.max_bound[0] + padding)
        # plt.ylim(vor.min_bound[1] - padding, vor.max_bound[1] + padding)
        # plt.show()

        idx_sheep_in_partition = []
        positions_sheep_in_my_partition = []
        for idx, pos in zip(id_visible_agents, pos_visible_agents):
            if Polygon(polygons[0]).contains(Point(pos)):
                idx_sheep_in_partition.append(idx)
                positions_sheep_in_my_partition.append(pos)

        return idx_sheep_in_partition, positions_sheep_in_my_partition


    def furthestSheepPosition(self, my_visible_sheep, flock_com):
        return np.max(np.array([np.linalg.norm(sheep - flock_com) for sheep in my_visible_sheep])) + flock_com


    def planState(self, my_visible_sheep, flock_com, distance2sheep):
        '''
        Generates features based on how spread out the flock.
        Assess how close it is to the sheep and how spread out the sheep are and
        take action because it is either too
            rmin: the minimum distance to the turtles
            fN: a function which generates a distance describing the
                minimum "radius" of the flock at which it can be driven
        '''

        # Default state
        my_state = 'searching'

        # The treshold distance from the goal to the peripheral sheep for the dog
        # to treat the task as complete
        max_distance_from_goal = 5
        max_distance_from_com = 0.5*len(my_visible_sheep)
        flock_goal_position = self.m_endgoal_position_vec

        # If no sheep are visible then dog is in a searching state
        n_neighbours = len(my_visible_sheep)
        if n_neighbours > 0:
            # If the sheep are close to the target position then go home
            if ( np.linalg.norm(flock_com - flock_goal_position) < max_distance_from_goal and
                np.max(np.linalg.norm(my_visible_sheep-flock_com,axis=1))<max_distance_from_com ):
                my_state = 'go_home'
            # The flock CoM is too far from the goal; I'm either collecting or driving the sheep
            else:
                max_distance = np.max(np.array([np.linalg.norm(sheep - flock_com) for sheep in my_visible_sheep]))

                # ?????
                # Add a bit of histeresis so the the flock needs to be more spread out
                # to switch from driving to collecting than from collecting to driving.
                # ?????
                if self.m_state_vec == 'driving':
                    x = 1.5# * n_neighbours
                else:
                    x = 1#n_neighbours

                # If the flock is sufficiently close together then start the drive.
                if max_distance < (max_distance_from_com*x): #(6 * x**(2/3) * 0.75):
                    # A dog only takes part in driving if its config allows it to
                    # if self.take_part_in_driving_flock:
                    #     my_state = 'driving'
                    # else:
                    #     my_state = 'go_home'
                    my_state = 'driving'
                else:
                    my_state = 'collecting'

        logging.debug(my_state)

        return my_state


    def calculateFlockFeatures(self, world, sheep_positions: list, sheep_ids: list, dog_positions: list):
        #in Matlab mode it's generateFeatures.  This top level function runs all the sub functions to
        #    1. update the dog's state
        #    2. generate and store features about the world which are needed in multiple places (helps efficiency)

        #   calculate the flock CoM based on all visible sheep
        #   flock_com = self.calcCoM(my_visible_sheep);

        # Calculate the Centre of Mass of the sheep by taking the (column-wise) average of the sheep_positions along 0th axis
        flock_com = np.average(sheep_positions, axis=0)

        #   calculate the distance to the nearest sheep
        #   (for efficiency, also return the index of the closest sheep (where it is in the list of my_visible_sheep)
        #       and a list of distances from the dog to each sheep in the flock)
        #   distance_2_closest_sheep, idx_closest_sheep, distance_to_sheeps = obj.distanceToNearestAgent(my_position, my_visible_sheep);
        idx_closest_sheep, distance_2_closest_sheep, distance_to_sheep = self.distanceToNearestAgent(
            self.m_position,
            sheep_ids,
            sheep_positions
        )

        #   if there are 3 dogs total
        #   find the sheep which are closest to my_position by performing a voronoi parition of space occupuied by the whole flock.
        #   positions_sheep_in_my_partition, idx_sheep_in_partition = obj.partitionVisibleAgents(dog_positions, my_visible_sheep);
        # dog_positions_inc_me = np.append(dog_positions, np.atleast_2d(self.m_position), axis=0)
        if len(dog_positions)>=2:
            idx_sheep_in_partition, positions_sheep_in_my_partition = self.partitionVisibleAgents(
                world,
                self.m_position,
                dog_positions,
                sheep_ids,
                sheep_positions
            )
        #   if there are 2 dogs
        #   divide up the flock based on which sheep are closest to which dog
        elif len(dog_positions)==1:
            #   TODO: find a faster method to parition the flock between 2 dogs
            #   APPROACH 1: split the flock in half and take those furthest from me
            #   (doesn't work and dogs eventually converge on the same target, fudge factor based on an agent it would work
            #   except adding and removing dogs means it's possible all dogs have odd or even ids!)
            # n_half_population = round(len(sheep_positions) / 2)
            # idx_sheep_in_partition = []
            # positions_sheep_in_my_partition = []
            # #extract the ids of the furthest half of the population
            # if self.m_id % 2==1:
            #     fudge_factor = 0
            # else:
            #     fudge_factor = 2
            # for id,dist in reversed(distance_to_sheep[(-n_half_population+fudge_factor):]):
            #         idx_sheep_in_partition.append(id)
            # #extract the position of the sheep matching the id
            # for i, id in enumerate(sheep_ids):
            #     if np.any(np.equal(idx_sheep_in_partition, id)):
            #         positions_sheep_in_my_partition.append(sheep_positions[i])

            # APPROACH 2: use brute force.  Calculate the distance to each sheep from myself and the other dog
            #   divide up the flock with each dog taking the sheep closest to it
            sheep_distances_to_other_dog = np.linalg.norm(dog_positions - sheep_positions, axis=1)
            sheep_distances_to_me = np.linalg.norm(self.m_position - sheep_positions, axis=1)
            idx_sheep_in_partition = []
            positions_sheep_in_my_partition = []
            for i,(dist2me,dist2other) in enumerate(zip(sheep_distances_to_me, sheep_distances_to_other_dog)):
                if dist2me<dist2other:
                    idx_sheep_in_partition.append(sheep_ids[i])
                    positions_sheep_in_my_partition.append(sheep_positions[i])
        # if there's only 1 dog (dog_positions is empty), then all sheep are in its partition
        else:
            idx_sheep_in_partition = sheep_ids
            positions_sheep_in_my_partition = sheep_positions

        #   calculate a weighted CoM.  This helps the dog to "forget" the main flock if it's collecting a very distant sheep.
            #This is important because otherwise the dog tries to circle around the global CoM when approaching the distant sheep which results in
            #the dog chasing the sheep rather than going around it.
        #    weighted_flock_com = obj.calculateWeightedLocalCoM(my_visible_sheep,distance_to_sheeps);
        if np.any(positions_sheep_in_my_partition):
            distances = np.linalg.norm(np.atleast_2d(positions_sheep_in_my_partition)-np.atleast_2d(self.m_position),axis=1, keepdims=True)
            weighted_flock_com = self.calculateWeightedLocalCoM(positions_sheep_in_my_partition,distances)


        else:
            #there's no sheep in the partition
            weighted_flock_com = flock_com

        #   calculate the dog's state (see planState)
        #   my_state = obj.planState(my_visible_sheep, flock_com, obj.flockCohehsionRadius);
        # my_state = 'go home'
        my_state = self.planState(sheep_positions, flock_com, distance_to_sheep)

        #   calculate which is the furthest sheep position of those in its voronoi partition
            #if there are no sheep in its partition then target the closest sheep it can see
        #   furthest_sheep_position = obj.furthestAgent(flock_com, sheeps_in_my_partition);
        if np.any(positions_sheep_in_my_partition):
            furthest_sheep_position = self.furthestSheepPosition(
                positions_sheep_in_my_partition,
                flock_com
            )
        else:
            furthest_sheep_position = []

        #The function returns a list called feature_state_vector with the following info:
        #   [FV_STATE, FV_LOCAL_COM, FV_FLOCK_COM, FV_FURTHEST_TURTLE_FROM_COM, FV_DISTANCE_TO_NEAREST_TURTLE]
        return my_state, flock_com, weighted_flock_com, furthest_sheep_position, distance_2_closest_sheep, positions_sheep_in_my_partition, idx_sheep_in_partition
    #end function


    # def calculateFlockFeatures(self, sheep_positions, target_position, target_id, target_value):
    #     #calculate the scalar values (features) that act as inputs to the controller
    #     #also calculate the direction vectors which form the action space of the agent
    #     PI = 3.14159265359

    #     sheep_positions = np.atleast_2d(sheep_positions)

    #     #calculate the CoM
    #     com_position = self.calcCoM(sheep_positions)

    #     #***calculate a bunch of vectors***
    #     #Make everything 2d so it's compatible with the matlab implementation
    #     vec_mypos2sheep = np.atleast_2d(target_position - self.m_position)
    #     vec_sheeppos2goal = np.atleast_2d(self.m_endgoal_position_vec - target_position)
    #     vec_sheeppos2com = np.atleast_2d(com_position - target_position)
    #     distance2target_sheep = np.linalg.norm(target_position - self.m_position)

    #     #***calculate a bunch of features used as inputs to the controller***
    #     #********************************************************************
    #     #calculate the cohesion of THE TARGET SHEEP ONLY as the RMS distance to its neighbours
    #     N = sheep_positions.shape[0]
    #     distance_sq = np.linalg.norm(sheep_positions - target_position,axis=1)**2
    #     target_sheep_cohesion = np.sqrt(1/N * sum(distance_sq))

    #     #calculate the angular Goal error (for driving) and CoM error (for collecting)
    #     th1,r1 = self.cart2pol(vec_mypos2sheep[0,0], vec_mypos2sheep[0,1])
    #     th2,r2 = self.cart2pol(vec_sheeppos2goal[0,0], vec_sheeppos2goal[0,1])
    #     goal_theta_error =th1 - th2

    #     th1,r1 = self.cart2pol(vec_mypos2sheep[0,0],vec_mypos2sheep[0,1])
    #     th2,r2 = self.cart2pol(vec_sheeppos2com[0,0],vec_sheeppos2com[0,1])
    #     com_theta_error = th1 - th2

    #     #pass the angular error values through the cosineStep function to create an easier signal for the NN
    #     goal_theta_error = self.cosineStep(goal_theta_error+PI/2)
    #     com_theta_error = self.cosineStep(com_theta_error+PI/2)

    #     #store the features in
    #     self.m_feature_vec = np.hstack((distance2target_sheep,
    #         target_sheep_cohesion,
    #         goal_theta_error,
    #         com_theta_error))

    #     self.m_target_sheep_position_vec = target_position
    #     self.m_target_sheep_value = target_value

    #     #****calculate the action space for the agent as a set of direction vectors it can move along****
    #     #************************************************************************************************
    #     #calculate the direction towards the sheep
    #     dog2sheep_vec = target_position - self.m_position

    #     #calculate the radial component as the perpendicular vector to the sheep to dog direction
    #     radsheep2dog_vec = self.rotate2D(self.m_position - target_position, 90)

    #     #calculate the direction towards the steering point used to move the target towards the goal (Driving)
    #     goal_sp = self.steeringPoint(self.m_endgoal_position_vec, target_position)
    #     dog2goalsp_vec = goal_sp - self.m_position

    #     #calculate the direction towards the steering point used to move the target towards the flock CoM (Collecting)
    #     com_sp = self.steeringPoint(com_position, target_position)
    #     dog2comsp_vec = com_sp - self.m_position

    #     #store the direction vectors
    #     self.m_direction_vector = np.vstack((dog2sheep_vec,
    #         radsheep2dog_vec,
    #         dog2goalsp_vec,
    #         dog2comsp_vec))
    # #end function


    def calculateWeightedLocalCoM(self, positions, distances2positions):
        # Weights the CoM calculating via the fomula (1 / d^2)
        # where d is the distance from the calculating agent to each position

        #base the local CoM only on sheep which are within
        #a distance R of the dog
        positions = np.atleast_2d(positions)
        distances2positions = np.asarray(distances2positions)
        R=80
        idx = distances2positions < R
        #If there are any sheep within R then calculate the
        #LocalCoM, if not, set it to empty
        if np.any(idx):
            #CoM = @(func, P,d) sum(P.*func(d),1) ./ (sum(func(d)));
            weighting = np.atleast_2d(1 / distances2positions[idx]**2)
            # frustratingly idx can be a shape 1,1 vector which needs two indicies to address
            # also need to transpose weighting because it defaults to a row vector and positions is nx2
            weight_CoM = np.sum(positions[idx[:,0],:] * weighting.T, axis=0) / np.sum(weighting)
        else:
            weight_CoM=[]
        return weight_CoM
    #end function


    # def calculateSheepValue(self, sheep_positions, sheep_ids, prob_sheep_my_types):
    #     #assigns a value to each sheep_position
    #     from numpy import linalg as LA
    #     from scipy.spatial import distance

    #     sheep_positions = np.atleast_2d(sheep_positions)

    #     #calculate the distance from each sheep to the dog
    #     sheep2dog_distances = LA.norm(sheep_positions - self.m_position, axis=1)

    #     #calculate the distance from each sheep to the goal
    #     sheep2goal_distances = LA.norm(self.m_endgoal_position_vec - sheep_positions, axis=1)

    #     #calculate the cohesion for each sheep as the RMS distance from the sheep to all other sheep in the population
    #     n_sheep = sheep_positions.shape[0]
    #     d = distance.pdist(sheep_positions, 'euclidean')
    #     dsquare = distance.squareform(d)
    #     sheep_cohesions = np.sqrt(1/n_sheep * np.sum(np.power(dsquare,2), axis=1))

    #     #extract the targetting weights from the genome
    #     w = self.parseGenome(self.m_genome)[0]

    #     #Add a scaling factor to try and keep all inputs in the range 0-1
    #     #and do a value calcuation for each sheep position
    #     #TODO: assign the world scale from the config
    #     world_scale = 1/60
    #     sheep_values = (w[0] * sheep2dog_distances * world_scale +
    #         w[1] * sheep2goal_distances * world_scale +
    #         w[2] * sheep_cohesions * world_scale +
    #         w[3] * prob_sheep_my_types)

    #     return sheep_values
    # #end function


    # def probabilitySheepAreMyType(self, sheep_ids):
    #     #calculate the probability that the sheep is my type
    #     #TODO: Not using this in the vanilla version but add code for the fullfat one

    #     n_sheep = np.size(sheep_ids)
    #     prob_sheeps_mytype_vec = np.ones((n_sheep,1))

    #     return prob_sheeps_mytype_vec
    # #end function


    def reset(self):
        #resets any observed values to their initial conditions

        #initialise vectors to hold the features and actions
        self.m_feature_vec = np.array([np.nan, np.nan])
        self.m_direction_vector = np.zeros((3,2))

        #initialise a count of how many ticks the dog hasn't moved
        self.m_n_stationary_tick = 0

        #important features related to targetting
        self.m_target_sheep_position_vec = np.empty((1,2))
        self.m_target_sheep_value = 0
    #end function


    def updateBeliefs(self, world, sheep_positions, sheep_ids):
        #updates the observations of sheep type using the internal model
        #TODO: not used for vanilla model
        pass
    #end function


    def estimateSheepType(self, world, last_position, current_position):
        #checks whether a sheep moved according to its internal model or not
        #TODO: not used for vanilla model
        pass
    #end function


    # def cosineStep(self, angles):
    #     #A radial magnitude function which is:
    #     #   cos(theta) for [0,+180] & [-360,-270]
    #     #   -1 for [+180,+270] & [-180,-90]
    #     #   +1 for [-90, 0] & [+270,+360]
    #     PI = 3.14159265359
    #     angles = np.asarray(angles)
    #     val = np.zeros(np.size(angles))

    #     for n, theta in enumerate(np.nditer(angles)):
    #         if (0<theta and theta<=PI) or (-2*PI<=theta and theta<=-PI):
    #             val[n] = np.cos(theta)
    #         elif (-PI/2<theta and theta<=0) or (3/2*PI<theta and theta<=2*PI):
    #             val[n] = +1
    #         #condition is (-180<theta and theta<=-90) or (+180<theta and theta<=+270)
    #         else:
    #             val[n] = -1
    #     return val
    # #end function

    def cosineStepGrid(self, vec1, vec2):
        # new way of calculating theta error.
        # the dot product guarnetees an angle between the two vectors
        # which smoothly varies between 0 and 180deg
        # use the sign of the cross product to assertain which direction
        # to rotate
            import math as m
            dotv = np.dot(vec1,vec2)
            if dotv==0:
                xd=0
            else:
                try:
                    xd = m.acos( dotv / (np.linalg.norm(vec1)*np.linalg.norm(vec2)) )
                except ValueError:
                    xd = 0

            xc = np.cross(np.append(vec1,0), np.append(vec2,0))
            return xd/m.pi * np.sign(xc[2])
    #end function


    def steeringPoint(self, goal_position, sheep_position):
        #calculate the vector describing the line collinear with the
        #goal and sheep
        steering_direction = goal_position - sheep_position
        #calculate the steering point as the point adjacent to the sheep, on the closest compass direction, to create a
        #co-linear line in the order steeringpoint-sheep-goal
        steering_direction = self.vector2CompassDirection(-steering_direction)[1]
        steering_point = sheep_position + steering_direction
        return steering_point
    #end function


    def updateSprite(self, scale_factor=0):
        #do any changes to the spite image
        if self.m_show_empowerment_b:
            i=0
            while i<len(self.m_empowerment_thresholds) and self.m_empowerment>self.m_empowerment_thresholds[i]:
                i+=1
            colour = colours.ERANGE[i]
        else:
            colour = colours.BLUE
            # img = self.loadImage(self.m_sprite_images[i], self.image.get_height())
        if scale_factor==0:
            scale_factor = self.image.get_width()
        img = pygame.Surface((scale_factor, scale_factor))
        img.fill(colour)
        self.setImage(img)
        super().updateSprite(scale_factor)
        return
    #end function

    # def updateSprite(self, scale_factor):
    #     """
    #     Dog over rides the parent class updateSprite function to enable it to display empowerment information
    #     """
    #     import pygame
    #     text = "1"
    #     size = 10
    #     color = [0,0,0]

    #     width = self.image.get_width()
    #     height = self.image.get_width()

    #     font = pygame.font.SysFont("Arial", size)
    #     textSurf = font.render(text, 1, color)
    #     #self.image = pygame.Surface((width, height))
    #     W = textSurf.get_width()
    #     H = textSurf.get_height()
    #     self.image.blit(textSurf, [width/2 - W/2, height/2 - H/2])

    #     #simple function which moves the agents image to align with its position in the screen coordinate frame
    #     pos_in_screen_coords = self.m_position * scale_factor
    #     #the screen draws by default as y = rows and x = columns so need to reverse it
    #     pos_in_screen_coords = np.flip(pos_in_screen_coords)
    #     self.setSpritePosition(pos_in_screen_coords)

    #     return
    # #end function















