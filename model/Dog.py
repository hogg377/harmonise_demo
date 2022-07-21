from model.Agent import Agent
import numpy as np
import sys, pygame
import colours as colours

class Dog(Agent):
    def __init__(self, position, type) -> None:
        super().__init__(position, type)

        #initialise vectors to hold the features and actions
        self.m_feature_vec = np.array([np.nan, np.nan])
        self.m_direction_vector = np.zeros((3,2))
        self.m_empowerment = 0

        #initialise a count of how many ticks the dog hasn't moved
        self.m_n_stationary_tick = 0

        #important features related to targetting
        self.m_target_sheep_position_vec = np.empty((1,2))
        self.m_target_sheep_value = 0

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
        self.m_empowerment_horizon = 2

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
    #end function


    def update(self, world):
        #main funciton to run once per tick.
        #causes agent to update its state and take actions

        #update sensors (note this is usually more efficient to do once at a population level and pass the relevant results into update())
        xy_positions_vec, agent_ids, agent_types = world.entitiesInRange(self.m_position, self.m_sensor_range, b_include_position = False)
        #update internal belief
        #TODO: add functions for internal belief, only needed in fullfat version

        #calculate the dog's current empowerement at the start of it's turn
        self.m_empowerment = self.calcEmpowerment(world)


        #if the dog hasn't moved for a while, take an action to become unstuck
        if self.m_n_stationary_tick > self.m_n_stationary_tick_limit:
            #make a random move
            move = self.randomMove(world, True)

        #elseif there are agents visible
        elif np.any(xy_positions_vec):
            idx_sheep = np.logical_not(world.isDogId(agent_ids))

            #if at least one visible agent is a sheep
            if np.any(idx_sheep):
                sheep_positions = xy_positions_vec[idx_sheep, :]
                sheep_ids = agent_ids[idx_sheep]
                #calculate the probability the sheep are my type
                #TODO: Add function for estimating probabilities
                prob_sheep_mytype = self.probabilitySheepAreMyType(sheep_ids)

                #value each sheep
                sheep_values = self.calculateSheepValue(sheep_positions, sheep_ids, prob_sheep_mytype)

                #identify which sheep to target
                target_sheep, target_sheep_id, max_sheep_value = self.selectTargetSheep(sheep_positions, sheep_ids, sheep_values)

                #calculate features
                self.calculateFlockFeatures(sheep_positions, target_sheep, target_sheep_id, max_sheep_value)

                #calculate the weghts for the directional vectors
                K = self.nnOutput(self.m_feature_vec)

                #convert the directional vectors to compass direction vectors
                compass_dirn_vectors = self.vector2CompassDirection(self.m_direction_vector)[1]

                #use matrix opperation to do: vec = K(1)*goal_component + K(2)*steeringpoint_component + K(3)*radial_component;
                vec = K @ compass_dirn_vectors

                #quantise the movement direction into a compass direction and update where to move next
                move = self.vector2CompassDirection(vec)[1]

            #else dog can't see any sheep so move randomly
            else:
                move = self.randomMove(world)

        #else dog can't see any sheep so move randomly
        else:
            move = self.randomMove(world)

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


    def calcEmpowerment(self, world, sum_method = 'none'):
        import model.GridEmpowerment as em
        from treelib import Tree
        move_horizon = self.m_empowerment_horizon
        #if no method provided for calculating empowerment then use the dog's internal parameter
        if sum_method=='none':
            sum_method = self.m_empowerment_sum_method
        #Perform an exhasutive search and return a tree structure of all possible 8 connected moves from the dog's current position
        #For each move, model the action of the sheep and store the predicted state
        tree = em.calcMovementTree(world, self.m_position, move_horizon)
        total_empowerment = em.sumEmpowerment(tree, sum_method)
        return total_empowerment


    def parseGenome(self,genome):
        #segments the genome into parameters used in various parts of the controller.
        #TODO: set the limits from the config rather than hardcoding them here
        targeting_weights = genome[0:4]
        nn_weights = genome[4:]
        parsed_genome = np.concatenate((targeting_weights, nn_weights))
        return (targeting_weights, nn_weights)
     #end function


    def steeringPoint(self, goal_position_vec, sheep_position_vec):
        #calculate the vector describing the line collinear with the goal and sheep
        #TODO: Add code
        #calculate vector from the sheep to the goal
        steering_direction_vec = goal_position_vec - sheep_position_vec
        #convert to a 8 point compass direction
        #[~, steering_direction] = vector2CompassDirection(-steering_direction);

        #return the resultant square relative to the sheep
        steering_point_vec = sheep_position_vec + steering_direction_vec
        return steering_direction_vec
    #end function


    def nnOutput(self, nn_inputs):
        #round the nn_inputs to 2 dp, runs them through network and returns the output
        import model.feedForwardNn as ff
        nn_inputs = np.round(nn_inputs,decimals=2)
        nn_weights = self.parseGenome(self.m_genome)[1]
        #scale the inputs
        nn_inputs = self.m_nn_input_scaling * nn_inputs
        K = ff.nnForwardPropergation(nn_weights, self.m_nn_input_layer_size, self.m_nn_hidden_layer_size,
        self.m_nn_output_layer_size, nn_inputs)
        return K
    #end function


    def setGenome(self, genome):
        self.m_genome = np.asarray(genome)
    #end function


    def selectTargetSheep(self, sheep_positions, sheep_ids, sheep_values):
        #takes a list of sheep_positions, calculates a value for each and then selects which to target
        #returns a tuple (target_sheep, target_sheep_id, max_sheep_value)

        #find the maximum sheep value and the element number it appears in
        idx = np.argmax(sheep_values)
        sheep_positions = np.atleast_2d(sheep_positions)
        target_sheep = sheep_positions[idx,:]
        target_sheep_id = sheep_ids[idx]
        target_sheep_value = sheep_values[idx]

        #return the target_sheep and target_sheep_id
        return (target_sheep, target_sheep_id, target_sheep_value)
    #end function


    def calculateFlockFeatures(self, sheep_positions, target_position, target_id, target_value):
        #calculate the scalar values (features) that act as inputs to the controller
        #also calculate the direction vectors which form the action space of the agent
        PI = 3.14159265359

        sheep_positions = np.atleast_2d(sheep_positions)

        #calculate the CoM
        com_position = self.calcCoM(sheep_positions)

        #***calculate a bunch of vectors***
        #Make everything 2d so it's compatible with the matlab implementation
        vec_mypos2sheep = np.atleast_2d(target_position - self.m_position)
        vec_sheeppos2goal = np.atleast_2d(self.m_endgoal_position_vec - target_position)
        vec_sheeppos2com = np.atleast_2d(com_position - target_position)
        distance2target_sheep = np.linalg.norm(target_position - self.m_position)

        #***calculate a bunch of features used as inputs to the controller***
        #********************************************************************
        #calculate the cohesion of THE TARGET SHEEP ONLY as the RMS distance to its neighbours
        N = sheep_positions.shape[0]
        distance_sq = np.linalg.norm(sheep_positions - target_position,axis=1)**2
        target_sheep_cohesion = np.sqrt(1/N * sum(distance_sq))

        #calculate the angular Goal error (for driving) and CoM error (for collecting)
        th1,r1 = self.cart2pol(vec_mypos2sheep[0,0], vec_mypos2sheep[0,1])
        th2,r2 = self.cart2pol(vec_sheeppos2goal[0,0], vec_sheeppos2goal[0,1])
        goal_theta_error =th1 - th2

        th1,r1 = self.cart2pol(vec_mypos2sheep[0,0],vec_mypos2sheep[0,1])
        th2,r2 = self.cart2pol(vec_sheeppos2com[0,0],vec_sheeppos2com[0,1])
        com_theta_error = th1 - th2

        #pass the angular error values through the cosineStep function to create an easier signal for the NN
        goal_theta_error = self.cosineStep(goal_theta_error+PI/2)
        com_theta_error = self.cosineStep(com_theta_error+PI/2)

        #store the features in
        self.m_feature_vec = np.hstack((distance2target_sheep,
            target_sheep_cohesion,
            goal_theta_error,
            com_theta_error))

        self.m_target_sheep_position_vec = target_position
        self.m_target_sheep_value = target_value

        #****calculate the action space for the agent as a set of direction vectors it can move along****
        #************************************************************************************************
        #calculate the direction towards the sheep
        dog2sheep_vec = target_position - self.m_position

        #calculate the radial component as the perpendicular vector to the sheep to dog direction
        radsheep2dog_vec = self.rotate2D(self.m_position - target_position, 90)

        #calculate the direction towards the steering point used to move the target towards the goal (Driving)
        goal_sp = self.steeringPoint(self.m_endgoal_position_vec, target_position)
        dog2goalsp_vec = goal_sp - self.m_position

        #calculate the direction towards the steering point used to move the target towards the flock CoM (Collecting)
        com_sp = self.steeringPoint(com_position, target_position)
        dog2comsp_vec = com_sp - self.m_position

        #store the direction vectors
        self.m_direction_vector = np.vstack((dog2sheep_vec,
            radsheep2dog_vec,
            dog2goalsp_vec,
            dog2comsp_vec))
    #end function


    def calculateSheepValue(self, sheep_positions, sheep_ids, prob_sheep_my_types):
        #assigns a value to each sheep_position
        from numpy import linalg as LA
        from scipy.spatial import distance

        sheep_positions = np.atleast_2d(sheep_positions)

        #calculate the distance from each sheep to the dog
        sheep2dog_distances = LA.norm(sheep_positions - self.m_position, axis=1)

        #calculate the distance from each sheep to the goal
        sheep2goal_distances = LA.norm(self.m_endgoal_position_vec - sheep_positions, axis=1)

        #calculate the cohesion for each sheep as the RMS distance from the sheep to all other sheep in the population
        n_sheep = sheep_positions.shape[0]
        d = distance.pdist(sheep_positions, 'euclidean')
        dsquare = distance.squareform(d)
        sheep_cohesions = np.sqrt(1/n_sheep * np.sum(np.power(dsquare,2), axis=1))

        #extract the targetting weights from the genome
        w = self.parseGenome(self.m_genome)[0]

        #Add a scaling factor to try and keep all inputs in the range 0-1
        #and do a value calcuation for each sheep position
        #TODO: assign the world scale from the config
        world_scale = 1/60
        sheep_values = (w[0] * sheep2dog_distances * world_scale +
            w[1] * sheep2goal_distances * world_scale +
            w[2] * sheep_cohesions * world_scale +
            w[3] * prob_sheep_my_types)

        return sheep_values
    #end function


    def probabilitySheepAreMyType(self, sheep_ids):
        #calculate the probability that the sheep is my type
        #TODO: Not using this in the vanilla version but add code for the fullfat one

        n_sheep = np.size(sheep_ids)
        prob_sheeps_mytype_vec = np.ones((n_sheep,1))

        return prob_sheeps_mytype_vec
    #end function


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


    def estimateSheepType(self, world, last_position, current_position):
        #checks whether a sheep moved according to its internal model or not
        #TODO: not used for vanilla model
        pass


    def cosineStep(self, angles):
        #A radial magnitude function which is:
        #   cos(theta) for [0,+180] & [-360,-270]
        #   -1 for [+180,+270] & [-180,-90]
        #   +1 for [-90, 0] & [+270,+360]
        PI = 3.14159265359
        angles = np.asarray(angles)
        val = np.zeros(np.size(angles))

        for n, theta in enumerate(np.nditer(angles)):
            if (0<theta and theta<=PI) or (-2*PI<=theta and theta<=-PI):
                val[n] = np.cos(theta)
            elif (-PI/2<theta and theta<=0) or (3/2*PI<theta and theta<=2*PI):
                val[n] = +1
            #condition is (-180<theta and theta<=-90) or (+180<theta and theta<=+270)
            else:
                val[n] = -1
        return val
    #end function


    def steeringPoint(self, goal_position, sheep_position):
        #calculate the vector describing the line collinear with the
        #goal and sheep
        steering_direction = goal_position - sheep_position
        #calculate the steering point as the point adjacent to the sheep, on the closest compass direction, to create a
        #co-linear line in the order steeringpoint-sheep-goal
        steering_direction = self.vector2CompassDirection(-steering_direction)[1]
        steering_point = sheep_position + steering_direction;
        return steering_point
    #end function


    def updateSprite(self, scale_factor=0):
        #do any changes to the spite image
        i=0
        while i<len(self.m_empowerment_thresholds) and self.m_empowerment>self.m_empowerment_thresholds[i]:
            i+=1
        # img = self.loadImage(self.m_sprite_images[i], self.image.get_height())
        if scale_factor==0:
            scale_factor = self.image.get_width()
        img = pygame.Surface((scale_factor, scale_factor))
        img.fill(colours.ERANGE[i])
        self.setImage(img)
        super().updateSprite(scale_factor)
        return

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















