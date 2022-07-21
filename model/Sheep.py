from functools import total_ordering

from numpy.core.shape_base import atleast_2d
from model.Agent import Agent
import numpy as np
import random
import sys, pygame

class Sheep(Agent):
    def __init__(self, position, type) -> None:
        #Run the superclass constructor
        super().__init__(position, type)
        #The probability of sheep grazing on each time step
        self.m_graze_probability = 0.05
        #Counter for the number of (non-consecutive) interactions with a sheepdog not matching the sheep's preference
        self.n_bad_interactions = 0
        #response map describe the behaviour to take in response to the agent type.  Simplest behaviour it to be repelled by an angle
        #in matlab they're stored as a cell array.  Maybe dictionary works better here?
        self.m_dog_response_map = {"normal" : 0}
        self.m_sheep_response_map = {"normal" : 0}
        #Set true to cause the sheep to bias its movement towards other sheep when being influcened by the dock
        self.m_use_flocking_behaviour = True

    #end function
    

    def setConfig(self, cfg):
        super().setConfig(cfg)
        #TODO: A pythony way of organising the config parameters
        self.m_graze_probability = cfg['graze_probability']
        self.m_dog_response_map = cfg['dog_repsonse_map']
        self.m_sheep_response_map = cfg['sheep_response_map']
        self.m_use_flocking_behaviour = cfg['use_flocking_behaviour']
    #end function


    def update(self, world):
        #main funciton to run once per tick.  Agent to updates its state and takes actions

        #update sensors (note this is usually more efficient to do once at a population level and pass the relevant results into update())
        xy_positions_vec, agent_ids, agent_types = world.entitiesInRange(self.m_position, self.m_sensor_range, b_include_position = False) 

        #calculate where to move due to sheep
        move_from_dogs = self.respondToAgents(world, xy_positions_vec, agent_types, self.m_dog_response_map)
        
        #calculate where to move due to dogs
        move_from_sheep = self.respondToAgents(world, xy_positions_vec, agent_types, self.m_sheep_response_map)

        #calculate where to move due to flocking effects
        #TODO translate the herding function from matlab to python.
        #   NB. This function may be able to replace the move_from_dogs component
        if self.m_use_flocking_behaviour:
            move_from_flocking = self.respondToHerding(xy_positions_vec, agent_types, self.m_dog_response_map)
        else:
            move_from_flocking = np.zeros((1,2))

        #if there's no movement from either sheep or dogs then initiate grazing behaviour
        if (np.all(move_from_sheep==0) and np.all(move_from_dogs==0)):
            move_from_graze = self.graze(world)
        else:
            move_from_graze = np.array((0,0))
        
        #sum up the movement contributions from grazing, sheep and dogs
        move = np.sum(move_from_dogs,axis=0) + np.sum(move_from_sheep,axis=0) + np.sum(move_from_flocking, axis=0) + move_from_graze
        move = self.vector2CompassDirection(move)[1]
        desired_next_position = self.m_position + np.squeeze(move)

        # is the desired_position inside the world boundary?
        # if it isn't then aim to move to a random adjacent space if one is available
        # else maintain current position
        if not(world.isInsideWorldBoundary(desired_next_position)):
            available_adj_sqs = self.getAdjacentFreeSquares(world, include_boundary_b=False)
            if np.any(available_adj_sqs):
                #move to a random adjacent square
                sz = np.shape(np.atleast_2d(available_adj_sqs))
                idx = random.randint(0, sz[0]-1)
                desired_next_position = available_adj_sqs[idx,:]
            else:
                desired_next_position = np.atleast_2d(self.getPosition())
        
        #try to move
        new_position = self.move(world, desired_next_position)
        
        #record the new position (this may be better set outside the agent if any further checks are needed before the agent updates its state)
        self.m_position = self.setPosition(new_position)

        #operations to tidy up at the end of an update
        self.m_adj_free_squares_is_current_b = False

        return new_position
    #end function         
     
          
    def respondToAgents(self, world, agent_positions, agent_types, response_map):
        #calculates where the sheep should move next in response to the agents surrounding it and the response_map
        #note it will ONLY respond to an agent if its type is listed in the response map

        move = np.array([0,0])
        #return the dictionary keys as a list
        known_types = np.array(list(response_map.keys()), dtype='i')
        #search for an agent type which matches a known type
        b_isknown_vec = self.ismember(known_types,agent_types)
        #if any of the agent types have an associated response
        if np.any(b_isknown_vec):
            agro_range = 1
            #loop through each known agent
            idx = np.nonzero(b_isknown_vec)
            for iagent in np.nditer(idx):
                grid_cell = np.asarray(agent_positions[iagent,:])
                # check if the same agent is still in the occupied cell on the next turn (this avoids two sheep bouncing off eachother and creating a gap of two squares).
                # TODO: Is there a better way to do this?  Could the world be updated after every agent takes an action (asynchoronous instead of synchronous)?
                if not(world.isAgentMoved(grid_cell)):
                    #check if it's inside the response range
                    if self.isInRange(grid_cell, self.m_position, agro_range):
                        # the agent is in range so now lookup my response by searching the table
                        for known_type, response in response_map.items():
                            if agent_types[iagent]==int(known_type):
                                #test which type of response
                                if type(response) is tuple:
                                    if response[0]=='s':
                                        #TODO: add code for slow response
                                        n_ticks = response[1]
                                        n_angle = response[2]
                                        V = [0,0]
                                    elif response[0]=='r':
                                        #TODO: add code for random response
                                        V = [0,0]
                                    else:
                                        #don't recognise the response
                                        V = [0,0]
                                else:
                                    #not a tuple so assume it's an integer angle to be repelled at
                                    angle = response
                                    V = self.rotate2D(self.m_position - grid_cell, angle)
                                #rotation vectpr is in continous space so round up the result
                                move = move + self.unitVec(V)
                        #TODO: Add code to convert the movement to a compass direction
                        move = move
        return move
    #end funciton



    def respondToHerding(self, agent_positions, agent_types, dog_response_map):

        PI = 3.1416
        SHEEP_VISION_ANGLE = PI/2

        move = np.array([0,0])
        #return the dictionary keys as a list
        known_dog_types = np.array(list(dog_response_map.keys()), dtype='i')
        #search for an agent type which matches a known type
        idx_dog = self.ismember(known_dog_types,agent_types)
        #if any of the agent types have an associated response
        if np.any(idx_dog):
            #check if any of the cells with dogs in are within response
            #range.  Minimise the range search to only those squares
            #with dogs in
            agro_range = 3
            #loop through each known agent
            idx = np.nonzero(idx_dog)
            for iagent in np.nditer(idx):
                grid_cell = np.asarray(agent_positions[iagent,:])
                #if the dog is within the agro range
                if np.linalg.norm(grid_cell-self.m_position) <= agro_range:
                    # find the nearest sheep in a direction away from the dog
                    #   identify which cells have sheep
                    idx_sheep = np.logical_not(idx_dog)
                    #if there are sheep visible
                    if np.any(idx_sheep):
                        #calculate the angle to each of the sheep to find which are "infront"
                        dog2me_vec = self.m_position-grid_cell
                        me2sheeps_vec = agent_positions[idx_sheep,:]-self.m_position
                        theta = self.angleBetweenVectors(dog2me_vec, me2sheeps_vec)

                        #need to keep the indexing the same length as
                        #   occupied cells, hence we just opperate on the
                        #   elements which are true (i.e. contain sheep)                   
                        idx_cells_with_sheep_infront = idx_sheep
                        idx = idx_cells_with_sheep_infront==1                        
                        idx_cells_with_sheep_infront[idx] = np.logical_and(idx_cells_with_sheep_infront[idx],np.less(np.abs(theta), SHEEP_VISION_ANGLE))
                        
                        if np.any(idx_cells_with_sheep_infront):
                            #calculate their CoM
                            com = self.calcCoM(agent_positions[idx_cells_with_sheep_infront,:])
                            move = move + self.unitVec(com - self.m_position)
            move = np.round(move)               
        return move

        # MATLAB CODE FOR FLOCKING BEHAVIOUR
        # ----------------------------------
        # function move = respondToHerding(obj, occupied_cells, agent_types_in_cells, dog_response_map)
            
        #     move = [0,0];            
                  
        #     %find which squares contain dogs
        #     idx_cells_with_dogs = myisMember(agent_types_in_cells, [dog_response_map{:,1}]);
            
        #     if any(idx_cells_with_dogs)
        #         %check if any of the cells with dogs in are within response
        #         %range.  Minimise the range search to only those squares
        #         %with dogs in
        #         agro_range = 3;
        #         idx_cells_with_dogs_in_range = idx_cells_with_dogs;
        #         idx = idx_cells_with_dogs==1;
        #         idx_cells_with_dogs_in_range(idx) = idx_cells_with_dogs(idx) & obj.isInRange(occupied_cells(idx_cells_with_dogs,:), obj.m_position, agro_range);
                
        #         if any(idx_cells_with_dogs_in_range)
        #             %loop through each square with a dog and take a step
        #             %towards other sheep
        #             for i_dog = find(makeRow(idx_cells_with_dogs_in_range))
        #                 grid_cell = occupied_cells(i_dog,:);
        #                 %find the nearest sheep in a direction away from
        #                 %the dog
        #                 %identify which cells have sheep
        #                 idx_cells_with_sheep = ~idx_cells_with_dogs;
                        
        #                 if any(idx_cells_with_sheep)
        #                     %calculate the angle to each of the sheep to find
        #                     %which are "infront"
        #                     dog2me_vec = obj.m_position-grid_cell;
        #                     me2sheeps_vec = occupied_cells(idx_cells_with_sheep,:)-obj.m_position;
        #                     theta = angleBetweenVectors(dog2me_vec, me2sheeps_vec);
        #                     %need to keep the indexing the same length as
        #                     %occupied cells, hence we just opperate on the
        #                     %elements which are true (i.e. contain sheep)
        #                     idx_cells_with_sheep_infront = idx_cells_with_sheep;
        #                     idx = idx_cells_with_sheep==1;
        #                     idx_cells_with_sheep_infront(idx) = idx_cells_with_sheep_infront(idx) & abs(theta)<deg2rad(90);
                            
        #                     if any(idx_cells_with_sheep_infront)
        #                         %calculate their CoM
        #                         com = calcCoM(occupied_cells(idx_cells_with_sheep_infront,:));
        #                         move = move + unitVec(com - obj.m_position);
        #                     end
        #                 end
        #             end
        #         end
        #         %move = round(move);
        #     end
            
        # end


    def graze(self, world):
        # samples a uniform random distribution and if the value is 
        # below a threshold move takes the agent to a randomly chosen adjacent square
        if random.random() < self.m_graze_probability:
            move = self.randomMove(world)
            # available_adj_sqs = self.getAdjacentFreeSquares(world, include_boundary_b=False)
            # if np.any(available_adj_sqs):
            #     sz = np.shape(np.atleast_2d(available_adj_sqs))
            #     idx = random.randint(0, sz[0]-1)
            #     move = available_adj_sqs[idx,:]
            # else:
            #     move = np.array([0,0])
        else:
            move = np.array([0,0])
        return move
    #end funciton


    def angleBetweenVectors(self, u,v):
        #Calculates the oblique angle (0..pi)  between two vectors.
        #if a list of vectors is provided then it must be N x dimensions e.g. Nx2
        #for 2D vectors

        #ensure both u and v are 2 axis and in the format Nx2
        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        
        if u.shape[0]==1 or v.shape[0]==1:
            dotuv = np.sum(u * v, axis=1)
        else:
            dotuv = np.dot(u,v)
        
        #normalise the result
        dotuv = dotuv/(np.linalg.norm(u) * np.linalg.norm(v))

        #need to ensure that numerical rounding doesn't cause the result to go outside the range -1 to +1
        cos_theta = np.clip(dotuv, -1,+1)

        #take the arc and return
        theta = np.real(np.arccos(cos_theta))

        return theta
    #end funciton

            


           



