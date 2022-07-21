import sys, pygame
import random

class Population:
    # Significant changes from the matlab model:
    #   + Population does not store a master list of the types and positions of all agents contained within it
    #   + Agents are stored in a pygame sprite group instead of a cell array
    #   + The createAgent function has been replaced by addAgent.  This means an agent must be created first and then added to
    #     to the population (the matlab model combines the two steps)
    def __init__(self, min_id = 1) -> None:
        import numpy as np
        #self.m_positions = np.empty(shape=[0,2],dtype=int) 
        #self.m_types = np.array([])    
        self.m_min_id_value = min_id
        self.m_next_avail_id = min_id
        self.m_agents = pygame.sprite.Group()
    #end function

    
    def addAgent(self, agent):
        #sets up an agent and adds it to the population
        self.m_agents.add(agent)
        agent.m_id = self.m_next_avail_id
        self.m_next_avail_id += 1
        return agent
    #end function


    def configure(self, cfg):
        #runs the setConfig function on every agent in the population
        for agent in self.m_agents:
            agent.setConfig(cfg)
    #end function


    def shuffleGroup(self, group_of_sprites):
        """Takes a group of sprites as input and returns them as a shuffled list of sprites"""
        #Return a shuffled list
        return random.sample(group_of_sprites.sprites(), len(group_of_sprites))
    #end function


    def update(self, world, b_in_rand_order = 1, b_update_sprites = 0, local2screen_scaling = 1):
        #calls the update function on each agent in the population
        #defaults to updating the agents in random order
        # changes from matlab model:
        #   + no message board passed (or used!)
        #   + the sensors are updated inside the agent and not within this population function.
        #   + if b_update_sprites is set then the visual element of the agent is also updated

        if b_in_rand_order:
            for agent in self.shuffleGroup(self.m_agents):
                new_position = agent.update(world)
                if b_update_sprites:
                    agent.updateSprite(local2screen_scaling)
        else:
            for agent in self.m_agents:
                new_position = agent.update(world)
                if b_update_sprites:
                    agent.updateSprite(local2screen_scaling)
        # #for now, try only storing the location information in the world map and the agent and see if
        # #less duplication makes up for a slight loss in speed
        # for i, agent in enumerate(self.m_agents):
        #     new_position = agent.update(world)
        #     #Record the agent's grid position in the population table, this duplicates information and can be avoided?  
        #     #Although it does make extracting all the agent positions in the population easier
        #     self.m_positions[i,:] = new_position
        return
    #end function


    def doNothing(self, world, b_update_sprites = 0, local2screen_scaling = 1):
        #Passes the turn for all agents in the population.  They take no action
        for i,agent in enumerate(self.m_agents):
            #new position should be the same as the current position but it's set just incase
            new_position = agent.doNothing(world)
            if b_update_sprites:
                agent.updateSprite(local2screen_scaling)
            #self.m_positions[i,:] = new_position
    #end function

#end class
