from tokenize import endpats
from model.Structures import WorldGrid
import numpy as np
from model.SimLog import Logger


class World:
    def __init__(self, width, height, boundary_width):
        #width and height are the number or columns and rows respectively the grid is indexed from 0 to (width-1) and 0 to (height-1)
        self.m_width = width
        self.m_height = height
        self.m_boundary_width = boundary_width
        self.m_grid = WorldGrid(width, height)
        self.m_grid_next = WorldGrid(width, height)
        self.m_grid_previous = WorldGrid(width, height)
        self.m_sheep_id_range = tuple((100,254))
        self.m_dog_id_range = tuple((1,99))

        # Set the ticks variable to be a member of the World class for easy access elsewhere.
        self.tick = 0 #changed from None (it's just casuing problems not being a number?!)

        #create a logger   
        self.log = Logger()

        return
    #end function


    def pickleLog(self, file_name):
        import pickle
        fileo = open(f'{file_name}.pkl', 'wb')
        pickle.dump(self.log,fileo)
        fileo.close()
        return

    
    def setGrid(self, agent_positions, agent_ids, agent_types):
        #**************TESTED**************************
        #initialise variables
        agent_positions_vec = np.atleast_2d(np.asarray(agent_positions))
        #convert to grid coordinates
        agent_positions_vec = self.pos2GridRef(agent_positions_vec)
        #empties the world and re-populates it with agents
        self.m_grid.ids = np.zeros((self.m_width, self.m_height))
        self.m_grid.types = np.zeros((self.m_width, self.m_height))
        n_agents = len(agent_types)
        #loads the agent_id and type to the grid provided it's a valids grid reference
        for i_agent, atype in enumerate(agent_types):
            position = agent_positions_vec[i_agent,:]
            if self.isValidGridPosition(position):
                self.m_grid.ids[position[0], position[1]] = agent_ids[i_agent]
                self.m_grid.types[position[0], position[1]] = agent_types[i_agent]
        return
    #end function


    def setNextGridPosition(self, new_position, current_position, agent_id, agent_type):
        #**************TESTED**************************
        #Checks the position is empty on both the current state of the
        #world and the new update then writes the new agent type and id
        #if it is.
        if ~self.isValidGridPosition(new_position):
            position_set = current_position
        elif ~self.isEmptyGridPosition(new_position, agent_id):
            position_set = current_position
        else:
            position_set = new_position
        grid_coord = self.pos2GridRef(position_set)
        #this function can only accept one new position at a time so ensure it's a 1D grid_coord
        grid_coord = np.squeeze(grid_coord)
        self.m_grid_next.ids[grid_coord[0],grid_coord[1]] = agent_id
        self.m_grid_next.types[grid_coord[0],grid_coord[1]] = agent_type
        return position_set
    #end function


    def isInsideWorldBoundary(self, positions):
        #**************TESTED**************************
        # checks if the position is within the smaller region of the world
        # which does not include the reserved boundary area.  The boundary exists to enable a dog to always
        # get behind a sheep

        #initialise variables
        pos_vec = np.atleast_2d(np.array(positions))

        #check if position is a valid grid reference
        if np.shape(pos_vec)[1] == 2 and np.array_equal(pos_vec, np.round(pos_vec)):
            #check if pos is inside the boundary or outside the grid world limits!
            bw = self.m_boundary_width
            b_x = np.logical_and(pos_vec[:,0]>(0+bw), pos_vec[:,0]<=(self.m_width-bw))
            b_y = np.logical_and(pos_vec[:,1]>(0+bw), pos_vec[:,1]<=(self.m_height-bw))
            b_inboundary = np.logical_and(b_x,b_y)
        else:
            b_inboundary = False
        return b_inboundary
    #end function


    def isEmptyGridPosition(self, positions, agent_id = 0, b_check_previous = False):
        #**************TESTED**************************
        # This checks if the square is empty in the current timestep and
        # another agent hasn't already claimed it in the next time step.
        # If my_id is provided then is still return an empty result if
        # the square on the current time step is occupied by my_id -
        # i.e. an agent which doesn't move is allowed to claim it's
        # current position for the next time step.
        #     positions may be an Nx2 list of positions

        #initialise the variables
        positions_vec = np.atleast_2d(np.asarray(positions))
        n_positions = self.count2dVectors(positions_vec)
        b_isempty = np.zeros(n_positions, dtype=bool)

        #convert coordinates
        positions_vec = self.pos2GridRef(positions_vec)

        # loop through each position
        # TODO: figure out how to vectorise this like the matlab code does
        # TODO: sort the interface between isValidGridPosition and isEmptyGridPosition to stop converting the grid ref back before passing!
        for i in range(n_positions):
            if self.isValidGridPosition(self.gridRef2Pos(positions_vec[i,:])):
                #check the previous state
                if b_check_previous:
                    id = self.m_grid_previous.ids[positions_vec[i,0],positions_vec[i,1]]
                    b_isempty[i] = id==0
                #check the current and next states
                else:
                    id = self.m_grid.ids[positions_vec[i,0],positions_vec[i,1]]
                    id_next =  self.m_grid_next.ids[positions_vec[i,0],positions_vec[i,1]]
                    b_isempty[i] = (id == 0) and (id_next == 0)
                #if an agent_id was provided then reutrn the position as empty if it contains that agent id
                if agent_id != 0:
                    b_isempty[i] =  b_isempty[i] or id == agent_id
        return b_isempty
    #end function


    def advanceGrid(self):
        #UNTESTED
        #advance the grid in time

        # store an historical copy of the world
        # could also do copy.deepcopy(gw.m_grid)
        self.m_grid_previous.ids = np.copy(self.m_grid.ids)
        self.m_grid_previous.types = np.copy(self.m_grid.types)
        # copy the positions at the next tick to the grid
        self.m_grid.ids = np.copy(self.m_grid_next.ids)
        self.m_grid.types = np.copy(self.m_grid_next.types)
        #zero the next grid
        self.m_grid_next.ids[:] = 0
        self.m_grid_next.types[:] = 0
        pass
    #end function


    def displayGrid(self):
        #maps the agent types to a set of colours and then displays the colours in the agents with their corresponding colour on a grid
        pass
    #end function


    def entitiesInRange(self, position, range, b_include_position=False):
        #*************************TESTED*******************************
        #returns information on the agents within range of position.
        #this function is NOT vectorised!!
        #   position: a 2d grid reference
        #   range: a scalar distance
        #   b_include_position: false excludes an agent located at position from being returne
        #returns a tuple of numpy arrays: (agent_positions, agent_ids, agent_types) in range

        if not(self.isValidGridPosition(position)):
            raise Exception(f'unable to calculate entities in range.  Position {position} is invalid')
        # Reduce the search space by applying a square mask to the world and
        # copying anything under the mask into search_grid
        # Also clip the range if needed to prevent it going outside the world boundary
        # Note, python quirk is to begin a slice on the number requested but end it at the number before
        # i.e. the slice ends on the element number (referenced from 0) so need to +1 to the upper range
        # also note, size of grid is 1 less than the width (referenced from 0)
        position_vec = np.asarray(position) #np.atleast_2d(np.asarray(position))
        position_vec = self.pos2GridRef(position_vec)
        position_vec = np.squeeze(position_vec)
        xlower = max([0, position_vec[0]-range])
        xupper = min([self.m_width-1, position_vec[0]+range])
        ylower = max([0, position_vec[1]-range])
        yupper = min([self.m_height-1, position_vec[1]+range])
        search_grid = self.m_grid.ids[xlower:xupper+1,ylower:yupper+1]
        # extract the agent information for any non-empty grid squares
        #idx is a tuple of arrays, one for each dimension
        idx_local = np.nonzero(search_grid)
        agent_ids = search_grid[idx_local]
        # convert to global coordinates
        idx_global = (idx_local[0]+xlower, idx_local[1]+ylower)
        # extract the agent types from obj.m_grid.id without repeating the search!!
        agent_types = self.m_grid.types[idx_global]
        # return the positions in global coordinates
        xy_positions_vec = np.array(idx_global)
        #transpose to get it in the form where the ith row is the ith occupied position
        xy_positions_vec = np.transpose(xy_positions_vec)

        #If the results don't include the contents of "position" provided then remove that row from the returned values
        if not(b_include_position):
            #or could do (xy_positions_vec==position_vec).all(axis=1).nonzero() the .nonzero() extracts the index of the row where they match
            #the opinion on stackoverflow is that using &s might be faster
            idx = (xy_positions_vec[:,0]==position_vec[0]) & ((xy_positions_vec[:,1]==position_vec[1]))
            agent_ids = agent_ids[np.logical_not(idx)] # or np.delete(agent_ids,np.nonzero(idx))
            agent_types = agent_types[np.logical_not(idx)] # or  np.delete(agent_types,np.nonzero(idx))
            xy_positions_vec = xy_positions_vec[np.logical_not(idx)] #or np.delete(xy_positions_vec,((n+1)/2, 0),1)

        #translate everything back to row, column format
        xy_positions_vec = self.gridRef2Pos(xy_positions_vec)
        return (xy_positions_vec, agent_ids, agent_types)
    #end function


    def isDogId(self, ids):
        #returns true if ids belongs to a dog.  If ids is an N element vector then returned value is a N element vector of boolean values
        ids_vec = np.asarray(ids)
        b_isDog = np.logical_and(ids_vec>=self.m_dog_id_range[0], ids_vec<=self.m_dog_id_range[1])
        return b_isDog
    #end function


    def isValidGridPosition(self, position_vec):
    #**************TESTED**************************
    #checks that the grid reference "pos" is an 2D integer and is within the limits of the grid world's size
    #TODO: find a way to force numpy to give a row every 2 dimensions i.e. the shape (1,2) and not (2,)
    #current approach is a nasty hack to check what type it is first before handling it
        b_valid = False
        position_vec = np.atleast_2d(np.asarray(position_vec))
        #confirm it has integers values
        if np.array_equal(position_vec, np.round(position_vec)):
            #convert to grid reference from 0 instead of position referenced from 1
            position_vec = self.pos2GridRef(position_vec)
            #no more than 2 columns
            if position_vec.shape[1] == 2:
                    #confirm the the values are within the boundary of the grid
                    #grid is indexed from zero so -1 from the supplied width and height
                    b_w = np.logical_and(position_vec[:,0] >= 0, position_vec[:,0] <= (self.m_width-1))
                    b_h = np.logical_and(position_vec[:,1] >= 0 , position_vec[:,1] <= (self.m_height-1))
                    b_valid = np.logical_and(b_w, b_h)
        return b_valid


    def count2dVectors(self, vectors):
        #**************TESTED**************************
        #counts the number of 2D vectors present.
        # Vectors are arranged as row vectors. i.e. the i'th row is 2 columns.
        # This is a work around because a 1D vector only provides a 1D element via the shape function which
        # screws up a lot of the way lists were handled in the matlab model

        vectors_vec = np.asarray(vectors)
        if vectors_vec.size == 2:
            return 1
        elif vectors_vec.shape[1]==2:
            return vectors_vec.shape[0]
        else:
            #something went wrong, it's got more than 2 elements and more than 2 columns
            return 0
        return


    def isAgentMoved(self, pos):
        #checks if the agent at pos has already moved this turn
        grid_cell = np.squeeze(np.asarray(pos)-1)
        b_moved = self.m_grid.ids[grid_cell[0],grid_cell[1]] != self.m_grid_next.ids[grid_cell[0],grid_cell[1]]
        return b_moved
    #end function


    def adjacentFreeSpaces(self, position, b_include_boundary = False):
        #**************TESTED**************************
        # Takes in a position and returns the coordinates of adjacent
        # spaces which are empty and within the world boundary.  Note it
        # doesn't include pos in the opperation!
        #    if include_boundary=true then spaces within the boundary of
        #    the world are included in the returned positions.  Default
        #    is false
        lookup_vec = np.array([[0,+1], [+1,+1], [+1,+0], [+1,-1], [+0,-1],
                    [-1,-1], [-1,+0], [-1,+1]])
        #Calculate the absolute references of the adjacent squares
        adjacent_squares = lookup_vec + np.atleast_2d(np.array(position))
        #Calculate which of the adjacent squares are within the world limits (optionally including the reserved boundary region)
        if b_include_boundary:
            b_insidelimits = self.isValidGridPosition(adjacent_squares)
        else:
            b_insidelimits = self.isInsideWorldBoundary(adjacent_squares)
        #Calculate which adjacent squares are occupied by another agent
        b_empty = self.isEmptyGridPosition(adjacent_squares)
        #return positions which are inside the world limit, adjacent to position and empty
        free_spaces = adjacent_squares[np.logical_and(b_insidelimits,b_empty)]
        return free_spaces #self.gridRef2Pos(free_spaces)
    #end function


    def pos2GridRef(self, position_vec):
        #**************TESTED**************************
        #converts a gridworld position referenced from 1 to a python array referenced from 0
        x = np.asarray(position_vec) - 1
        return x.astype('i')
    #end function


    def gridRef2Pos(self, position_vec):
        #**************TESTED**************************
        #converts a gridworld position referenced from 1 to a python array referenced from 0
        return np.asarray(position_vec) + 1
    #end function


    def getSheepPositions(self):
        #idx is a tuple of arrays, one for each dimension
        idx_occupied = np.nonzero(self.m_grid.ids)
        agent_ids = self.m_grid.ids[idx_occupied]
        agent_positions = np.array(idx_occupied)
        agent_positions = agent_positions.transpose()
        idx_sheep = np.logical_not(self.isDogId(agent_ids))
        return agent_positions[idx_sheep,:]
    #end function


    def calcCoM(self, vector_list):
        #Calculates the centre of mass as the average position of the
        #vectors listed in vector_list
        #vector_list = [x1,y1;x2,y2;....]

        if np.any(vector_list):
            V = np.atleast_2d(vector_list)
            N = V.shape[0]
            com = np.sum(vector_list,axis=0)/N
        else:
            com = np.array([])
        return com
    #end function

#end class
