import numpy as np
import colours as colours
import sys, pygame

#Agent is a super class for dogs and sheep
class Agent(pygame.sprite.Sprite):
    # Agents are a subclass of Pygame sprites.
    # They work the same as the matlab model except for the additional "sprite" functions which are used for visualisation only
    # (sprites do have additional functions such as checking for collisions but these aren't being used at the moment (because they clash with the functions used in the World class))
    def __init__(self, position, type) -> None:
        #intialise the agent as a pygame sprite
        pygame.sprite.Sprite.__init__(self)
        # self.m_image = []
        # self.m_rect = []

        #declare the physical properties (need to be declared before the sprite image)
        self.m_sensor_range = 10
        self.m_position = np.asarray(position)

        #declare the locical properties
        self.m_id = 0
        self.m_type = type

        #create a token sprint image
        self.m_sprite_position = pygame.math.Vector2((position[0], position[1] ))
        img = pygame.Surface((10,10))
        img = pygame.transform.smoothscale(img, (10,10))
        img.fill(colours.GREEN)
        self.setImage(img)

        #declare containers for cached information
        self.m_adj_free_squares = []
        self.m_adj_free_squares_is_current_b = False
        return
    #end function


    def loadImage(self,imageName, sq_size):
        #loads and scales an image for the agent
        image = pygame.image.load(imageName)
        image = pygame.transform.smoothscale(image, (sq_size,sq_size))
        return image
    #end function


    def setImage(self, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = self.m_sprite_position
        return
    #end function


    def setPosition(self, new_position):
        self.m_position = np.asarray(new_position)
        return self.m_position
    #end function


    def setSpritePosition(self, new_position):
        #ensure new_position is a 1D vector
        new_position = np.squeeze(new_position)
        self.m_sprite_position = pygame.math.Vector2((new_position[0],new_position[1]))
        self.rect.center = round(self.m_sprite_position.x), round(self.m_sprite_position.y)
        return self.m_sprite_position
    #end function


    def updateSprite(self, scale_factor):
        #simple function which moves the agents image to align with its position in the screen coordinate frame
        pos_in_screen_coords = self.m_position * scale_factor
        #the screen draws by default as y = rows and x = columns so need to reverse it
        pos_in_screen_coords = np.flip(pos_in_screen_coords)
        self.setSpritePosition(pos_in_screen_coords)
        return
    #end function


    def setConfig(self, cfg):
        #copies configurable parameters into the agent
        #cfg is a structure of parameters.  There maybe a pythony way of doing this??
        self.m_sensor_range = cfg['sensor_range']
    #end function


    def move(self, world, desired_next_position):
        #attempts to move the agent in the world.  next_position is the actual position it moves to which may or may not be the position it wants.
        next_position = world.setNextGridPosition(desired_next_position, self.m_position, self.m_id, self.m_type)
        self.setPosition(next_position)
        return next_position
    #end function


    def doNothing(self, world):
        # agent moves to its current position.
        # this causes the agent to redraw itself in the world using its current position and takes no other actions
        next_position = self.move(world, self.m_position)
        return next_position
    #end function


    def update(self,world):
        # the main function used by the agent to update its state and take actions for a single time tick.
        # each subclass will override this function with the specific actions it wants to take
        #TODO: This needs pythoning
        #update sensors
        xy_positions_vec, agent_ids, agent_types = world.entitiesInRange(self.m_position, self.m_sensor_range, include_agent_position_b = False)
        #do stuff
        desired_new_position = self.m_position
        #move
        new_position = self.move(world, desired_new_position)
        self.setPosition(new_position)

        return new_position
    #end function


    def getPosition(self):
        #returns the agent's current position as a python standard tuple
        return (self.m_position[0], self.m_position[1])


    def getId(self):
        #returns the agent's Id
        return self.m_id


    def getType(self):
        #returns the agent's type
        return self.m_type


    def getAdjacentFreeSquares(self, world, include_boundary_b=False):
        # returns the adjacent free squares from the world and caches
        # the result so that the request will only be made once per tick
        if not(self.m_adj_free_squares_is_current_b):
            self.m_adj_free_squares = world.adjacentFreeSpaces(self.m_position, include_boundary_b)
            self.m_adj_free_squares_is_current_b = True
        return self.m_adj_free_squares


    def isInRange(self, cell_a, cell_b, range):
        #uses a square mask to calculate is cell_b is in range of cell_a
        in_range_b = 0
        #check in x
        if cell_a[0]>=(cell_b[0] - range) and cell_a[0]<=(cell_b[0] + range):
            #check in y
            if cell_a[1]>=(cell_b[1]-range) and cell_a[1]<=( cell_b[1] + range):
                in_range_b = 1
        return in_range_b
    #end function


    def rotate2D(self, in_vec, theta):
        import math as m
        #performs a 2D rotation where positive is CCW (right hand screw rule)
        #TODO: move this into the agent superclass?

        #create the rotation matrix
        theta =m.radians(theta)
        R_mat = np.array([[m.cos(theta), -m.sin(theta)], [m.sin(theta), m.cos(theta)]])

        #rotate the vector
        #python seems insensitive to whether in_vec is a row or column vector so skip this check
        in_vec = np.atleast_2d(in_vec)
        sz = in_vec.shape
        if sz[1]>sz[0]:
            #in_vec is a column vector
            #multiply the input by the rotation matrix (need to transpose it to do this)
            out_vec = R_mat @ in_vec.transpose()
            #transpose the result to match the input format
            out_vec = np.transpose(out_vec)
        else:
            #in_vec is a row vector
            out_vec = R_mat @ in_vec

        return out_vec
    #end function


    def ismember(self,x,y):
        #searches for the elements of x in y and returns a vector of length y
        #where true indicates the associated element of y matches a element in x
        #adapted from https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
        import numpy as np

        index = np.argsort(x)
        sorted_x = x[index]
        sorted_index = np.searchsorted(sorted_x, y)

        yindex = np.take(index, sorted_index, mode="clip")
        #mask = x[yindex] != y
        mask = x[yindex] == y

        #result = np.ma.array(yindex, mask=mask)
        result = mask
        return result
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


    def cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    #end function


    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    #end function


    def vector2CompassDirection(self, V):
        import math as m
        # Converts a vector into a compass direction where:
        # north could be 0 or 360 so it's included twice in the lookup table
        # table = { 1,[+0,+1],'N';
        #           2, [+1,+1], 'NE';
        #           3, [+1,+0], 'E';
        #           4, [+1,-1], 'SE';
        #           5, [+0,-1], 'S';
        #           6, [-1,-1], 'SW';
        #           7, [-1,+0], 'W';
        #           8, [-1,+1], 'NW';
        #           9, [+0,+1], 'N'};

        #do some inititalisation
        V = np.atleast_2d(V)
        #Could be expanded to work for 4 or 16 divisions
        ndivisions = 8
        div_angle = (2*m.pi)/ndivisions
        # if the vector is zero then return a stationary result
        if np.equal(V,[0,0]).all():
            compass_dir_num = np.nan
            compass_dir_vec = np.array([0,0])
            compass_dir_name = '0'
            return (compass_dir_num, compass_dir_vec, compass_dir_name)
        # create the lookup table
        lookup_dir = np.arange(0,9)
        lookup_vec = np.array([
            [+0,+1],
            [+1,+1],
            [+1,+0],
            [+1,-1],
            [+0,-1],
            [-1,-1],
            [-1,+0],
            [-1,+1],
            [+0,+1]
        ])
        lookup_name = np.array(['N','NE','E','SE','S','SW','W','NW','N'])
        # find the angle of vector
        # (https://stackoverflow.com/questions/1311049/how-to-map-atan2-to-degrees-0-360)
        # theta is measured ccw in radians from the positive x axis
        theta = np.arctan2(V[:,1], V[:,0])
        #convert theta is measured CW where 0 is along the positive y axis
        theta = -(theta - m.pi/2)
        #convert theta to the range 0..360
        theta = np.remainder(theta+(2*m.pi),2*m.pi)
        # lookup the result in the array
        # https://www.campbellsci.eu/blog/convert-wind-directions#:~:text=To%20convert%20degrees%20to%20compass,degrees%20instead%20of%2022.5%20degrees.
        idx = np.round(theta / div_angle)
        #idx is used as an index so ensure it's cast as an int
        idx = idx.astype(int)
        compass_dir_num = lookup_dir[idx]
        # for clarity, ensure any 'north' which is 360 is returned as 0
        compass_dir_num[compass_dir_num==8] = 1;
        compass_dir_vec = lookup_vec[idx,:]
        compass_dir_name = lookup_name[idx]
        return (compass_dir_num, compass_dir_vec, compass_dir_name)
    #end function


    def randomMove(self, world, allow_move_into_boundary_b = False):
        import random
        available_adj_sqs = self.getAdjacentFreeSquares(world, allow_move_into_boundary_b)
        if np.any(available_adj_sqs):
            sz = np.shape(np.atleast_2d(available_adj_sqs))
            idx = random.randint(0, sz[0]-1)
            move = available_adj_sqs[idx,:] - self.m_position
        else:
            move = np.array([0,0])
        return move
    #end function


    def unitVec(self,V):
        # UNITVEC Converts an input vector in to a unit vector
        #   Function accepts a list of vectors in row format [xi,yi]
        V = np.atleast_2d(V)
        sz = V.shape
        V_normalised = np.zeros(sz)

        #convert to row vectors if needed
        if sz[0] == 2 and sz[1] == 1:
            V = np.transpose(V)
            V_normalised = np.transpose(V_normalised)

        mag = np.linalg.norm(V, axis=1)
        idx = mag!=0

        #if any of the vectors are not zero then convect them to unit vectors
        if np.any(idx):
            V_normalised[idx,:] = V[idx,:]/mag[idx]
        
        return V_normalised
    #end function

#end class