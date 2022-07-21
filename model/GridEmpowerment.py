from numpy.core.fromnumeric import squeeze
from treelib import Node, Tree
import numpy as np
SHEEP_TYPE = 10

class NodeState():
    def __init__(self, pos = (0,0), sensor_state = np.zeros((5,5)), local2global_mapping = (0,0)) -> None:
        self.m_position = pos
        self.m_map = sensor_state
        self.m_local2global = local2global_mapping
        return
    #end function


def uniqueByFirst(a):
    #for a set of 2D arrays stacked along the first axis, this function returns the unique 2D arrays.
    #It works by converting the 2D arrays into 1D tuples and arranying them in a list
    #it removes the duplicate tuples then converts the uniques back into a stack of 2D arrays
    #There's a faster (but harder to understand method) listed here
    #https://stackoverflow.com/questions/41071116/how-to-remove-duplicates-from-a-3d-array-in-python
    b = np.array(a)
    n_2d_arrays = b.shape[0]
    n_2d_rows = b.shape[1]
    n_2d_cols = b.shape[2]
    new_array = [tuple(row) for row in b.reshape(n_2d_arrays, n_2d_rows*n_2d_cols)]
    uniques = list(set(new_array))
    output = np.array(uniques).reshape(len(uniques), n_2d_rows, n_2d_cols)
    return output


def uniqueByFirstNp(a):
    #for a set of 2D arrays stacked along the first axis, this function returns the unique 2D arrays.
    #It works by converting the 2D arrays into 1D tuples and arranying them in a list
    #it removes the duplicate tuples then converts the uniques back into a stack of 2D arrays
    #There's a faster (but harder to understand method) listed here
    #https://stackoverflow.com/questions/41071116/how-to-remove-duplicates-from-a-3d-array-in-python
    b = np.array(a)
    n_2d_arrays = b.shape[0]
    n_2d_rows = b.shape[1]
    n_2d_cols = b.shape[2]
    new_array = [tuple(row) for row in b.reshape(n_2d_arrays, n_2d_rows*n_2d_cols)]
    uniques, idx = np.unique(new_array, axis=0, return_index=True)
    output = np.array(uniques).reshape(uniques.shape[0], n_2d_rows, n_2d_cols)
    return output,idx


def nodeNameToPos(node_name):
    idx_R = node_name.find("R")
    idx_C = node_name.find("C")
    idx_D = node_name.find("D")
    return (int(node_name[idx_R+1:idx_C]), int(node_name[idx_C+1:idx_D]))
#end function


def isValidSquare(positions, map_width, map_height):
    positions = np.atleast_2d(positions)
    b_w = np.logical_and(positions[:,0] >= 0, positions[:,0] <= map_width-1)
    b_h = np.logical_and(positions[:,1] >= 0 , positions[:,1] <= map_height-1)
    b_insidelimits = np.logical_and(b_w, b_h)
    return b_insidelimits
#end function


def isEmptySquare(position,map):
    position = squeeze(position)
    return map[position[0],position[1]] == 0


def getAdjacentSquares(position, map):
    #adapted from isValidGridPosition and adjacentFreeSpaces in World.py
    lookup_vec = np.array([[0,+1], [+1,+1], [+1,+0], [+1,-1], [+0,-1], 
                    [-1,-1], [-1,+0], [-1,+1]])
    #Calculate the absolute references of the adjacent squares
    adjacent_squares = lookup_vec + np.atleast_2d(np.array(position))
    #Calculate which of the adjacent squares are within the map limits
    b_is_valid_sq = isValidSquare(adjacent_squares, map.shape[0],map.shape[1])
    #Calculate which adjacent squares are occupied by another agent
    b_empty = np.zeros(b_is_valid_sq.shape)
    b_empty[b_is_valid_sq] =  map[adjacent_squares[b_is_valid_sq,0], adjacent_squares[b_is_valid_sq,1]]==0
    #return positions which are inside the world limit, adjacent to position and empty
    free_spaces = adjacent_squares[np.logical_and(b_is_valid_sq,b_empty)]
    return free_spaces
#end function


def simulateWorld(map, position):
    #intialise the returned state as the current state
    new_map = map.copy()
    
    #extract the sheep_positions
    #TODO: Figure out why np.nonzero returns a touple of arrays instead of an array!
    sheep_positions = np.nonzero(map==SHEEP_TYPE) 
    #numpy indexes positions with a touple of 2 arrays, one for rows and 1 for columns so need to manpiulate it to get
    #it into a vector of 2d positions
    sheep_positions = np.asarray(sheep_positions).transpose()
    
    #test if next to a sheep
    position = np.atleast_2d(position)
    d = np.linalg.norm(sheep_positions-position, axis=1)
    adjacent_sheep = sheep_positions[d<1.5,:]

    #if there's no adjacent sheep then assume state is unchanged
    #NOTE: empowerment ignores the actions of all other actors in the world
    if not(np.any(adjacent_sheep)):
        return new_map
    else:
        #loop through each of the adjacent sheep and move it directly away from the position     
        for sheep_pos in adjacent_sheep:
            sheep_pos = np.squeeze(sheep_pos)
            sheep_type = map[sheep_pos[0],sheep_pos[1]]
            #test if it's feasible for the sheep to move
            expected_pos = sheep_pos + (sheep_pos - position)
            if (isValidSquare(expected_pos,map.shape[0],map.shape[1]) and isEmptySquare(expected_pos,map)):
                #move the sheep
                new_map[sheep_pos[0],sheep_pos[1]] = 0
                expected_pos = np.squeeze(expected_pos)
                new_map[expected_pos[0],expected_pos[1]] = sheep_type   
    return new_map
#end function


def createTreeOfMoves(tree, head, depth, depth_max):
    """
    A recursive function which builds the movement tree.
    It does this by adding reachable locations into tree with head as the parent node
    If depth<=n_moves_max then the function will call itself for each of the child nodes it creates
    """
    sub_t = expandTreeOneLevel(tree, head, depth, sensor_range=depth_max+1)
    children = sub_t.leaves(head.identifier)
    #simulate a move to each child position and expand the tree if needed
    for child in children:
        # simulate the world state starting from the child position and store it against the parent
        current_state = head.data.m_map
        state = simulateWorld(current_state, child.data.m_position)
        child.data.m_map = state
        child.data.m_local2global = head.data.m_local2global
        #depth first search
        if depth<depth_max:
            createTreeOfMoves(tree,child,depth+1, depth_max)
    return tree
#end function
    

def expandTreeOneLevel(tree, head, current_tree_level, sensor_range):
    """
        Adds one level of nodes to the current tree.
        VERY IMPORTANT: the nodes are created with random (unique) identifiers
                        This means it's difficult to directly access a single node based on it's human readable tag
    """
    #decode the positon of the parent node
    head_position = nodeNameToPos(head.tag)
     # 2: Create a child node for each reachable square from the parent node
    reachable_squares = getAdjacentSquares(head.data.m_position, head.data.m_map)
    #add the current position to the child nodes, this accounts for the null option where the agent doesn't move
    #reachable_squares.append(head.data.m_position)
    reachable_squares = np.append(reachable_squares, np.atleast_2d(head.data.m_position), axis=0)
    for square in reachable_squares:
        gposition = np.squeeze(square + head.data.m_local2global)
        nname = f"R{gposition[0]}C{gposition[1]}D{current_tree_level}"
        #initalise the children without a sensor state
        ndata = NodeState(pos=square)
        tree.create_node(tag=nname, parent=head.identifier, data=ndata)
    #return the tree of all reachable squares in one step from the head position
    return tree.subtree(head.identifier)
#end function


def calcMovementTree(world, start_position, max_moves):

    #NOTE: this function assumes all position vectors are numpy 1D
    # So it needs to do usual irritating numpy vector manipulation to avoid the 1x2 2D row vector or a 0 x 2 1D vector confusion
    
    #start_position is in row, column format where the first square is row 1, column 1
    #to keep things simple, the whole empowerment function works referenced from 0,0   
    start_position_1d = np.squeeze(start_position.copy())
    start_position_local = start_position_1d - 1

    #update the sensors
    sight_horizon = max_moves+2
    #world returns positions in row, column format (as per an array)
    agent_positions, agent_ids, agent_types =  world.entitiesInRange(start_position_1d, sight_horizon)
    #translate agent_positions to start from 0,0
    agent_positions = agent_positions - 1
    #locate the position of the lower left corner of the local grid in world coordinates (respect the world limits)
    P00 = np.atleast_2d( [max(0,start_position_local[0]-sight_horizon), max(0,start_position_local[1]-sight_horizon)] )
    #locate the position of the upper right corner of the local grid in world coordinates (respect the world limits)
    P11 = np.atleast_2d( [min(start_position_local[0]+sight_horizon, world.m_width-1), min(start_position_local[1]+sight_horizon, world.m_height-1)] )
    #create a blank grid centered on the agent's position
    sstate = np.zeros(np.squeeze(P11-P00)+1)  
    #translate the agent positions from a world coordinate frame to a local one with the starting position at the centre
    #and referenced from 0,0 (this is a convienice so the)
    agent_positions = agent_positions - P00
    # copy the agent types into the grid (references from 0)
    sstate[agent_positions[:,0], agent_positions[:,1]] = agent_types

    # Create an intial tree based on the current reachable squares
    # Each node stores a map showing the positions and types of agents seen at it's position
    #   and a position which is the nodes position in local map coordinates
    #   the node tag is in global coordinates
    moves = Tree()
    i_depth = 0
    
    # create the root node
    nname = f"R{start_position_1d[0]}C{start_position_1d[1]}D{i_depth}"
    ntag = nname
    position_local = np.squeeze(start_position_local - P00)
    ndata = NodeState((position_local[0],position_local[1]), sstate, P00+1)
    #print(f"Root Node Sensor state is \n {sstate}")
    moves.create_node(tag=nname, identifier=ntag, data=ndata)
    # create the rest of the tree nodes
    head = moves.get_node(ntag)
    tree = createTreeOfMoves(moves, head, i_depth+1, max_moves)
    return tree


def sumEmpowerment(tree : Tree, sum_type='transition', debug_mode=False, task_weight_empowerment_b=False, goal_location=[0,0]) -> int:
    #TODO Add a debug mode to print route information for how empowerment is calculated
    #sum type accepts the following
    #'transition'   If the move from parent to child causes a state change (default)
    #'leaf'         If the leaf has a different state than the root
    #'leaf_unique'  If the leaf has a unique state
    #'node_unique'  If a node has a unique state

    total_empowerment = 0

    #if the move from parent to child causes a state change (default)
    if sum_type=='transition':
        for node_id in tree.expand_tree(mode=Tree.DEPTH):
            parent = tree.parent(node_id)
            node = tree.get_node(node_id)
            #if not the root node (move 0)
            if parent!=None:
                if np.array_equal(parent.data.m_map, node.data.m_map):
                    #nothing changed so empowerment is the same
                    pass
                else:
                    #the state of other agents changed as a result of my action
                    if not task_weight_empowerment_b:
                        total_empowerment +=1
                    else:
                        #calculate if the new state improved on the old state
                        if isImprovedState(parent.data, node.data, goal_location):
                            total_empowerment +=1

    #if the leaf has a different state than the root
    elif sum_type=='leaf':
        root = tree.get_node(tree.root)
        for node_id in tree.expand_tree(mode=Tree.DEPTH):
            node = tree.get_node(node_id)
            #if it's a leaf
            if node.is_leaf():
                if np.array_equal(root.data.m_map, node.data.m_map):
                    #nothing changed so empowerment is the same
                    pass
                else:
                    if not task_weight_empowerment_b:
                        total_empowerment +=1
                    else:
                    #+1 to the empowerment at the LEAF nodes position (the world looked different from the final position than it did from the start)
                        if isImprovedState(root.data, node.data, goal_location):
                            total_empowerment +=1
    
    #if node has a unique state or the subset where leaf has a unique state:
    elif sum_type=='node_unique' or sum_type=='leaf_unique':
        #get the root node of the tree
        root = tree.get_node(tree.root)
        #initalise a big list to hold the world state seen at each node
        #the dimensions of the arrays are important, they need to be stacked along the first dimension
        all_states = np.array([root.data.m_map])
        leaf_states = np.array([root.data.m_map])
        all_states_local2global = np.array(root.data.m_local2global)
        leaf_states_local2global = np.array(root.data.m_local2global)
        #populate the stack of states by cycling through each node (depth first)
        for node_id in tree.expand_tree(mode=Tree.DEPTH):
            node = tree.get_node(node_id)
            all_states = np.concatenate((all_states, np.array([node.data.m_map])), axis=0)
            all_states_local2global = np.concatenate((all_states_local2global, np.array(node.data.m_local2global)), axis=0)
            if node.is_leaf():
                leaf_states = np.concatenate((leaf_states, np.array([node.data.m_map])), axis=0)
                leaf_states_local2global = np.concatenate((leaf_states_local2global, np.array(node.data.m_local2global)), axis=0)
        #extract the unique states and count to give empowerment
        if sum_type=='leaf_unique':
            leaf_states_unique,idx = uniqueByFirstNp(leaf_states)
            if not task_weight_empowerment_b:
                total_empowerment = leaf_states_unique.shape[0]
            else:
            # itterate through each unique node and increase empowerment if it's better than the root
                leaf_states_local2global = leaf_states_local2global[idx,:]
                i=0
                for state in leaf_states_unique:
                    ndata = NodeState(sensor_state=state,  local2global_mapping=leaf_states_local2global[i,:])
                    if isImprovedState(root.data, ndata, goal_location):
                        total_empowerment +=1
                    i+=1

        else:
            all_states_unique,idx = uniqueByFirstNp(all_states)
            if not task_weight_empowerment_b:
                total_empowerment = all_states_unique.shape[0]
            else:
               # itterate through each unique node and increase empowerment if it's better than the root
               all_states_local2global = all_states_local2global[idx,:]
               i=0
               for state in all_states_unique:
                    ndata = NodeState(sensor_state=state,  local2global_mapping=all_states_local2global[i,:])
                    if isImprovedState(root.data, ndata, goal_location):
                        total_empowerment +=1 
                    i+=1

    #requested method to sum empowerment isn't recognised
    else:
        print(f"ERROR sum type {sum_type} not recognised")
        total_empowerment = 0

    return total_empowerment
    #END sumEmpowerment


def isImprovedState(old_data, new_data, goal_position):
    '''
    Returns true if the new state is an improvement on the old state.
    The criteria for true is if the cumulative distance of the sheep from the goal has been reduced
    '''
    #extract the sheep positions from the map data and format them into a list of points
    sheep_positions= np.nonzero(old_data.m_map==SHEEP_TYPE) 
    #numpy indexes positions with a touple of 2 arrays, one for rows and 1 for columns so need to manpiulate it to get
    #it into a vector of 2d positions
    sheep_positions_old = np.asarray(sheep_positions).transpose()
    sheep_positions= np.nonzero(new_data.m_map==SHEEP_TYPE) 
    sheep_positions_new = np.asarray(sheep_positions).transpose()

    #convert the positions to global coordinates to match the goal_position
    sheep_positions_old = sheep_positions_old + np.asarray(old_data.m_local2global)
    sheep_positions_new = sheep_positions_new + np.asarray(new_data.m_local2global)

    #calculate the cumlative distance from the sheep to the goal
    score_old = np.sum(np.linalg.norm(sheep_positions_old - goal_position, axis=1))        
    score_new = np.sum(np.linalg.norm(sheep_positions_new - goal_position, axis=1))

    return np.less(score_new,score_old)
    

# if __name__ == '__main__':
    #TODO: Add a unit test or example behaviour

# %%
