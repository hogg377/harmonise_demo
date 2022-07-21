from math import floor
from xml.dom.expatbuilder import FragmentBuilderNS
from numpy.core.defchararray import asarray
# from pygame.constants import WINDOWHITTEST
from treelib import exceptions
import model.Sheep as Sheep
import model.DogHeuristic as Dog
import model.World as World
import model.Population as Population
import colours, logging, pygame, sys, VideoRecorder
import numpy as np
from datetime import datetime
import os

#--------CONSTANTS--------
FULL_SCREEN = False
INITIAL_PAUSE_DURATION = 3000


#--------GLOBAL VARIABLES--------
screen = ''
log_path = os.path.join(os.path.expanduser('~'), "OneDrive - University of Bristol", "Empowerment Results")


#--------CLASSES--------
class Text(pygame.sprite.Sprite):
    """ 
        Used to display text values on a pygame screen
        Adapted from https://stackoverflow.com/questions/23056597/python-pygame-writing-text-in-sprite
    """
    
    def __init__(self, text, position, font_size=20, color=[0,0,0], box_width=-1, box_height=-1):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self)

        # Create a text surface and render the text on it
        self.font = pygame.font.SysFont("Arial", font_size)
        textSurf = self.font.render(text, 1, color)
        
        # This creates the text with a transparant background, to create it with a coloured one,
        if box_width==-1:
            self.image = textSurf

        # User provides a box width so create a bounding box and overlay the text
        else:
            self.image = pygame.Surface((box_width, box_height))
            # TODO: add switch to change the bounding box colour
            self.image.fill(colours.WHITE)
            W = textSurf.get_width()
            H = textSurf.get_height()
            self.image.blit(self.textSurf, [box_width/2 - W/2, box_height/2 - H/2])
        self.rect = self.image.get_rect()
        self.rect.center = position
        return
#end class


#--------FUNCTIONS--------
def text_objects(text, font):
    """ Returns text and the rectangle containing it.
    See here: https://pythonprogramming.net/displaying-text-pygame-screen/ """
    textSurface = font.render(text, True, colours.BLACK)
    return textSurface, textSurface.get_rect()
#end function


def createFlock(world : World, group : pygame.sprite.Group, positions : list, types : list, sheep_cfg : dict, square_size : int):
    """ Creates a sheep agent for each position and adds it to the group"""
    n_sheep = len(positions)

    for i in range (n_sheep):
        # Create a sheep at a dummy position of 1,1
        sheep = Sheep.Sheep((1,1), types[i])
        
        # Set a default image for the sheep
        img = pygame.Surface((square_size,square_size))
        img.fill(colours.BLACK)
        sheep.setImage(img)
        
        # Check to make sure sheep isn't colliding with other sheep in the group
        # REMOVED on the basis positions provided in the configuration files 
        #   and should have been checked for things like collisions with other sheep, walls etc
        # collision = 1
        # while collision:
        #     #test for collisions
        #     collided_sheep = pygame.sprite.spritecollide(sheep, group.m_agents, False)
        #     if len(collided_sheep) > 0:
        #         #try to find an adjacent square which is free
        #         x = random.randint(0,1)
        #         y = random.randint(0,1)
        #         adj_pos = pos + pygame.math.Vector2(x,y) * square_size
        #         sheep.setPosition = adj_pos
        #         collision = 1
        #     else:
        #         #check for collision with dogs, walls using the above as a templateetc
        #         collision = 0
        # Use the world reference to check whether position is clear

        # Configure the sheep
        sheep.setConfig(sheep_cfg)

        # Add the sheep to the flock
        group.addAgent(sheep)

        # Try to add the sheep to the world at the position required.  This sets the sheep to a valid position and record it in the world
        position_set = sheep.move(world, positions[i])
        if not(np.array_equal(position_set, positions[i])):
            raise Exception(f'unable to initialise sheep {i} at position {positions[i]}.  Has an agent already been initialised there?')

        world.log.addNewAgentInLog('sheep', sheep.m_id, position_set, world.tick)
    return
#end function


def createDogs(world : World, group : pygame.sprite.Group, positions : list, types : list, dog_cfg : dict, square_size: int):
    """ Creates a dog agent for each position and adds it to the group """
    n_dogs = len(positions)
    for i in range (n_dogs):
        # Create a dog at a dummy position of 1,1
        dog = Dog.Dog((1,1),types[i])

        # Set a default image for the dog
        img = pygame.Surface((square_size,square_size))
        img.fill(colours.RED)
        dog.setImage(img)

        # TODO: check to make sure dog isn't colliding with other dogs, walls, agents etc

        # Configure the dog
        dog.setConfig(dog_cfg)

        # Add the dog to the pack
        group.addAgent(dog)

        # Try to add the dog to the world at the position required.  This sets the dog to a valid position and record it in the world
        position_set = dog.move(world, positions[i])
        if not(np.array_equal(position_set, positions[i])):
            raise Exception(f'unable to initialise dog {i} at position {positions[i]}.  Has an agent already been initialised there?')

        world.log.addNewAgentInLog('dog', dog.m_id, position_set, world.tick)
    return
#end function


def removeDog(world, group, position):
    # Removes the closet dog to a position within a radius
    # TODO: Expand the function to remove the closest agent to the click not the first one it finds within the treshold range
    THRESHOLD=5
    # Find the dog
    n_removed = 0
    for d in group.m_agents:
        d_to_pos = np.linalg.norm(np.asarray(position) - np.asarray(d.m_position))
        if d_to_pos<THRESHOLD:
            # Removing an agent from its parent population should delete the instance provided no other references to it exist but the
            #   Kill method is more reliable (removes it from all populations it's a member of and deletes the instance)
            # group.m_agents.remove(d)
            d.kill()
            n_removed+=1
            world.log.destroyAgentInLog('dog', d.m_id, position, world.tick)
            # Do a return here to ensure we don't remove move than 1
            return n_removed
    return n_removed 
#end function


def empowermentValues(dogs):
    FONTSIZE = 15
    COLOUR = [0,0,0]
    text_sprites = pygame.sprite.Group()
    for i,dog in enumerate(dogs):
        text = str(dog.m_empowerment)
        # Agent sprite positions are pygame vectors
        position = dog.m_sprite_position + (FONTSIZE,0)
        sprite = Text(text, position, FONTSIZE, COLOUR)
        text_sprites.add(sprite)

    return text_sprites
#end function


def updateScore(gw, score, goal_position, n_dogs, goal_rect, local2screen_scaling=1):

    # MAX_DISTANCE_FROM_GOAL = 6
    sheep_positions = gw.getSheepPositions()
    com = gw.calcCoM(sheep_positions)

    # Calculate the new score
    new_score = score + 1 / (np.linalg.norm(com - goal_position)*n_dogs)

    # Test if the task has been completed - OLD VERSION
    # TODO: make this test for task completion part of the game world and alter dog class to use it
    #   This code is currently a cut and paste of the code used in the dog class function planState()
    # max_distance_from_com = 0.5*len(sheep_positions)
    # if ( np.linalg.norm(com - goal_position) < MAX_DISTANCE_FROM_GOAL and
    #     np.max(np.linalg.norm(sheep_positions-com,axis=1))<max_distance_from_com ):
    #     task_complete_b = True
    # else:
    #     task_complete_b = False

    # Test if the task has been completed - NEW VERSION
    #Loop through each sheep positions and use pygame's collision detection function to detect if a point is inside a rectangle
    # (could also do via arrays and numpy's "any" function to check all points, it might be faster...)
    task_complete_b = True
    for p in sheep_positions:
        if not goal_rect.collidepoint(p*local2screen_scaling):
            task_complete_b = False
            break

    return new_score, task_complete_b
#end function


def playerInstructions(text_lines, origin):
    # Creates a set of sprites to represent any text or instructions to be displayed to the player
    #   text_lines is a list of lines and origin is the topleft corner of the box they appear in
    FONTSIZE = 20
    COLOUR = colours.BLACK
    text_sprites = pygame.sprite.Group()
    for i, text in enumerate(text_lines):
        sprite = Text(text, (origin[0],origin[1]+FONTSIZE*i), FONTSIZE, COLOUR)
        text_sprites.add(sprite)
    return text_sprites
#end function


#FUNCTION REMOVED
def button(msg,x,y,w,h,ic,ac,fontsz=20,action=None):
    # from https://pythonprogramming.net/pygame-button-function-events/
    global screen
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    print(click)
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(screen, ac,(x,y,w,h))

        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(screen, ic,(x,y,w,h))

    smallText = pygame.font.SysFont("comicsansms",fontsz)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ( (x+(w/2)), (y+(h/2)) )
    screen.blit(textSurf, textRect)
#end function


def changeEmpowermentMethod(new_method, world, pack):
    if (new_method == 'transition' or new_method == 'leaf' or new_method == 'leaf_unique' or new_method == 'node_unique' or new_method == 'toggle_weighted'):
        for dog in pack.m_agents:
            if new_method == 'toggle_weighted':
                dog.m_empowerement_task_weighted_b = not dog.m_empowerement_task_weighted_b
            else:
                # Change the dog's config so it remembers the change
                dog.m_empowerment_sum_method = new_method
            # Ask it to recalculate its empowerment value and remember the result
            dog.m_empowerment = dog.calcEmpowerment(world, use_task_weighted =dog.m_empowerement_task_weighted_b)
    else:
        raise Exception(f"Empowerment method {new_method} is not a recognised method")
#end function

#--------MAIN FUNCTION--------
def main(config_name='experiment_config_files.config_exp_1', show_empowerment=False, use_task_weighted_empowerment=False, passed_screen='', sim_session_id='000000T000000', log_file_name=''):
    global screen
    # Set whether to record the results
    RECORD_VIDEO = True
    # Set true to cause the agents to take their turn in random order
    RANDOM_UPDATES = True
    # This is always true unless it's running without user interaction e.g. as a pure simulation
    UPDATE_SPRITES = True
    # The amount to increase the tick rate by when the participant clicks the up/down button, used for debugging, removed in final version
    # TICK_RATE_INTERVAL = 5

    # Logging setup - TODO: We ought to make this produce unique logs based on datetime
    # for actual experiments.
    # Logging levels: CRITICAL > ERROR > WARNING > INFO > DEBUG
    print(f"I'm running config {config_name} with empowerment set to {show_empowerment}")
    logging.basicConfig(
        filename='EmpoweredHerding.log', level=logging.DEBUG,
        format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S'
    )
    logging.info("Simulation started: config_name: {} - show_empowerment: {}".format(config_name, show_empowerment))

    # Load the config
    import importlib
    config = importlib.import_module(config_name)
    cfg = config.main()
    cfg['dog']["show_empowerment_b"] = show_empowerment
    cfg['dog']['use_taskweighted_empowerment_b'] = use_task_weighted_empowerment

    # Create the world
    gw = World.World(cfg['world']['width'], cfg['world']['height'], 2)

    # Setup logging
    gw.log.initialise(sim_session_id, config_name, show_empowerment, use_task_weighted_empowerment)

    # Create the populations
    pack = Population.Population(1)
    flock = Population.Population(10000)

    # Create the sheep
    createFlock(gw, flock, cfg['sheep']['initial_positions'], cfg['sheep']['types'],
        cfg['sheep'], cfg['world']['square_size'])

    # Create the dogs
    createDogs(gw, pack, cfg['dog']['initial_positions'], cfg['dog']['types'],
        cfg['dog'], cfg['world']['square_size'])
    n_dogs_created = len(cfg['dog']['initial_positions'])

    # Set the genome
    # config now set inside the config file. OK for now but won't work if this is used for artificial evolution!

    # Setup the world
    #   Agents now do this by performing a "move" when they're created.  Shows any errors with the initial positions
    #   the alternative is to do: gw.setGrid(cfg['sheep']['initial_positions'] + cfg['dog']['initial_positions'], )

    # Set the size and create the pygame screen for visualisation
    SCREEN_SCALING =  cfg['world']['square_size']
    sidebar_width = 150
    screen_width = cfg['world']['width'] * SCREEN_SCALING  + sidebar_width
    screen_height = cfg['world']['height'] * SCREEN_SCALING  #+150
    if FULL_SCREEN:
        screen =pygame.display.set_mode((1280,720), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode([screen_width,screen_height])

    # Set the origin and size of the playing area
    #   i.e., the part of the window where the flock moves
    arena_width = cfg['world']['width'] * SCREEN_SCALING
    arena_height =  cfg['world']['height'] * SCREEN_SCALING
    # LMARGIN = screen_width - arena_width - sidebar_width
    LMARGIN=0
    sidebar_origin = (arena_width+LMARGIN,0)

    # Initalises all the pygame modules for us, this checks it hasn't already been done by a calling module
    if pygame.get_init()==False:
        pygame.init()

    # Create any text instructions to be displayed on the window as sprites
    # NOTE: there are other ways to do this withtout using sprites although pygame is optimised to work with sprites
    instructions_text = playerInstructions(
        [
            'LMB:Add Dog', 'RMB:Del Dog',
            'UP: S-', 'DOWN: S+',
            'LEFT: S--', 'RIGHT: S++',
            'Space:Pause', '      /Unpause', '',
            'Select Method', '1:transition',
            '2:leaf','3:leaf_unique','4:node_unique','0:toggle_weight'
        ],
        (sidebar_origin[0]+10, 25)
    )
    game_state_text = playerInstructions(['PAUSED'], (sidebar_origin[0]+50, screen_height-50))
    
    # Left align the instructions
    for text in instructions_text:
        text.rect.left = sidebar_origin[0]+20

    # Create the static graphical elements
    w=cfg['world']['goal_width'] * SCREEN_SCALING
    h=cfg['world']['goal_height'] * SCREEN_SCALING
    goal_rectangle = pygame.Rect(cfg['dog']['endgoal_position'][0] * SCREEN_SCALING-w/2, cfg['dog']['endgoal_position'][1] * SCREEN_SCALING-h/2, w, h)
    arena_boundary = pygame.Rect(LMARGIN, 0, cfg['world']['width'] * SCREEN_SCALING,  cfg['world']['height'] * SCREEN_SCALING)

    # Create the "kennel"
    kennel_origin = (sidebar_origin[0]+20,sidebar_origin[1]+50)
    dogs_in_kennel_rects = []
    kennel_rect = pygame.Rect(kennel_origin[0],kennel_origin[1], sidebar_width - 50,  sidebar_width - 50)
    import math
    for i in range(0, cfg['dog']['n_dogs_max']):
        top = kennel_origin[1] + 15 + (math.floor(i/3) * cfg['world']['square_size'] * 3)
        left = kennel_origin[0] + 15 + ((i%3) * cfg['world']['square_size'] * 3)
        dogs_in_kennel_rects.append(pygame.Rect(left, top, cfg['world']['square_size'], cfg['world']['square_size']))

    # If recording video then set up the video recorder
    if RECORD_VIDEO:
        (screen_width,screen_height)= screen.get_size()
        resolution = (screen_width, screen_height)
        filename = "Recording3.avi"
        video = VideoRecorder.VideoRecorder()
        fps = 10
        video.setConfig(filename, fps, resolution)
        video.filename = filename
        video.startRecorder()

    # Create the clock to track time
    clock = pygame.time.Clock()

    # To simplify the update code, two ticks of the model are needed for both the sheep and dogs to move
    TIME_LIMIT = cfg['sim']['time_limit'] * 2
    TICK_RATE = cfg['sim']['tick_rate']
    tr_text_position = (sidebar_origin[0]+75, screen_height-80)
    
    # Create and set the position of the tick rate, score and whether weighted empowerment is being used
    tick_rate_text = playerInstructions(['TICK RATE: {}'.format(TICK_RATE)], tr_text_position)
    score_text_position = (sidebar_origin[0]+75, screen_height-120)
    toggle_text_position = (sidebar_origin[0]+75, screen_height-180)
    weighted_toggle = False

    # Initialise a counter to track game time
    ticks = 0
    gw.tick = ticks

    # Initalise vectors to keep score and empowerment
    score = np.zeros(TIME_LIMIT+1)
    mean_empowerment = np.zeros(TIME_LIMIT+1)
    b_world_changed = False

    # Intialise a random number generator to use in the game loop
    rng = np.random.default_rng()

    # Initialise a variable to track if the game is in a paused state
    #   defaults to run immediately
    game_run_b = True
    quit_selected = False

    # Initialise variables used to tracks if dogs have completed the task
    in_task_completed_state_b = False
    task_completed_b = False
    task_completed_on_tick = 0
    # task_completed_count = 0
    # TASK_COMPLETED_COUNT_LIMIT = 10

    #log the initial positions
    gw.log.logPopulationStates('dog', pack, gw.tick)
    gw.log.logPopulationStates('sheep', flock, gw.tick)
    gw.log.logPopulations([pack, flock], gw.tick)

    # Run the game
    while (ticks < TIME_LIMIT) and not quit_selected and not ( in_task_completed_state_b and (ticks - task_completed_on_tick> cfg['sim']['post_task_run_time'] ) ):

        # Handle the event queue
        for event in pygame.event.get():
            # Exit cleanly if user closes the window
            if event.type == pygame.QUIT:
                print('Exiting')
                if RECORD_VIDEO:
                    video.stopRecorder()
                logging.info("System exit.")
                # sys.exit()
                quit_selected = True

            # Handle keyboard button presses
            if event.type == pygame.KEYDOWN:
                # keyboard functions disabled for final version
                pass
            #   # Allow us to pause the game
            #   if event.key == pygame.K_SPACE and not game_run_b:
            #        logging.info("Tick {}: USER INPUT: KEY DOWN: SPACE".format(ticks))
            #        if game_run_b:
            #           game_state_text = playerInstructions(['PAUSED'], (sidebar_origin[0]+50, screen_height-50))
            #        else:
            #            game_state_text = playerInstructions(['RUNNING'], (sidebar_origin[0]+50, screen_height-50))
            #        game_run_b = not game_run_b
            #     # Allow us to play with the tick_rate of the simulation while we figure stuff out.
            #     elif event.key == pygame.K_LEFT:
            #         logging.info("Tick {}: USER INPUT: KEY DOWN: LEFT".format(ticks))
            #         TICK_RATE -= TICK_RATE_INTERVAL
            #         tick_rate_text = playerInstructions(['TICK RATE: {}'.format(TICK_RATE)], tr_text_position)
            #     elif event.key == pygame.K_RIGHT:
            #         logging.info("Tick {}: USER INPUT: KEY DOWN: RIGHT".format(ticks))
            #         TICK_RATE += TICK_RATE_INTERVAL
            #         tick_rate_text = playerInstructions(['TICK RATE: {}'.format(TICK_RATE)], tr_text_position)
            #     elif event.key == pygame.K_UP:
            #         logging.info("Tick {}: USER INPUT: KEY DOWN: UP".format(ticks))
            #         TICK_RATE += 1
            #     elif event.key == pygame.K_DOWN:
            #         logging.info("Tick {}: USER INPUT: KEY DOWN: DOWN".format(ticks))
            #         TICK_RATE -= 1
            #     # end tick_rate
            #     elif event.key==pygame.K_1:
            #         logging.info("Tick {}: USER INPUT: KEY DOWN: 1".format(ticks))
            #         changeEmpowermentMethod('transition', gw, pack)
            #     elif event.key==pygame.K_2:
            #         logging.info("Tick {}: USER INPUT: KEY DOWN: 2".format(ticks))
            #         changeEmpowermentMethod('leaf', gw, pack)
            #     elif event.key==pygame.K_3:
            #         logging.info("Tick {}: USER INPUT: KEY DOWN: 3".format(ticks))
            #         changeEmpowermentMethod('leaf_unique', gw, pack)
            #     elif event.key==pygame.K_4:
            #         logging.info("Tick {}: USER INPUT: KEY DOWN: 4".format(ticks))
            #         changeEmpowermentMethod('node_unique', gw, pack)
            #     elif event.key==pygame.K_0:
            #         logging.info("Tick {}: USER INPUT: KEY DOWN: 0".format(ticks))
            #         changeEmpowermentMethod('toggle_weighted', gw, pack)
            #         weighted_toggle = not weighted_toggle

            #     if TICK_RATE <= TICK_RATE_INTERVAL:
            #         TICK_RATE = TICK_RATE_INTERVAL
            #     elif TICK_RATE >= 10 * TICK_RATE_INTERVAL:
            #         TICK_RATE = 10 * TICK_RATE_INTERVAL

            #     tick_rate_text = playerInstructions(['TICK RATE: {}'.format(TICK_RATE)], tr_text_position)

            # Handle mouse clicks
            if event.type == pygame.MOUSEBUTTONDOWN and game_run_b:
                # Check if the button click was on the window
                is_focused_b = pygame.mouse.get_focused()
                
                # Convert the window position to a grid position
                #   (Note, the vertical and horizonal axis are swapped on the screen relative to the grid world's frame of reference
                #   and the screen is scalled up relative to the world grid)
                grid_pos = np.array((event.pos[1],event.pos[0]),dtype=int) / cfg['world']['square_size']
                
                # Grid positions must be integers
                # TODO: Add a catch in the gridworld and agent functions to ensure positions are integers
                grid_pos = np.round(grid_pos).astype(int)
                logging.info("Tick {}: USER INPUT: MOUSE DOWN: pos {}: grid_pos {}.".format(ticks, event.pos, grid_pos))

                # If the left mouse is clicked then spawn a dog at the click locaiton
                if cfg['dog']['spawn_radius'] == 0:
                    spawn_pos = grid_pos
                else:
                    spawn_pos = grid_pos + rng.integers(low=-cfg['dog']['spawn_radius'], high=cfg['dog']['spawn_radius'], size=2)
                if event.button==1 and gw.isValidGridPosition(spawn_pos) and gw.isEmptyGridPosition(spawn_pos):
                    gw.log.user_log.addMouseClick(ticks, datetime.now(), "MB:DOWN:LEFT", event.pos, grid_pos)
                    if len(pack.m_agents) < cfg['dog']['n_dogs_max']:
                        createDogs(gw, pack, [spawn_pos], [1],  cfg['dog'], cfg['world']['square_size'])
                        n_dogs_created +=1
                        logging.info("Tick {}: Dog spawned: spawn_pos {}.".format(ticks, spawn_pos))
                if event.button==3:
                    gw.log.user_log.addMouseClick(ticks, datetime.now(), "MB:DOWN:RIGHT", event.pos, grid_pos)
                    if len(pack.m_agents) > cfg['dog']['n_dogs_min']:
                        n_dogs_removed = removeDog(gw,pack,grid_pos)
                        n_dogs_created -=n_dogs_removed
                        logging.info("Tick {}: Dog despawned: grid_pos {}.".format(ticks, grid_pos))
                
                # Print debug info
                logging.debug(f"focus is {is_focused_b}, button pressed is {event.button} and mouse at ({event.pos}")

        # Run the simulation.
        if game_run_b == True:
            # Draw the initial agent positions
            if ticks == 0:
                flock.doNothing(gw, UPDATE_SPRITES , SCREEN_SCALING)
                pack.doNothing(gw, UPDATE_SPRITES , SCREEN_SCALING)
                # Record when the first tick took place
                gw.log.recordStartTime(datetime.now())
                b_world_changed = True
            # Sheep take a turn on an even tick (but not the first turn) 
            elif (ticks % (cfg['dog']['dog_2_sheep_speed']+1))==0:
                pack.doNothing(gw, UPDATE_SPRITES , SCREEN_SCALING)
                flock.update(gw, RANDOM_UPDATES, UPDATE_SPRITES, SCREEN_SCALING)
                b_world_changed = True
            # Dogs take a turn on an odd tick (includes the first turn)
            else: # (ticks % 2)==0:
                flock.doNothing(gw, UPDATE_SPRITES , SCREEN_SCALING)
                pack.update(gw, RANDOM_UPDATES, UPDATE_SPRITES , SCREEN_SCALING)
                b_world_changed = True

            # NOTE: the grid needs to be advanced before scores and logging otherwise these are based on the previous time step.
            #   This causes problems if the game is set to run immeidately (ie default is unpaused at the start)
            gw.advanceGrid()

            # Advance game time (not actual time)
            ticks += 1
            gw.tick = ticks
            (score[ticks], task_completed_b) = updateScore(gw, score[ticks-1], cfg['dog']['endgoal_position'], len(pack.m_agents), goal_rectangle, SCREEN_SCALING)
            
            e = 0
            for dog in pack.m_agents:
                e+=dog.m_empowerment
            try:
                mean_empowerment[ticks] = e / len(pack.m_agents)
            except ZeroDivisionError:
                mean_empowerment[ticks] = e
            # Log the world state at the end of the update to ensure the final tick is captured when the simulation ends
            gw.log.logPopulationStates('dog', pack, gw.tick)
            gw.log.logPopulationStates('sheep', flock, gw.tick)
            gw.log.logPopulations([pack, flock], gw.tick)

        # Game paused, redraw to ensure any dogs removed or spawned appear on screen
        else:
            flock.doNothing(gw, UPDATE_SPRITES , SCREEN_SCALING)
            pack.doNothing(gw, UPDATE_SPRITES , SCREEN_SCALING)
            b_world_changed = True
            # Draw the current world state
            # NOTE: previous implementation would only redrew the screen if something has changed.  This was more efficient but meant events needed to be carefully tracked.
            #   e.g., things like spawning or removing a dog would need to be followed by flock.doNothing, pack.doNothing if the game was paused but not otherwise.
            #   It's more straightforward to just draw the display every single time but potentially more of a resource hog.
            # if b_world_changed == True:
            gw.advanceGrid()
        # end game_run


        # Instructions, scores etc were removed after development completed
        # Update the score text
        # score_text = playerInstructions(['Score: {}'.format(round(score[ticks],4))], score_text_position)
        # toggle_text = playerInstructions(['W_Emp: {}'.format("On" if weighted_toggle else "Off")], toggle_text_position)

        # Blank the screen buffer
        screen.fill(colours.WHITE)

        # Draw the agents
        flock.m_agents.draw(screen)
        pack.m_agents.draw(screen)
        if cfg['dog']['show_empowerment_values_b']:
            empowerment_values = empowermentValues(pack.m_agents)
            empowerment_values.draw(screen)

        # Draw static screen elements
        # Instructions, scores etc were removed after development completed
        # instructions_text.draw(screen)
        # tick_rate_text.draw(screen)
        # game_state_text.draw(screen)
        # score_text.draw(screen)
        # toggle_text.draw(screen)
        pygame.draw.rect(screen, colours.RED, goal_rectangle, 2)
        pygame.draw.rect(screen, colours.BLACK, arena_boundary, 2)
        pygame.draw.rect(screen, colours.BLACK, kennel_rect, 2 )
        for i in range(0, cfg['dog']['n_dogs_max']-len(pack.m_agents)):
             pygame.draw.rect(screen, colours.BLUE, dogs_in_kennel_rects[i])

        # Save the frame if recording a video of the run
        if RECORD_VIDEO and game_run_b == True:
            # Screenshot the current pygame screen
            img = pygame.surfarray.array3d(screen)
            # Convert the screenshot to a numpy array
            frame = np.array(img)
            frame = np.fliplr(frame)
            frame = np.rot90(frame)
            video.grabScreen(frame)

        # Flip the buffers to show the updated screen
        pygame.display.flip()
        # Sets the update speed of the simulation.  A lower number is slower
        clock.tick(TICK_RATE)
        b_world_changed = False

        # Test if the task is complete - OLD VERSION, REPLACED AFTER BETA TESTS
        #   The task is complete when ALL sheep have been inside the goal rectangle for a continous count of TASK_COMPLETED_COUNT_LIMIT
        # Keep track of how many ticks have elapsed since all the sheep were last inside the goal rectangle
        # if task_completed_b:
        #     task_completed_count+=1
        # else:
        #     task_completed_count = 0
        # in_task_completed_state_b = True will terminate the main while loop and the simulation will move to post task clean up
        # if task_completed_count > TASK_COMPLETED_COUNT_LIMIT:
        #     in_task_completed_state_b = True
        #     task_completed_on_tick = ticks
        # else:
        #     in_task_completed_state_b = False

        # Test if the task is complete - NEW VERSION, INTRODUCED AFTER BETA TESTS
        #   The task is complete when ALL sheep have been inside the goal rectangle for a continous count of cfg['sim']['post_task_run_time'].
        #   This is checked in the main while loop condition: (ticks - task_completed_on_tick> cfg['sim']['post_task_run_time'] ).
        if task_completed_b and not in_task_completed_state_b:
            in_task_completed_state_b = True
            task_completed_on_tick = ticks
        elif not task_completed_b and in_task_completed_state_b:
            in_task_completed_state_b = False
    #end while loop


    # Record the time when the game ended
    gw.log.recordEndTime(datetime.now())

    # Stop recording video and save
    if RECORD_VIDEO:
        video.stopRecorder()

    #---------SAVE THE SIMULATION LOG-------------
    import os
    if not os.path.exists(os.path.join(log_path, sim_session_id)):
        os.makedirs(os.path.join(log_path, sim_session_id))
   
    # Create a meaningful name for the log file if one isn't provided
    # NOTE: This code is identical to a block in run_simulation() in main.py
    if not log_file_name:
        log_name = config_name
        if show_empowerment:           
            log_name = log_name + "_empshown"       
        if use_task_weighted_empowerment:
            log_name = log_name + "_taskweighted"
    else:
        log_name = log_file_name
    
    # Save the log to disk
    gw.log.pickleLog(os.path.join(log_path, sim_session_id, "{}_simlog".format(log_name)))

    # -----------DISPLAY A PLOT OF THE SCORE------------
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(score, color='blue')
    # plt.title("Hybrid Team Performance Over Time", fontsize=16)
    # ax.set_xlabel("time (ticks)", fontsize=14)
    # # ax.set_ylim([0, 15])
    # ax.set_ylabel("herding score", color='blue', fontsize=14)

    # ax2 = ax.twinx()
    # ax2.plot(mean_empowerment, color='red')
    # # ax2.set_ylim([0, 50])
    # ax2.set_ylabel("total empowerment", color='red', fontsize=14)
    # plt.show()
    
    #exit and return a copy of the simulaiton logs
    return gw.log
#end function


if __name__ == '__main__':
    main()