"""
A gui for the empowerment simulation which also runs the experiment via a series of menus.

Menu IDS are:
  -1: quit
   0: start menu
  10: information menu 1  (Removed following feedback from beta testers May 2022)
  11: information menu 2  (Removed following feedback from beta testers May 2022)
  12: information menu 3  (Removed following feedback from beta testers May 2022)
  13: information menu 4  (Removed following feedback from beta testers May 2022)
  14: information menu 5  (Removed following feedback from beta testers May 2022)
  20: instructions menu
  30: tutorial start page
  31: tutorial part 1 instructions
  32: tutorial part 2 instructions
  33: tutorial part 3 instructions
  34: tutorial part 4 instructions
  35: tutorial complete
  40: experimental block 1
  50: experimental block 2
  60: participants details
  70: debrief             (Removed following feedback from beta testers May 2022)
  80: final consent form  (Removed following feedback from beta testers May 2022)
  90: trial start
  91: post trial questions 1 (sliders)
  92: post trial questions 2 (sliders)
"""

#----------------EXTERNAL MODULES-----------------
import pygame
import pygame_menu
# import runSimulation as sim
import colours
import gspread
# import model.MenuLog as MenuLog
import os
import sys
#-------------------------------------------------

#------------------- MY libraries -------------------

import simulation.map_gen as map_gen
import simulation.environment
import simulation.asim as asim
import simulation.faulty_swarm as faulty_swarm
import model.dataLogger as dataLogger
import model.SimLog as SimLog

import random
import numpy as np
import pickle
import sys
import time 
import importlib       


import matplotlib.pyplot as plt
from matplotlib import animation, rc, rcParams
rcParams['animation.embed_limit'] = 2**128
rcParams['toolbar'] = 'None' 

from matplotlib import collections  as mc
import matplotlib as mpl
from matplotlib import image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


from scipy.spatial.distance import cdist, pdist, euclidean



# -----------------DIRECTORIES, PATHS AND CONFIGURATIONS-----------------------
# Set the configuration and results directories folders
CONFIG_DIR = "experiment_config_files/version2/"
# RESULTS_DIR = os.path.join(os.path.expanduser('~'), "OneDrive - University of Bristol", "Empowerment Results")
RESULTS_DIR = os.path.join('Results/')
# print('Result DIR: ', RESULTS_DIR)

# Credentials needed to log to a google sheet
#   This function was removed at the beta stage
# from oauth2client.service_account import ServiceAccountCredentials
# GOOGLE_SHEET_NAME = "empowerment_data" # save_details() has been removed
# CREDENTIALS_FILE_PATH = 'halogen-order-334818-44181576e631.json' save_details() has been removed

# Set the configuration files to present to the participant
# LIVETEST_SEQUENCE_A = [
#      'config_exp_5', 'config_exp_6',
#     'config_exp_7', 'config_exp_8', 'config_exp_9',
#     'config_exp_10', 'config_exp_11', 'config_exp_12',
#     'config_exp_13', 'config_exp_14', 'config_exp_15'
# ]
# LIVETEST_SEQUENCE_B = [
#      'config_exp_5', 'config_exp_6',
#     'config_exp_7', 'config_exp_8', 'config_exp_9',
#     'config_exp_10', 'config_exp_11', 'config_exp_12',
#     'config_exp_13', 'config_exp_14', 'config_exp_15'
# ]

# reduce to two options of one faulty trial and one malicious trial
LIVETEST_SEQUENCE_A = ['config_exp_7', 'config_exp_10']
LIVETEST_SEQUENCE_B = ['config_exp_7', 'config_exp_10']


TUTORIAL_SEQUENCE_A = ['config_fam_1', 'config_fam_2', 'config_fam_3', 'config_fam_4']

global answer_dict
answer_dict = {1: 'healthy', 2: 'healthy', 3: 'healthy',
    4: 'healthy', 5: 'faulty', 6: 'faulty',
    7: 'faulty', 8: 'malicious', 9: 'malicious',
    10: 'malicious', 11: 'malicious', 12: 'malicious',
    13: 'faulty', 14: 'faulty', 15: 'faulty'}



# Add config directory to all config files:
LIVETEST_SEQUENCE_A = [string for string in LIVETEST_SEQUENCE_A]
LIVETEST_SEQUENCE_B = [string for string in LIVETEST_SEQUENCE_B]
TUTORIAL_SEQUENCE_A = [string for string in TUTORIAL_SEQUENCE_A]

# Add folder to system path so config modules can be found and .
sys.path.append(CONFIG_DIR)
# ------------------------------------------------------------------


# --------------GLOBAL VARIABLES------------
# These initial values shouldn't be changed unless you know what you're doing!!
DEBUG_MODE_B = False
SCREEN_RESOLUTION = (1280, 720)
menu_screen = pygame.display.set_mode((1280, 720))
current_menu_id = 1
session_id = '000000T000000'

# tracks which test in the sequence is being undertaken
test_number = 0

# logging
menu_log = []
global current_config
global last_config_run
last_config_run = 'none'

# font sizes
title_size = 28
text_size = 22
max_char = 110
button_size = 28
our_theme = pygame_menu.themes.THEME_BLUE

# --------------FUNCTIONS------------
def generate_session_id():
    global session_id
    from datetime import datetime
    session_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    os.mkdir(RESULTS_DIR + session_id)
#end function


def create_start_screen(full_screen_b=False):
    # Setup the screen 
    if full_screen_b:
        return pygame.display.set_mode((0, 0))
    else:
        return pygame.display.set_mode(SCREEN_RESOLUTION)
#end function


def set_difficulty(value, difficulty):
    # Do the job here !
    pass
#end function


def set_menu_id(menu_id, menu=[], save_details_b=False):
    """"
    Handles the change of one menu to the next.  

    menu_id: the ID of the menu to change to
    menu: the instance of the current menu
    save_details_b: if True then any user completed fields on the current menu will be logged
    """
    global current_menu_id
    global menu_log
    global last_config_run
    global control_active
    global current_config

    # If the current menu is directly related to a simulation then keep a record of the config name used in that simulation
    if current_menu_id == 92 or current_menu_id == 91:
        current_config = last_config_run
    else:
        current_config = 'none'

    # If true then pass the calling menu through a function which extracts any user inputs and stores them in a format for easy recall and logging
    #   The menu ID MUST be known to menu_log
    # if save_details_b:
    #     menu_log.save_responses(menu, current_menu_id, current_config)
    if save_details_b:
        menu_log.save_responses(menu, current_menu_id, current_config)


    # Set the current menu id which will cause the menu rendering function to change which menu is displayed
    current_menu_id = menu_id

    # -1 is a special menu which causes the program to exit cleanly
    if menu_id == -1:
        saveAndQuit()
    return
#end function



def start_menu_setup():
    # create the menu
    global current_menu_id
    menu = pygame_menu.Menu('Welcome', 800, 500,
                        theme=our_theme)                   
    menu.add.button('Start', set_menu_id, 20, border_width=2)  # this is the information section, just renamed the button as START for congruency with the QUIT button
    # menu.add.button('Instructions', set_menu_id, 20)
    # menu.add.button('Enter Details', set_menu_id, 40)
    if DEBUG_MODE_B:
        menu.add.button('Tutorial (debug)', run_tutorial)
        menu.add.button('Play (debug)', run_simulation, 0)
    # menu.add.button ('Final Consent', set_menu_id, 50)
    # menu.add.button('Export Details', export_details, "live")
    # menu.add.button('Quit',  pygame_menu.events.EXIT)
    return menu
#end function


#THIS FUNCTION HAS BEEN REMOVED
def information_sheet1_setup():
    global menu_screen
    title1 = ("Project title: Human-Robot Teaming")
    text1 = ("    We would like to invite you to take part in our research project on human-robot teaming. "
            "You will receive a £6 Amazon voucher for taking part. "
            "Before you decide whether to participate or not, we would like you to understand why the research is being conducted "
            "and what it will involve for you. Please ask us questions if anything is unclear.\n")
    title2 = ("What is the purpose of the project?")
    text2 = ("    The project is about human-robot teams and how to make them work well, e.g., when nurses work with care robots, or when "
            "emergency teams work with search and rescue drones. When do humans that are working with autonomous robots feel part of a team? "
            "And how does this affect team performance?\n")
    title3 = ("Do I have to take part?")
    text3 = ("   Your participation in the task is entirely voluntary "
            "and is based on informed consent. We will describe the study and go through this information sheet with you before you "
            "participate and answer any questions you might have. If you agree to take part, we will then ask you to sign a consent form. "
            "You may withdraw from the study at any point, and you will not be penalised for withdrawing. If you no longer wish to take "
            "part in the experiment, any data collected up to that point will be removed from the study and deleted. All data collected "
            "within the study will be entirely confidential. Data stored electronically will be password protected, and your data will be "
            "referenced only by a participant number.\n")
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu('Information Sheet', SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(title1, max_char=max_char, font_size=title_size, align=pygame_menu.locals.ALIGN_LEFT)
    menu.add.label(text1, max_char=max_char, font_size=text_size, align=pygame_menu.locals.ALIGN_LEFT)
    menu.add.label(title2, max_char=max_char, font_size=title_size, align=pygame_menu.locals.ALIGN_LEFT)
    menu.add.label(text2, max_char=max_char, font_size=text_size, align=pygame_menu.locals.ALIGN_LEFT)
    menu.add.label(title3, max_char=max_char, font_size=title_size, align=pygame_menu.locals.ALIGN_LEFT)
    menu.add.label(text3, max_char=max_char, font_size=text_size, align=pygame_menu.locals.ALIGN_LEFT)
    menu.add.button('Continue', set_menu_id, 11, font_size=button_size)
    if DEBUG_MODE_B:
        menu.add.button('Back to Main Menu (Debug)', set_menu_id, 1, font_size=button_size)
    return menu
#end function




def instructions_menu_setup1():
    global menu_screen
    title1 = ("\nDemo Information\n")

    text1 = ("\nIn this demo, you will observe and interact with a team of virtual robots.\n\n"+

        "Your task is to identify whether this team is operating properly,\nor if it contains 'faulty' robots or 'malicious' robots.\n"+

        #"\n- Faulty robots have flaws that cause them to behave abnormally. "

        #"\n- Malicious robots deliberately try to disrupt the team. "

        "\n- When all robots are 'working properly'...\n'all robots are functioning properly\nand trying to complete the task\nto the best of their ability'\n"+
        "\n- In a swarm containing 'faulty' robots...\n'at least some robots are broken\nand can’t consistently function properly'\n"+
        "\n- In a swarm containing 'malicious' robots...\n'at least some robots are deliberately trying\nto prevent the swarm from being successful'"+

        "\n\nBoth faulty and malicious robots can tend to prevent a team from completing their task effectively.")
        

    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu('Demo Instructions', SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    # menu.add.label(title1, max_char=max_char, font_size=title_size)#, align=pygame_menu.locals.ALIGN_LEFT)
    menu.add.label(text1, max_char=max_char, font_size=text_size)  # , align=pygame_menu.locals.ALIGN_LEFT)
    # menu.add.button('Ok', set_menu_id, 30,font_size=20)
    menu.add.button('Ok', set_menu_id, 21, font_size=button_size)
    menu.add.label("\n", max_char=max_char, font_size=text_size)

    ##### !!!!!!!!!!!!! Point to run experiment, skipping tutorial
    # menu.add.button('Ok', run_experiment, font_size=button_size)
    return menu
#end function

def instructions_menu_setup2():
    global menu_screen
    title1 = ("Robot Team Task\n")

    text1 = ("A dangerous chemical accident has occurred in a small factory.\n\n"

            "A team of simple robots enters the factory to explore the entire space as quickly\n" 
            "as possible to check for people that have not been evacuated.\n\n"

            "The robots do not have a map and are not particularly sophisticated searchers – they explore at random.\n\n"

            "However, they can communicate with nearby team members to help avoid getting stuck in corners, etc. \n\n"

            "Robots let nearby team members know if their current direction of travel is successful or not.\n\n"

            "This allows a robot that is stuck to copy the direction of travel \n"
            "of a nearby team member that is moving in a good direction.\n ")
    
    # changes


    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu('Robot Team Task', SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    # menu.add.label(title1, max_char=max_char, font_size=title_size)#, align=pygame_menu.locals.ALIGN_LEFT)
    menu.add.label(text1, max_char=max_char, font_size=text_size)  # , align=pygame_menu.locals.ALIGN_LEFT)
    # menu.add.button('Ok', set_menu_id, 30,font_size=20)
    menu.add.button('Ok', run_experiment, font_size=button_size)
    menu.add.label("\n", max_char=max_char, font_size=text_size)  # , align=pygame_menu.locals.ALIGN_LEFT)

    ##### !!!!!!!!!!!!! Point to run experiment, skipping tutorial
    # menu.add.button('Ok', run_experiment, font_size=button_size)
    return menu


def tutorial_start_menu_setup():
    """ Generates a menu for explaning and starting the tutorial """
    global menu_screen
    title = "Robot Exploration Tutorial"
    text = ("This tutorial will help you become familiar with the robot team's task.\n\n"
            "The tutorial is arranged into 3 parts.\n"
            "You can repeat the tutorial until you are comfortable with the task.\n\n"
            "Press 'Ok' to Start. \n")
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu(title, SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(text, max_char=max_char, font_size=title_size)

    # Button updates the menu id, pointing to the next menu in the high level loop to display next
    menu.add.button('Ok', set_menu_id, 31, font_size=button_size)
    return menu


def tutorial_part1_setup():
    """ Generates a menu for the first part of the tutorial """
    global menu_screen
    title = "Tutorial Part 1: Robot Exploration"
    text = ("In Part 1, you will see a team of 20 very simple robots working together properly.\n\n"

            "Their joint task is to visit as much of the environment\nas possible during the fixed time available.\n"
            "Each robot attempts to move freely and avoid collisions."
            
            "\n\nWhilst some robots will often retrace the same areas and may occasionally get stuck or "
            "\nsuffer a collision, overall, the team will tend to spread out into the environment.\n\n"
            
            "Press 'Ok' to Start Part 1. \n")
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu(title, SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(text, max_char=max_char, font_size=title_size)

    # Point here to the simulation to run
    menu.add.button('OK', run_swarmsim, 32, TUTORIAL_SEQUENCE_A[0], font_size=button_size)
    # menu.add.button('Start', run_tutorial)
    if DEBUG_MODE_B:
        menu.add.button('Skip (Debug)', set_menu_id, 35)
        menu.add.button('Main Menu (Debug)', set_menu_id, 0)
    return menu
#end function


def tutorial_part2_setup():
    """ Generates a menu for the second part of the tutorial """
    global menu_screen
    title = "Tutorial Part 2: Robot Control"
    text = ("\nIn some experiment trials, you will have some control over the robot team.\n\n"
            
            "In these trials, you can direct all the robots to travel, North, or East, or South, or West \n"
            "for a short period by pressing an arrow key on your keyboard."

            "\n\nUsing the arrow keys will help the robots to explore more effectively."

            "\nPress 'Ok' to Start Part 2. \n")
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu(title, SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(text, max_char=max_char, font_size=title_size)
    control_active = True
    menu.add.button('OK', run_swarmsim, 33, TUTORIAL_SEQUENCE_A[1], font_size=button_size)

    # menu.add.button('Go', run_swarmsim, 91, '', list_of_configs, show_empowerment, use_taskweighted_empowerment)

    # menu.add.button('Start', run_tutorial)
    if DEBUG_MODE_B:
        menu.add.button('Skip (Debug)', set_menu_id, 33)
        menu.add.button('Main Menu (Debug)', set_menu_id, 0)
    return menu
#end function


def tutorial_part3_setup():
    """ Generates a menu for the third part of the tutorial """
    global menu_screen
    title = "Tutorial Part 3: Robot Collisions"
    text = ("\nProperly operating robots try hard to avoid colliding with walls and with each other, \n"
            "but collisions may still occur occasionally, particularly in crowded conditions such as \n"
            "at the start of each trial."
            
            "\n\nWhen robots collide, they briefly become stuck before they can continue exploring."
            "\nThe following simulation shows an example of this occurring in a team of \nproperly functioning robots. You do not need to press the arrow keys."
            "\n\nPress 'Ok' to start Part 3 (or you can choose to repeat Parts 1 and 2)\n")
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu(title, SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(text, max_char=max_char, font_size=title_size)
    menu.add.button('OK', run_swarmsim, 35, TUTORIAL_SEQUENCE_A[2], font_size=button_size)
    menu.add.button('Click here to repeat Parts 1 and 2', set_menu_id, 31)
    # menu.add.button('Start', run_tutorial)
    if DEBUG_MODE_B:
        menu.add.button('Skip', set_menu_id, 35)
        menu.add.button('Main Menu', set_menu_id, 0)
    return menu
#end function


def tutorial_part4_setup():
    """ Generates a menu for the forth part of the tutorial """
    global menu_screen
    global session_id
    title = "Tutorial Part 4: Efficiency"
    text = ("In Part 4, your team is *easily* herding a flock of sheep.\n\n"
            "There are more than enough dogs in the field. Some are not needed.\n"
            "Removing one or two dogs will help the team to work more *efficiently*.\n"
            "Completing the task with fewer dogs will *increase* your team's performance score.\n\n"
            "Press 'Ok' to Start Part 4. \n")
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu(title, SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(text, max_char=max_char, font_size=title_size)
    menu.add.button('OK', run_swarmsim, 35, TUTORIAL_SEQUENCE_A[3], font_size=button_size)
    # menu.add.button('Start', run_tutorial)
    if DEBUG_MODE_B:
        menu.add.button('Skip (Debug)', set_menu_id, 33)
        menu.add.button('Main Menu (Debug)', set_menu_id, 0)
    return menu
#end function   


def tutorial_complete_setup():
    """ Generates a menu for after the final part of the tutorial has been completed """
    global menu_screen
    title = "Tutorial Complete"
    text = ("Congratulations, you have completed the tutorial!\n\n"
            "Press 'Continue' to start the experiment.\n"
            "Or, you can repeat the Tutorial. \n")
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu(title, SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(text, max_char=max_char, font_size=title_size)

    # ------------- Button points to start the experimental blocks -----------------
    menu.add.button('Continue to the Experiment', run_experiment)
    # menu.add.button('Repeat Parts 3 and 4 of the Tutorial', set_menu_id, 33)
    menu.add.button('Click here to repeat the Tutorial', set_menu_id, 31)
    # menu.add.button('Repeat', set_menu_id, 30)
    return menu
#end function


### Here, the experiment should start. Specific instructions, depending on the block need to be added. Since there is no section dedicated to each block
### i have created two sections in which instructions are displayed for each block. Parcicipants could either press 'Start' to start the experiment
### or press 'Back' to return to the familiarisation trials
### I also added a 'quit' button. What is needed here, is to load the simulation

def experimental_block_1_setup():
    global menu_screen
    # show the third screen
    title = "Attempt 1 (Passive)"
    text = ("You will now be presented with a simulation of the robot team exploring an environment. \n\n"

            "Your task is to observe the robot team and identify whether the team is operating properly, "
            "\nor if some of the robots are faulty or malicious. "
            "\nFaulty robots have flaws that cause them to behave abnormally. \nMalicious robots deliberately try to disrupt the team.\n\n"

            "Remember: the team's task is to explore the whole building as quickly as possible."

            "\n\nPress 'Continue' to start the simulation.\n")
            # "If you wish to repeat the tutorial, press 'Repeat Tutorial'\n")
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu(title, SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(text, max_char=max_char, font_size=title_size)
    menu.add.button('Continue', set_menu_id, 93)  # at the moment, this goes traight to block 2 as I am not able to link it to the simulation.
    # menu.add.button('Repeat Tutorial', run_tutorial)
    return menu
#end function


def experimental_block_2_setup():
    global menu_screen
    # show the third screen
    title = "Attempt 2 (Active)"
    text = ("You will now be presented with a simulation where you have some control over the robot team.\n\n"

            "In these trials, you can direct all the robots to travel, North, or East, or South, or West \n"
            "for a short period by pressing an arrow key on your keyboard."

            "\n\nUsing these controls, you should help the team of robots to complete their task."
            "\nYour task is to observe the swarm and identify any fault or malicious behaviour. "
            "\nFaulty robots have flaws that cause them to behave abormally. \nMalicious robots deliberately try to disrupt the team.\n\n"

            "Remember: the team's task is to explore the whole building as quickly as possible."

            "\nPress 'Continue' to start the simulation.\n")

    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu(title, SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(text, max_char=max_char, font_size=title_size)
    menu.add.button('Continue', set_menu_id, 93)  # at the moment, the code goes straight to the final consent.
    if DEBUG_MODE_B:
        menu.add.button('Quit (Debug)', set_menu_id, -1, menu, True)  # Alternatively, given that this is the second block, participants can decide just to quit the experiment.
    return menu
#end function


#THIS FUNCTION HAS BEEN REMOVED
def save_details(menu):
    """
        Saves information from the final consent details to a google sheet
    """
    global session_id
    print('Settings data:')
    # extract the details from the menu and store in a list "user_data"
    data = menu.get_input_data()
    user_data = []
    for k in data.keys():
        print(f'\t{k}\t=>\t{data[k]}')
        user_data.append(data[k])

    # selectors have a weird format which googlesheets can't work with so need to select the text portion of it
    # user_data[3] = user_data[3][0][0] # I have commented this adn everything seems to work fine on the spreadsheet
    # insert the unique session ID in the first column
    user_data.insert(0, session_id)

    # upload the data to the google sheet
    gc = gspread.service_account(filename=CREDENTIALS_FILE_PATH)
    sh = gc.open(GOOGLE_SHEET_NAME)
    print(sh.sheet1.get('A1'))
    sh.sheet1.append_row(user_data)

    # after storing the data return to the main menu
    set_menu_id(1)
    return
#end function


def end_screen():
    global menu_screen

    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20

    title = "Final Questions"
    text = ("Thank you for taking part in this demo. Press Finish to return to the start.\n")

    menu = pygame_menu.Menu(title, SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(text, max_char=max_char, font_size=title_size)
    
    #menu.add.button('Finish', set_menu_id, 70, menu, True)
    menu.add.button('Finish',  set_menu_id, 40, menu, False)
    return menu
#end function





#THIS FUNCTION HAS BEEN REMOVED
def final_consent_setup():
    title = ("Thank you for Participating in our Experiment!")
    text = ("\nPlease sign the final Consent Form by agreeing with the following:\n \n"
            "I agree to the University of Bristol keeping and processing the data that I have provided during the course of this study.\n"
            "I understand that these data will be used only for the purpose(s) set out in the information sheet, and my consent is conditional "
            "upon the University complying with its duties and obligations under the Data Protection Act.\n\n"
            "If you have any concerns related to your participation in this study please direct them to the Faculty of Engineering Human Research Ethics "
            "Committee, via Liam McKervey, Research Governance and Ethics Officer (Tel: 0117 331 7472 email: Liam.McKervey@bristol.ac.uk ).\n")
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu("Final Consent Form", SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(title, max_char=max_char, font_size=title_size, align=pygame_menu.locals.ALIGN_LEFT)
    menu.add.label(text, max_char=max_char, font_size=text_size)
    # menu.add.button('Agree and Exit', save_details, menu)
    #menu.add.button('Agree and Exit', set_menu_id, 0, menu, True)
    menu.add.button('Agree and Exit', set_menu_id, -1, menu, True)
    
    menu.add.button('Quit', set_menu_id, -1, menu, False)
    return menu
#end function


def test_start_setup(config, control_active):
    global menu_screen
    global test_number
    # show the third screen
    title = "Attempt 1"
    text = ("Press 'Go' when you are ready to start the trial.\n")
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    menu = pygame_menu.Menu(title, SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label(text, max_char=max_char, font_size=28)


    # menu.add.button('Go', run_simulation, 91, '', list_of_configs, show_empowerment, use_taskweighted_empowerment)

    menu.add.button('Go', run_swarmsim, 90, config, [], control_active, False)



    if DEBUG_MODE_B:
        menu.add.button('Skip', set_menu_id, 0)
        menu.add.button('Main Menu', set_menu_id, 0)
    return menu

#end function


def post_test_questions_setup1():
    global menu_screen
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20
    import numpy as np
    SLIDER_VALUES = np.arange(0, 20.5, 0.5).tolist()
    menu = pygame_menu.Menu('Done!', SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    menu.add.label('Please answer the following question...\n', max_char=max_char, font_size=title_size)
    menu.add.text_input('Type (in seconds) how long you think this trial \ntook from start to finish:\n', default='', textinput_id='time', input_underline='_', input_underline_len=5, max_char=max_char, font_size=title_size)



    menu.add.button('Done', set_menu_id, 92, menu, True)
    # menu.add.button('Main Menu', set_menu_id, 0)
    return menu
#end function


def post_test_questions_setup2(config, control_active):
    global menu_screen
    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20

    #Q1_VALUES = {0: 'A is more likely', 1: '', 2: '', 3: 'A and B are equally likely', 4: '', 5: '',6: 'B is more likely'}
    #Q2_VALUES = {0: 'C is more likely', 1: '', 2: '', 3: "C and D are equally likely", 4: '', 5: '',6: 'D is more likely'}

    Q1_VALUES = {0: 'Normal', 1: 'Not sure', 2: 'Abnormal'}
    


    menu = pygame_menu.Menu('Done!', SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    #menu.add.label('Please answer the following two questions...\n', max_char=max_char, font_size=title_size)
    
    menu.add.label('Based on your experience in the last trial, is it more likely that\nA) All the robots were working properly, or that\nB) Some robots were not working properly', max_char=max_char, font_size=title_size, underline=False)
    
    # menu.add.label('From not at all (left) to completely (right)', max_char=max_char, font_size=text_size)
    menu.add.range_slider('', default=2, range_values=list(Q1_VALUES.keys()), increment=1, rangeslider_id='behaviour_perception', width=800, range_line_height=10,
        range_text_value_color=(255, 0, 125), range_text_value_enabled=True, slider_text_value_enabled=False, value_format=lambda x: Q1_VALUES[x])
    
    menu.add.label("\n", max_char=max_char, font_size=3)
    
    # split = config.split('_')[2]
    # if answer_dict[int(split)] == 'faulty':
    #     next_menu = 92
    # else:
    #     next_menu = 91

    # menu.add.vertical_fill()

    menu.add.button('> Click here to see the answer <', set_menu_id, 91, menu, False)
    # menu.add.button('Main Menu', set_menu_id, 0)
    menu.add.label("\n", max_char=max_char, font_size=3)



    return menu
#end function


def post_trial_answer1(config, control_active):
    global menu_screen
    # global last_config_run
    global current_config
    global answer_dict
    global last_config_run
    


    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20

    # print('\n\nThe answer is ', last_config_run)

    # print('answer dict:' , answer_dict)
    # print('current config: ', current_config)

 
    menu = pygame_menu.Menu('Done!', SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    #menu.add.label('Please answer the following two questions...\n', max_char=max_char, font_size=title_size)
    
    menu.add.label('The correct answer was faultyyyyy', max_char=max_char, font_size=title_size, underline=False, label_id='answer_text1')
    menu.add.label('\n', max_char=max_char, font_size=title_size, underline=False)
    menu.add.label('The correct answer was faultyyyyy', max_char=max_char, font_size=title_size, underline=False, label_id='answer_text2')
    menu.add.label('The correct answer was faultyyyyy', max_char=max_char, font_size=title_size, underline=False, label_id='answer_text3')
    
    # menu.add.button('> Continue <', set_menu_id, 50, menu, True)
    # menu.add.button('> Replay Trial? <', set_menu_id, 50, menu, True)
    menu.add.label('\n', max_char=max_char, font_size=title_size, underline=False)
    menu.add.button('> Watch replay <', run_swarmsim, 50, config, [], True, True)

    
    # menu.add.button('Main Menu', set_menu_id, 0)
    menu.add.label("\n", max_char=max_char, font_size=3)

    return menu

def post_trial_answer2(config, control_active):
    global menu_screen
    # global last_config_run
    global current_config
    global answer_dict
    


    SCREEN_W, SCREEN_H = menu_screen.get_size()
    BORDER = 20

    # print('\n\nThe answer is ', last_config_run)

    # print('answer dict:' , answer_dict)
    # print('current config: ', current_config)

 
    menu = pygame_menu.Menu('Done!', SCREEN_W - BORDER, SCREEN_H - BORDER, theme=our_theme)
    #menu.add.label('Please answer the following two questions...\n', max_char=max_char, font_size=title_size)
    
    menu.add.label('The correct answer was faultyyyyy\n', max_char=max_char, font_size=title_size, underline=False, label_id='answer_text1')

    menu.add.label('\n', max_char=max_char, font_size=title_size, underline=False)
    menu.add.label('The correct answer was faultyyyyy', max_char=max_char, font_size=title_size, underline=False, label_id='answer_text2')
    menu.add.label('The correct answer was faultyyyyy', max_char=max_char, font_size=title_size, underline=False, label_id='answer_text3')

    menu.add.label('\n', max_char=max_char, font_size=title_size, underline=False)
    menu.add.button('> Watch replay <', run_swarmsim, 50, config, [], True, True)
    # menu.add.button('> Replay Trial? <', set_menu_id, 50, menu, True)


    
    # menu.add.button('Main Menu', set_menu_id, 0)
    menu.add.label("\n", max_char=max_char, font_size=3)

    return menu




def draw_menu_background():
    # fills the screen with black
    global menu_screen
    menu_screen.fill(colours.BLACK)
    return
#end function


def run_experimental_block(list_of_configs, control_active):
    """"
    Runs a set of experiments. Dynamic switches to the parameter values used in the configurations are processed in here
    
    list_of_configs: the names of the configs to run in this block
    show_empowerment: if true then the dogs show their empowerment value to the user
    use_taskweighted: if true then the dogs use a task weighted measure of empowerment
    """

    global test_number
    global current_menu_id
    global last_config_run
    global answer_dict
    test_number = 0
    is_test_complete_b = False


    config = random.choice(list_of_configs)

    list_of_configs.remove(config)

    # print('This is the chosen config: ', config)


    # Create and setup the menus
    experimental_block_1_setup_m = experimental_block_1_setup()
    experimental_block_2_setup_m = experimental_block_2_setup()
    test_start_setup_m = test_start_setup(config, control_active)
    
    question_window = post_test_questions_setup2(config, control_active)

    post_trial1 = post_trial_answer1(config, control_active)

    post_trial2 = post_trial_answer2(config, control_active)

    fault_text1 = "The team had 6 robots which were experiencing a sensor fault, causing incorrect "
    fault_text2 =  "range measurement. This causes the robots to collide with other robots and with walls. "
    mal_text1 = "The team had 6 malicious robots which purposefully attempt to block doorways in the "
    mal_text2 =  "environment, making it more difficult for healthy robots to explore."
    # Loop until all the experiments given in list_of_configs have been run once
    while test_number <= 1:
        # Blank the screen so old menu's aren't displayed behind new ones
        draw_menu_background()

        # Handle any keyboard or mouse inputs
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exit()

        # A sneaky trick, if the menu is set to the start test screen but we've reached the end of the config list then the block of tests is complete
        #   and the exectution flow should switch back to the master function which is running the experiments.
        #   This only works because after a test is run it will go to the post test questions before returning to the start new test menu
        # if current_menu_id == 90 and test_number == len(list_of_configs):
        #     is_test_complete_b = True
        
        if current_menu_id == 40:
            experimental_block_1_setup_m.update(events)
            experimental_block_1_setup_m.draw(menu_screen)
        
        elif current_menu_id == 50:
            experimental_block_2_setup_m.update(events)
            experimental_block_2_setup_m.draw(menu_screen)

        elif current_menu_id == 93:
            # These lines reset the sliders on the post question menus to their default values.  
            question_window.get_widget('behaviour_perception').reset_value()          
            test_start_setup_m.update(events)
            test_start_setup_m.draw(menu_screen)
        
        elif current_menu_id == 90:
            # These lines reset the sliders on the post question menus to their default values.  
            #   There may be a better way to do this via a call back.
            # post_test_questions_m1.get_widget('time').reset_value()
            # post_test_questions_m2.get_widget('faultOrMal').reset_value()

            question_window.update(events)
            question_window.draw(menu_screen)
        

        elif current_menu_id == 91:
            
            split = config.split('_')[2]

            text = "In the previous trial, the robot team was showing "+answer_dict[int(split)]+" behaviour."
            if answer_dict[int(split)] == 'faulty':
                text2 = fault_text1
                text3 = fault_text2
            else:
                text2 = mal_text1
                text3 = mal_text2
     
            post_trial1.get_widget('answer_text1').set_title(text)
            post_trial1.get_widget('answer_text2').set_title(text2)
            post_trial1.get_widget('answer_text3').set_title(text3)
            post_trial1.update(events)
            post_trial1.draw(menu_screen)

        elif current_menu_id == 92:
            
            split = config.split('_')[2]

            text = "In the previous trial, the robot team was showing "+answer_dict[int(split)]+" behaviour."
            if answer_dict[int(split)] == 'faulty':
                text2 = fault_text1
                text3 = fault_text2
            else:
                text2 = mal_text1
                text3 = mal_text2
     
            post_trial2.get_widget('answer_text1').set_title(text)
            post_trial2.get_widget('answer_text2').set_title(text2)
            post_trial2.get_widget('answer_text3').set_title(text3)
            post_trial2.update(events)
            post_trial2.draw(menu_screen)
       
        else:
            # the menu is outside the list for the experimental block so mark as complete
            is_test_complete_b = True

        # display any changes to the menu
        pygame.display.update()

    # exiting this function should drop back to a master function which is running the experimental blocks
    return
#end function


def run_experiment():
    """
    Runs both the empowerment shown and not shown blocks of trials, 
    and it applies the dynamic configuration parameters to show/not show empowerment and use vanilla/taskweighted empowerment 
    """
    global menu_screen
    global current_menu_id
    import numpy as np
    global control_active

    # Initialise the dynamic parameters to be set later

    control_active = None
    
    # determine the order of the experimental blocks
    num_of_dirs = len(os.listdir(RESULTS_DIR))

    # if num_of_dirs % 2 == 0:
    #     block_order = (['empowerment_shown', 'no_empowerment_shown'])

    # else:
    #     block_order = (['no_empowerment_shown', 'empowerment_shown'])

    block_order = ['passive', 'active']

    
    current_menu_id = 40
    # determine the order to run the sequence
    config_order = np.random.permutation(len(LIVETEST_SEQUENCE_A))
    configs = np.array(LIVETEST_SEQUENCE_A)
    show_empowerment = True
    # Set whether users can control the swarm
   

    # shuffle the test sequence order
    # print(f"I'm running experiment block {block} with the config order {config_order}")
    configs = configs[config_order]

    # Run each block of trials
    for block in block_order:

        if block == 'passive':
            control_active = False
        else:
            control_active = True
            #  for second run set menu to start with active instructions
            current_menu_id = 50
            print('RUNNING THE ACTIVE BLOCK')

        # print('Configs for experiment block')
        run_experimental_block(configs.tolist(), control_active)

    # set the menu ID to the post experiment questions and debreif
    current_menu_id = 60
    # this should return to main()
    return
#end function


def saveAndQuit():
    """
    Saves everything logged from the menu's to file and closes the program
        This should be run in order to exit the program cleanly and not lose data!
    """
    global menu_log
    menu_log.pickleLog(os.path.join(RESULTS_DIR, session_id, ""))

    menu_log.save_toexcel()
    exit()
    return
#end function


def sim_animate(i, timesteps, control_active, agent_pos, max_length, trails, malicious_trails, faulty_pos, sim_speed, totSwarm_size, agents_withFaults, display_abnormal):

    # Check gird intersection
    #grid_check(swarmy)
    global score
    global swarmy
    global malicious_swarm
    global agent_set
    global SimRecorder
    # global malicious_trails
    # global unhappy_agents
    global agent_trails
    global maliciousAgent_trails
    global coverage_data
    start = time.time()


    # Automatically close the simulation 
    if i == timesteps - 1:
        plt.close()


    swarmy.time = i

    # ---------------------------- Spawn agents from edge of environment over time -------------------------------------
    # agent_set = np.arange(0, totSwarm_size, 1)
    total_agentSize = swarmy.size + malicious_blockers
    # input()
    pos_variance = 4
    if i%15 == 0 and len(agent_set) >= 1:

        # At each step pick a random agent to spawn
        pick = np.random.choice(agent_set)
        agent_set = np.delete(agent_set, np.where(agent_set == pick)[0])
        # print('The agent set: ', agent_set)
        if pick < malicious_blockers:
            # spawn a malicious agent
            malicious_swarm.agents[pick] = np.array([swarmy.map.swarm_origin[0], swarmy.map.swarm_origin[1]])
        else:
            swarmy.agents[pick-malicious_swarm.size] = np.array([swarmy.map.swarm_origin[0], swarmy.map.swarm_origin[1] + np.random.uniform(-pos_variance, pos_variance)])
            swarmy.previous_state[pick-malicious_swarm.size] = np.array([swarmy.map.swarm_origin[0], swarmy.map.swarm_origin[1] + np.random.uniform(-pos_variance ,pos_variance)])
            swarmy.opinion_timelimit[pick-malicious_swarm.size] = 50
            swarmy.behaviour[pick-malicious_swarm.size] = 4
    if i <= 150:
        swarmy.behaviour = 4*np.ones(swarmy.size)
        swarmy.param = 10
    else:
        swarmy.param = 60


    # ------------------------------------------------------------------------------------------------------------------

    swarmy.time = i

    # swarmy.happiness = np.random.normal(0.9,.01, swarmy.size)
    if i >= 50:
        faulty_swarm.collision_check(swarmy, malicious_swarm)
    swarmy.iterate(malicious_swarm, noise[i-1])

   
    if blockers_active == True:
        malicious_swarm.iterate(malicious_noise[i-1])
        malicious_swarm.get_state()


    swarmy.get_state_opinion()
    swarmy.get_state()
    score += targets.get_state(swarmy, i, timesteps)


    time_data.append(i)
    coverage_data.append(targets.coverage)

    agents = np.concatenate((swarmy.agents, malicious_swarm.agents), axis = 0)

    x = agents.T[0]
    y = agents.T[1]

    agent_pos.set_data(x,y)

    agent_trails = np.concatenate((agent_trails, swarmy.agents), axis = 0)

    if len(agent_trails) > max_length:

        agent_trails = agent_trails[swarmy.size:]

    trails.set_data(agent_trails.T[0], agent_trails.T[1])


    maliciousAgent_trails = np.concatenate((maliciousAgent_trails, malicious_swarm.agents), axis = 0)

    if len(maliciousAgent_trails) > 10*malicious_swarm.size:

        maliciousAgent_trails = maliciousAgent_trails[malicious_swarm.size:]

    malicious_trails.set_data(maliciousAgent_trails.T[0], maliciousAgent_trails.T[1])



    taken = 1000*(time.time() - start)
    sim_speed.append(taken)

    # ---------------- Data Logging --------------------

    # Pass in positions of healthy swarm and malicious agents (blockers)

    SimRecorder.record_Step(agents)

   
    faulty_indicies = np.where(agents_withFaults == 1)[0]
    # Highlight agents which are faulty
    if display_abnormal == True and len(faulty_indicies) != 0:

        positions = swarmy.agents[agents_withFaults == 1]
        # print('fault position data: ', positions)
        faulty_pos.set_data(positions.T[0] + 1, positions.T[1] + 1)
    else:
        faulty_pos.set_data([],[])
    
    # if (i == timesteps - 1):
    #     plt.close(fig)
 
    return (trails, malicious_trails, agent_pos, faulty_pos, )




def run_swarmsim(exit_to_menu, config_file_name='', list_of_configs=[], control_active=False, display_abnormal=False):
    """
    Run's a single trial of the empowerment simulation

    """
    global test_number
    global session_id
    global menu_screen
    global current_menu_id
    global last_config_run


    # Menu callbacks aren't dynamic so need to do some tricks if the config file name isn't supplied but a list is
    # if config_file_name == '' and not list_of_configs:
    #     raise Exception("list of configs can't be blank if no config file name provided!!")
    # elif config_file_name == '':
    #     config_file_name = list_of_configs[test_number]
    # elif config_file_name == '' and display_abnormal == True:
    #     config_file_name = list_of_configs[test_number-1]

    # config_file_name = random.choice(list_of_configs)

   
    if "_fam_2" in config_file_name:
         control_active = True
    

    # create a name for the config which includes information on the dynamic parameters
    #   i.e. those which are set at runtime rather than written in the config file
    #   These are useful for naming log files!
    config_name_with_parameters = config_file_name
    # if show_empowerment:
    #     config_name_with_parameters = config_name_with_parameters + "_empshown"

    # if use_task_weighted_empowerment:
    #     config_name_with_parameters = config_name_with_parameters + "_taskweighted"

    if control_active == True:
        config_name_with_parameters = config_file_name + "_active"
    else:
        config_name_with_parameters = config_file_name + "_passive"


    # print(f'running a simulation with config {config_file_name} and #test {test_number}, exiting to menu {exit_to_menu}')
    # if not DEBUG_MODE_B:
    #     sim.main(config_file_name, show_empowerment, use_task_weighted_empowerment, sim_session_id=session_id, log_file_name=config_name_with_parameters)


    # ======================= Very important! needs to be set in order to save the config name for results =====================
    last_config_run = config_name_with_parameters

    print('This is the last config: ', last_config_run)

    # Load data from config file ----------------------------

    config = importlib.import_module(config_file_name)
    cfg = config.main()
    # print(cfg)


    fig, ax1 = plt.subplots( figsize=(11,11), dpi=100, facecolor='w', edgecolor='k')
   
    ax1.set_aspect('equal')
    
    global line, line1
    # Agent plotting 
    robot_size = 10

    random_pos = [(0,0),(10,10),(20,34)]

   
    # ---------- ----------------- Setup for data plotting --------------------------------------

    agent_pos, = ax1.plot([], [], 'rh', markersize = 8, markeredgecolor="black", alpha = 1, zorder=10)

    faulty_pos, = ax1.plot([], [], 'r*', markersize = 8, alpha = 1, zorder=10)

    trails, = ax1.plot([], [], 'bh', markersize = 6, alpha = 0.2)
    malicious_trails, = ax1.plot([], [], 'bh', markersize = 6, alpha = 0.2)


    fsize = 12

    # Always keep seed the same for the purpose of replays
    seed = 99999

    random.seed(seed)
    np.random.seed(seed)

    # print('\nChosen seed: ', seed)

    #          Creat environment object
    env_map = asim.map()
    env_map.map1_simplified()
    env_map.swarm_origin = np.array([44,15])
    env_map.gen()


    ax1.set_ylim([-45,45])
    ax1.set_xlim([-45,45])

    # plt.legend(loc="upper left")

    # ----------- Plot walls except the opening entry point ---------
    for a in range(len(env_map.obsticles)):

        if a != 0:
            ax1.plot([env_map.obsticles[a].start[0], env_map.obsticles[a].end[0]], [env_map.obsticles[a].start[1], env_map.obsticles[a].end[1]], '-', color = 'black', lw=3, markeredgecolor = 'black', markeredgewidth = 3) 

    # global timesteps

    timesteps = int(cfg["timesteps"]/10)


    # ===================== Swarm Faults/Malicious behaviours =========================

    totSwarm_size = cfg["swarm_size"]
    # positive sensor error added to distance measurement between agents
    num_sensorfault = cfg["faulty_num"]
    # Channels of communication between certain robots is completely lost
    num_robotblind = 0
    # Motor error causing agents to move half speed with a degree of fluctuation
    num_motorslow = cfg["faulty_motor"]
    # agents have a persistent heading error 
    num_headingerror = 0

    # Malicious behaviours
    global malicious_blockers
    malicious_blockers = cfg["mal_blockers_num"]
    '''
    Malicious broadcasting agents always communicate that they have maximum happiness.
    Agents which have significantly lower happiness will copy broadcasting agents.
    Broadcasters do not attempt to find good behaviours, creating sinks where clusters of 
    agents form copying the behaviour of the malicious agent.
    '''
    num_maliciousBroadcast = cfg["mal_comms_num"]

    global blockers_active
    if malicious_blockers <= 1:
        blockers_active = False
    else:
        blockers_active = True



    # Set target positions
    global targets
    

    global score
    global coverage_data
    global time_data
    global happy_data
    score = 0

    global malicious_noise
    global noise

    # Declare agent motion noise
    # noise = np.random.uniform(-.1,.1,(timesteps, base_swarm.size, 2))
    coverage_data = list()
    time_data = list()

    global agent_set
    global swarmy
    global malicious_swarm
    

    swarmy = faulty_swarm.swarm()
    swarmy.size = totSwarm_size - malicious_blockers
    swarmy.speed = 0.2
    swarmy.origin = env_map.swarm_origin[:]
    swarmy.map = env_map
    swarmy.gen_agents()

    # Set agent to agent repulsion
    swarmy.param = cfg["repulsion_strength"]



    if blockers_active == True:
        malicious_swarm = faulty_swarm.malicious_swarm()
        malicious_swarm.size = malicious_blockers
        malicious_swarm.speed = 0.2
        malicious_swarm.origin = env_map.swarm_origin[:]
        malicious_swarm.map = env_map
        malicious_swarm.gen_agents()
        malicious_swarm.behaviour = 10*np.ones(malicious_swarm.size)
    else:
        malicious_swarm = faulty_swarm.malicious_swarm()
        malicious_swarm.size = 10
        malicious_swarm.speed = 0.2
        malicious_swarm.origin = env_map.swarm_origin[:]
        malicious_swarm.map = env_map
        malicious_swarm.gen_agents()
        # Initilaize swarm positions outside environment
        malicious_swarm.agents = 1000*np.ones((10,2))


    # Generate potential field map of environment
    field, grid = asim.potentialField_map(swarmy.map)
    swarmy.field = field
    swarmy.grid = grid

    targets = asim.target_set()
    targets.radius = 2.5
    targets.set_state('4x4')
    targets.reset()   
    
    coverage_data = []
    time_data = []
    happy_data = []

    # ===================== Setting fault intermittance ======================

    swarmy.fault_rate = 100
    '''
       Fault intermittance sets the proportion of time that
       the fault is active. i.e 0 means the fault is never active 
       and 1 would mean the fault is always active.

       The fault rate defines the period over which the fault can
       switch between active and inactive.
    '''
    swarmy.fault_intermittance = 0.5
    swarmy.fault_limit = np.random.randint(0, swarmy.fault_rate, swarmy.size)



    # Set the length of agent trails in simulation
    max_length = 10*swarmy.size
    #max_length = 200000000000000*swarmy.size
    global agent_trails
    global maliciousAgent_trails
    agent_trails = 1000*np.ones((swarmy.size, 2))
    maliciousAgent_trails = 1000*np.ones((swarmy.size, 2))


    # Generate swarm motion noise for entire simulation

    noise = np.random.uniform(-.1,.1,(timesteps, swarmy.size, 2))
    malicious_noise = np.random.uniform(-.1,.1,(timesteps, malicious_swarm.size, 2))

    score = 0 

    # ***** Initially randomise the starting behaviour of agents
    swarmy.behaviour = np.random.randint(1, 9, swarmy.size)
    swarmy.behaviour = 1*np.ones(swarmy.size)
    # swarmy.param = 2


    #====================== Assign robots which will have faults =================================

    # Create list of all robots for random selection 
    agent_set = np.arange(0, totSwarm_size, 1)


    #  ----------  Agents with sensor error fault ----------- 


    swarmy.sensor_mean = 10
    swarmy.sensor_dev = 2

    malicious_swarm.sensor_mean = 10
    malicious_swarm.sensor_dev = 2

    agent_set = np.arange(0, swarmy.size, 1)

    for n in range(0, num_sensorfault):
        pick = np.random.choice(agent_set)
        agent_set = np.delete(agent_set, np.where(agent_set == pick)[0])
        # Chosen agent has error added to sensors
        swarmy.sensor_fault[pick] = 1


    for n in range(0, num_robotblind):

        pick = np.random.randint(0, swarmy.size - 1)

        swarmy.sensor_fault[pick] = 1


    # # -------------- Agents with slow motors ---------------


    # # Define default motor speeds for agents
    swarmy.motor_error = np.zeros(swarmy.size)
    swarmy.motor_speeds = np.ones(swarmy.size)

    malicious_swarm.motor_error = np.zeros(malicious_swarm.size)
    malicious_swarm.motor_speeds = np.ones(malicious_swarm.size)
    swarmy.motor_mean = 0.5
    swarmy.motor_dev = 0.3
    for n in range(0, num_motorslow):

        
        pick = np.random.choice(agent_set)
        agent_set = np.delete(agent_set, np.where(agent_set == pick)[0])
        # Chosen agent has error added to sensors
        
        swarmy.motor_error[pick] = 1

    # ------------- Agents with heading error --------------


    swarmy.heading_error = np.zeros(swarmy.size)
    malicious_swarm.heading_error = np.zeros(malicious_swarm.size)
    # swarmy.motor_speeds = np.ones(swarmy.size)
    swarmy.heading_mean = 1.2
    swarmy.heading_dev = 0.2
    for n in range(0, num_headingerror):

        pick = np.random.choice(agent_set)
        agent_set = np.delete(agent_set, np.where(agent_set == pick)[0])
        # Chosen agent has error added to sensors
        
        swarmy.heading_error[pick] = 1

    # ------------- Malicious broadcasting agents ----------------------------

    swarmy.malicious_broadcasters = np.zeros(swarmy.size)

    for n in range(0, num_maliciousBroadcast):

        pick = np.random.choice(agent_set)  
        agent_set = np.delete(agent_set, np.where(agent_set == pick)[0])
        # pick = np.random.randint(0, swarmy.size - 1)
        # Chosen agent performs the malicious swarm behaviour
        swarmy.malicious_broadcasters[pick] = 1

    # happiness_plot = input("Plot happiness?")
    # if happiness_plot == 'yes' or happiness_plot == 'y':
    #   print('plotting happiness')
    #   happiness_plot = True

    spawned_state = np.zeros((swarmy.size + malicious_swarm.size))  

    # Agents are initially not in view
    swarmy.agents = 1000*np.ones((swarmy.size, 2))
    malicious_swarm.agents = 1000*np.ones((malicious_swarm.size, 2))
    agent_set = np.arange(0, swarmy.size + malicious_blockers, 1)

    global plot_happiness
    global plot_faulty

    plot_happiness = True
    plot_faulty = False

    time_data = list()
    coverage_data = list()

    if control_active == True:
        fig.canvas.mpl_connect('key_press_event', on_press)
    

    agents_withFaults = np.logical_or(swarmy.malicious_broadcasters, swarmy.heading_error)

    agents_withFaults = np.logical_or(agents_withFaults, swarmy.motor_error)

    agents_withFaults = np.logical_or(agents_withFaults, swarmy.sensor_fault)

    faulty_indicies = np.where(agents_withFaults == 1)[0]
    # print(faulty_indicies)

    sim_speed = list()

    # Creat simulation data logger
    global SimRecorder
    # import model.SimLog
    SimRecorder = SimLog.data_recorder()

    SimRecorder.initialise(session_id, config_file_name, seed, control_active)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    
    anim = animation.FuncAnimation(fig, sim_animate, frames=timesteps, interval=15, blit=True, repeat = False,
                                   fargs = (timesteps, control_active, agent_pos, max_length,
                                   trails, malicious_trails, faulty_pos, sim_speed, totSwarm_size, agents_withFaults, display_abnormal))

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()


    plt.show()
    # anim

    # Save the coverage achieved 
    SimRecorder.record_coverage(coverage_data)

    global RESULTS_DIR

    if control_active == True:
        file_name = RESULTS_DIR + session_id +'/'+ config_file_name + '_Active'
    else:
        file_name = RESULTS_DIR + session_id +'/'+ config_file_name + '_Passive'

    # Save simulation data to file
    SimRecorder.pickleLog(file_name)

    # Not sure if this is always needed but to be on the safe side, recreate the menu window after the simulation window closes.
    menu_screen = create_start_screen()

    # set the menu we go to after the simulation is complete based on the parameter passed when the simulation was initiated
    current_menu_id = exit_to_menu
    
    # increment the number of tests completed
    test_number += 1
    return



def on_press(event):
    # print('press', event.key)
    # sys.stdout.flush()
    global SimRecorder
    
    # Handle keyboard input for interactive trials
    if event.key == 'up':
        swarmy.behaviour = 1*np.ones(swarmy.size)
    if event.key == 'left':
        swarmy.behaviour = 4*np.ones(swarmy.size)
    if event.key == 'down':
        swarmy.behaviour = 2*np.ones(swarmy.size)
    if event.key == 'right':
        swarmy.behaviour = 3*np.ones(swarmy.size)
    swarmy.opinion_timer = 1*np.ones(swarmy.size)
    swarmy.opinion_timelimit = 50*np.ones(swarmy.size)

    # print('The time is ', swarmy.time)

    SimRecorder.user_log.record_event(event.key, swarmy.time)
    # print('User input log: ', SimRecorder.user_log.events)

    # ---------- Record command event
    # sys.stdout.flush()

    global plot_happiness
    global plot_faulty

    if event.key == 'h':
        # turn on/off agent happiness display
        plot_happiness = np.logical_not(plot_happiness)
    if event.key == 'j':
        # turn on/off agent happiness display
        plot_faulty = np.logical_not(plot_faulty)


def run_tutorial():
    """
    Runs the block of trials used in the tutorial
    """
    global menu_screen
    global current_menu_id

    # print("I'm running the tutorial")

    # Create an initialise the menus
    tutorial_start_m = tutorial_start_menu_setup()
    tutorial_part1_m = tutorial_part1_setup()
    tutorial_part2_m = tutorial_part2_setup()
    tutorial_part3_m = tutorial_part3_setup()
    tutorial_part4_m = tutorial_part4_setup()
    tutorial_complete_m = tutorial_complete_setup()

    # Make sure the starting menu screen is tutorial start
    current_menu_id = 30
    running_tutorial_b = True

    while running_tutorial_b:
        # blank the screen so old menu's aren't displayed behind new ones
        draw_menu_background()

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                saveAndQuit()

        # select the right menu for the state, update it (check if buttons have been pushed etc) and draw.
        if current_menu_id == 30:
            tutorial_start_m.update(events)
            tutorial_start_m.draw(menu_screen)
        elif current_menu_id == 31:
            tutorial_part1_m.update(events)
            tutorial_part1_m.draw(menu_screen)
        elif current_menu_id == 32:
            tutorial_part2_m.update(events)
            tutorial_part2_m.draw(menu_screen)
        elif current_menu_id == 33:
            tutorial_part3_m.update(events)
            tutorial_part3_m.draw(menu_screen)
        elif current_menu_id == 34:
            tutorial_part4_m.update(events)
            tutorial_part4_m.draw(menu_screen)
        elif current_menu_id == 35:
            tutorial_complete_m.update(events)
            tutorial_complete_m.draw(menu_screen)
        else:
            running_tutorial_b = False

        pygame.display.update()
    return
#end function


def main():
    """
    Runs the top level menu structure which contains the main menu, briefing and debriefing screens.
    The program uses two sub menu structures which aren't handled in this function. These are:
     - run_tutorial: runs the menus used in the tutorial
     - run_experment: runs the menus used in the experment
    """
    global menu_screen
    global current_menu_id
    global menu_log
    global session_id

    # Create a unique session ID
    generate_session_id()

    # Cetup the logging for user responses entered on the menus
    # menu_log = MenuLog.MenuLog(session_id)
    menu_log = dataLogger.MenuLog(session_id)

    # Create and setup the pygame window/screen
    menu_screen = create_start_screen();  

    # Start pygame
    pygame.init()

    # Create and setup the menus
    start_m = start_menu_setup()
    # information_1_m = information_sheet1_setup()
    # information_2_m = information_sheet2_setup()
    # information_3_m = information_sheet3_setup()
    # information_4_m = information_sheet4_setup()
    # information_5_m = information_sheet5_setup()
    instructions_m1 = instructions_menu_setup1()
    instructions_m2 = instructions_menu_setup2()
    # details_m = details_setup()

    final_screen = end_screen()
    # debrief_setup_m = debrief_setup()
    # personal_details_m = personal_details_setup()
    # final_consent_m = final_consent_setup()

    # Loop until the program exits
    while True:
        # Blank the screen so old menu's aren't displayed behind new ones
        draw_menu_background()

        # Handle the event queue
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                # menu_log.save_responses(details_m,60)
                saveAndQuit()

        # Select the right menu for the program state, update it (check if buttons have been pushed etc) and draw.
        if current_menu_id == 0:
            start_m.update(events)
            start_m.draw(menu_screen)
        elif current_menu_id == 10:
            # information_1_m.update(events)
            # information_1_m.draw(menu_screen)
            pass
        elif current_menu_id == 11:
            # information_2_m.update(events)
            # information_2_m.draw(menu_screen)
            pass
        elif current_menu_id == 12:
            # information_3_m.update(events)
            # information_3_m.draw(menu_screen)
            pass
        elif current_menu_id == 13:
            # information_4_m.update(events)
            # information_4_m.draw(menu_screen)
            pass
        elif current_menu_id == 14:
            # information_5_m.update(events)
            # information_5_m.draw(menu_screen)
            pass
        elif current_menu_id == 20:
            instructions_m1.update(events)
            instructions_m1.draw(menu_screen)
        elif current_menu_id == 21:
            instructions_m2.update(events)
            instructions_m2.draw(menu_screen)
        # menu_id 30-35 are handled in run_tutorial()
        # menu_id 40,50,90-92 are handled in run_experiment()
        elif current_menu_id == 60:
            final_screen.update(events)
            final_screen.draw(menu_screen)
        elif current_menu_id == 70:
            # debrief_setup_m.update(events)
            # debrief_setup_m.draw(menu_screen)
            pass
        elif current_menu_id == 80:
            # final_consent_m.update(events)
            # final_consent_m.draw(menu_screen)
            pass
        else:
            start_m.update(events)
            start_m.draw(menu_screen)

        pygame.display.update()
    #end while loop
    return
#end function


if __name__ == '__main__':
    main()
