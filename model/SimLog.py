import numpy as np


class data_recorder:

    def __init__(self) -> None:
        #logging lists
        
        self.user_log = UserInputs()
        self.user_log.initialise()
        #self.config = ''
        #self.session_id = ''
        self.meta = {}

        self.robot_pos = None
        self.faulty_id = None
        self.mal_id = None
        self.happiness = None
        self.coverage = None

    def initialise(self, session_id, config_name, seed, control_active):
        
        self.robot_pos = list()
        self.faulty_id = None
        self.mal_id = None
        self.happiness = None
        self.coverage = None

        self.user_log = UserInputs()
        self.user_log.initialise()
        #self.config = config_name
        #self.session_id = session_id
        self.meta = {   'config_name' : config_name, 
                        'session_id' : session_id,
                        'sim_seed' : seed,
                        'start_time' : None, 
                        'end_time' : None, 
                        'control_active' : control_active
                        }

    def record_Step(self, agents):

        # Save robot positions

        self.robot_pos.append(agents)

    def record_coverage(self, coverage):

        # Save coverage data for whole trial
        self.coverage = coverage



    def save_agentId(self):

        pass

    def recordStartTime(self, time):
        self.meta['start_time'] = time
        return


    def recordEndTime(self, time):
        self.meta['end_time'] = time  
        return     


    def pickleLog(self, file_name):
        import pickle
        data = {
            'robot_pos' : self.robot_pos,
            'faulty_id' : self.faulty_id,
            'mal_id' : self.mal_id,
            'happiness' : self.happiness,
            'coverage' : self.coverage,
            'user_log' : self.user_log.events,
            'meta_data' : self.meta
        }
        fileo = open(f'{file_name}.pkl', 'wb')
        pickle.dump(data, fileo)
        fileo.close()
        return



class UserInputs:
    def __init__(self) -> None:
        self.input_at_t = {}
        self.events_at_t = {}
        self.events = list()
        return

    def initialise(self):
        self.input_at_t = {}
        self.events_at_t = {}
        self.events = list()

    def record_event(self, command, time):

        # Save command event as command given and time in sim
        self.events.append((command, time))
