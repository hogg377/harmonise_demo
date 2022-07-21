import numpy as np

class AgentState:
    '''
    Records the key information about an agents state history
    The update(...) method should be called once for every tick the agent is "alive"
    id: the unique identifier for the agent
    time_created:   the simulaltion tick whent he agent was created
    time_destroyed: the simulaltion tick whent he agent was removed,
                    -1 (default) means it was alive when the simulation ended
    positions:      2d numpy array where the ith "row" is the xi,yi position of the agent at tick i since it was created
    tick:           not used.  Idea was to use it as an internal tracker to playback the agents positions (since the may be different to world time)
    '''
    def __init__(self, id) -> None:
        self.id = id
        self.state = {}
        #this will be used to "play back" the state
        #   it's essentially local time for the agent from when it was created
        #   and so potentially different from simulation time.
        self.tick = 0
        return


    def initialise(self, time, init_position):
        #going to append stuff during the update so useful to set something on the first row we can ignore later
        #init_time = np.empty(1)
        #init_time[:] = np.NaN
        #init_position = np.empty((1,2))
        #init_position[:] = np.NaN
        self.state = {
            'id' : self.id,
            'time_created' : time,
            #'times' : init_time,
            'time_destroyed': -1,
            'positions' : np.atleast_2d(init_position),
            'time' : np.array([time]),
            'empowerment' : np.array([-1])
        }
        self.tick = 0
        return


    def destroy(self, time):
        self.state['time_destroyed'] = time
        return


    def update(self, time, position, empowerment=-1):
        #self.agent_state['times'] = np.append(self.agent_state['times'], time)
        self.state['positions'] = np.append(self.state['positions'], np.atleast_2d(position), axis=0)
        self.state['time'] = np.append(self.state['time'], [time], axis=0)
        self.state['empowerment'] = np.append(self.state['empowerment'], [empowerment], axis=0)
        self.tick+=1
        return

class UserInputs:
    def __init__(self) -> None:
        self.input_at_t = {}
        self.events_at_t = {}
        return

    def initialise(self):
        self.input_at_t = {}
        self.events_at_t = {}


    def mouseclick_at_t_template(self):
        foo = {
            'type' : [],
            'screen_position' : [],
            'grid_position' : [],
            'realtime' : []
        }
        return foo


    def agentevent_at_t_template(self):
        foo = {
            'id' : [],
            'event': [],
            'grid_position' : []
        }
        return foo


    def addMouseClick(self, time, realtime, mouse_button, screen_pos, grid_pos):
        if time not in self.input_at_t.keys():
            self.input_at_t[time] = self.mouseclick_at_t_template()

        self.input_at_t[time]['type'].append(mouse_button)
        self.input_at_t[time]['screen_position'].append(screen_pos)
        self.input_at_t[time]['grid_position'].append(grid_pos)
        self.input_at_t[time]['realtime'].append(realtime)
        return


    def addAgentEvent(self, time, agent_id, grid_pos, event_type):
        if time not in self.events_at_t.keys():
            self.events_at_t[time] = self.agentevent_at_t_template()

        if event_type == 'destroy':
            agent_id *=-1
        self.events_at_t[time]['id'].append(agent_id)
        self.events_at_t[time]['grid_position'].append(grid_pos)
        self.events_at_t[time]['event'].append(event_type)
        return
    #end function



class Logger:
    def __init__(self) -> None:
        #logging lists
        self.dog_logs = {}
        self.sheep_logs = {}
        self.world_at_t = {}
        self.user_log = UserInputs()
        self.user_log.initialise()
        #self.config = ''
        #self.session_id = ''
        self.meta = {}


    def initialise(self, session_id, config_name, show_empowerment_b, task_weighted_empowerment_b):
        self.dog_logs = {}
        self.sheep_logs = {}
        self.world_at_t = {}
        self.user_log = UserInputs()
        self.user_log.initialise()
        #self.config = config_name
        #self.session_id = session_id
        self.meta = {   'config_name' : config_name, 
                        'session_id' : session_id, 
                        'start_time' : None, 
                        'end_time' : None, 
                        'taskweighted_empowerment' : task_weighted_empowerment_b,
                        'empowerment_shown' : show_empowerment_b
                        }


    def recordStartTime(self, time):
        self.meta['start_time'] = time
        return


    def recordEndTime(self, time):
        self.meta['end_time'] = time  
        return     


    def getList(self, type):
        if type=='dog':
            m_list = self.dog_logs
        elif type=='sheep':
            m_list = self.sheep_logs
        return m_list


    def addNewAgentInLog(self, type, id, position, time):
        #create the agent in the log
        m_list = self.getList(type)
        m_list[id] = AgentState(id)
        #initialise the agent
        m_list[id].initialise(time, position)
        #record when the agent was created
        self.user_log.addAgentEvent(time,id,position,'add')
        return


    def destroyAgentInLog(self, type, id, position, time):
        m_list = self.getList(type)
        m_list[id].destroy(time)

        self.user_log.addAgentEvent(time, id, position, 'destroy')
        return


    def logPopulationStates(self, type, population, time):
        m_list = self.getList(type)

        for agent in population.m_agents:
            #if the agent is a dog then it will have an empowerment value else set the default to -1 (sheep don't calculate empowerment)
            #TODO add empowerment to the sheep class (or agent class!) and set the default to -1 then all agents would have the agent.m_empowerment parameter but only dogs update it
            if type =='dog':
                empowerment_value = agent.m_empowerment
            else:
                empowerment_value = -1

            m_list[agent.m_id].update(time, agent.m_position, empowerment_value)
        return


    def logPopulations(self, populations, time):
        #size a numpy array to hold the agent ids and agent times
        n_agents = 0
        for pop in populations:
            n_agents+=len(pop.m_agents)

        #iniitalise the numpy arrays to hold the world state
        ids_at_t = np.zeros(n_agents)
        positions_at_t = np.zeros((n_agents,2))

        #itterate through each agent in each population storing their id and position in the master lists
        idx = 0
        for pop in populations:
            for agent in pop.m_agents:
                ids_at_t[idx] = agent.m_id
                positions_at_t[idx,:] = agent.m_position
                idx+=1

        #store the results
        self.world_at_t[time] = {'ids' : ids_at_t, 'positions' : positions_at_t}
        return


    def pickleLog(self, file_name):
        import pickle
        data = {
            'dog_logs' : self.dog_logs,
            'sheep_logs' : self.sheep_logs,
            'world_at_t' : self.world_at_t,
            'user_log' : self.user_log,
            'meta_data' : self.meta
        }
        fileo = open(f'{file_name}.pkl', 'wb')
        pickle.dump(data, fileo)
        fileo.close()
        return


    #nice idea but think it's easier to just store everything in the world_at_t structure
    # def convertPositionsToNumpyTs(self, type, time_range):
    #     '''
    #     Returns:
    #         ts :  a time series of 2d grid positions indexed as (time, i_agent, [x,y])
    #     '''
    #     import numpy as np
    #     n_time_points = time_range[1] - time_range[0] + 1
    #     m_list = self.getList(type)

    #     #initialise the time series to account for all the agents then set everything to nan
    #     #   this is because different dogs will be created and destroyed as different times
    #     ts = np.zeros((n_time_points, len(m_list),2))
    #     ts[:] = np.NaN

    #     #loop through the agents
    #     for i in range(0, ts.shape[1]):
    #         for id, in sorted(self.dog_logs):
    #             t_start = self.dog_logs[id].state['time_created']
    #             positions = self.dog_logs[id].state['positions']
    #             t_end = t_start + positions.shape[0]
    #             ts[t_start:t_end,i] = positions
    #     return ts



