from pickle import FALSE, TRUE
import numpy as np
import hashlib, json, logging
import os
import scipy.io

def loadConfig(fname = 'cfg.npy'):
    if ~os.path.isfile(fname):
        saveConfig(fname)
    cfg = np.load(fname, allow_pickle='TRUE').item()
    logging.info("Config file {} loaded.".format(fname))
    logging.info("Config hash: {}.".format(hash(cfg)))
    return cfg


def saveConfig(fname = 'cfg.npy'):
    matlab_data = scipy.io.loadmat('homogeneous_flock_genome.mat')
    # set the world size to an integer (between 5 and 80 works ok on a HD screen)
    world_sz = 60
    cfg = {"world" : {
        "width" : world_sz,
        "height" : world_sz,
        #square size sets the size of the dogs and sheep.  Total window size is square_size * world_sz
        "square_size" : 10,
        #goal square dimensions in grid squares.  An integer between 1 and world_sz.  Note the goal rectangle will be centered on ["dog"]["endgoal_position"]
        "goal_width" : 20,
        "goal_height" :20
        },
        "flock" : {
            #the total number of sheep
            "flock_size" : 15
            },
        "sheep" : {
            #the starting positions of the sheep, must be list of 2 element tuples in the range [2..(world_sz - 2)].
            "initial_positions" : [(50, 42),(13, 27), (23, 33)], #(24, 21), (35, 49), (50, 21), (15, 47), (16, 27), (17, 27), (37, 21), (36, 31), (11, 32), (27, 45), (50, 34), (44, 45)],
            #"initial_positions" : [(5,47+5), (25,65), (36,41+5), (60,44+5), (51,57), (8,60), (47,70), (21,43+5), (5,60), (72,54)],
            #"initial_positions" : [(60,60), (20,25), (65,25)],
            #a list of 10s where the number of elements = flock size
            "types" : [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],
            #how far the sheep can "see", default is 40
            "sensor_range" : 40,
            #the probability a sheep takes a random step if not influenced by a dog or other sheep
            "graze_probability" : 0.05,
            #how the different sheep types respond to dog types and other sheep.  Leave these alone for now
            "dog_repsonse_map" : {"1" : 0, "2" : 0},
            "sheep_response_map" : {"10" : 0, "20" : 0},
            #switch to turn on/off the behaviour which biases a sheeps movement towards other sheep when being influenced by a dog
            "use_flocking_behaviour" : True
        },
        "dog" : {
            #the starting positions of the dogs, must be list of 2 element tuples in the range [2..(world_sz - 2)].
            #"initial_positions" : [(10,10),(30,50), (50,30)],
            "initial_positions" : [(10,10)],
            #"initial_positions" : [(10,10+20),(30,10+20), (50, 10+20)],
            #"initial_positions" : [(72,60),(15,40)],
            #a list of 1s where the number of elements = number of dogs
            "types" : [1,1,1],
            #the target location for the dogs to herd the sheep to
            "endgoal_position" : [40,40],
            #how far the dogs can "see", default is 600 (god view)
            "sensor_range" : 600,
            #the method used to calculate empowerment.  Valid options are: 'transition', 'leaf', 'leaf_unique', 'node_unique'
            "empowerment_method" : 'transition',
            #the number of moves to project into the future when calculating empowerment.  Use an integer.  3+ is very slow(!), 2 is default.
            "empowerment_horizon" : 2,
            #parameters for the dog controller.  Leave these alone for now
            "nn" : {
                "input_layer_size" : 4,
                "output_layer_size" : 4,
                "hidden_layer_size" : 5,
                "input_scaling" : [1/world_sz, 1/world_sz, 1, 1]
            },
            #where the dogs should move to when the task is complete (not sure if this is used??)
            "home_position" : [2,4],
            #used for splitting the genome between targetting and moveing.  Leave this set to 4
            "n_valueweights" : 4,
            #how many consecutive ticks a dog will allow itself to be "stuck" but still try to follow its policy. Default 5
            "stationary_tick_limit" : 5,
            #loads the genome for the dog controller
            "genome" : np.squeeze(matlab_data['genome']),
            #set the maximum and minimum number of allowed dogs
            "n_dogs_max" : 10,
            "n_dogs_min" : 1,
            #set True to calculate and show empowerment
            "calculate_empowerment_b" : True,
            #set True to change the dog's colour in response to its empowerment value
            "show_empowerment_b": True,
            #set True to show numberical empowerment values
            "show_empowerment_values_b": True,
            #set the relative speed of the dogs to the sheep, default is 1
            "dog_2_sheep_speed": 4,
            #set the uncertainty between the click position and where a new dog appears.  Default is 0
            "spawn_radius": 2
        },
        "sim" : {
            #the simulation length.  One round of moves (dog then sheep) counts as a tick.
            "time_limit" : 500,
            #the speed of the simulation when it's being visualised.  Integer values of 2-50 are reasonable.  Lower values cause the agents to move slower
            "tick_rate" : 15,
        }
    }
    np.save(fname, cfg)


def hash(cfg):

    # Ref: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    hash = hashlib.md5()
    encoded = json.dumps(cfg, sort_keys=True, cls=NumpyEncoder).encode()
    hash.update(encoded)

    return hash.hexdigest()


def main():
    saveConfig()
    cfg = loadConfig()
    return cfg


if __name__ == '__main__':
    main()
