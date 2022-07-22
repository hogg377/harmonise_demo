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
    cfg = {"swarm_size": 20, "faulty_num": 0, "mal_blockers_num": 0, "mal_comms_num": 0,
            "timesteps": 1000}    
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
