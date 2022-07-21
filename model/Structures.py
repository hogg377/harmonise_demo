class WorldGrid:
    #TODO: Add a __copy__() method to this class which copys the ids and types

    def __init__(self, width, height) -> None:
        import numpy as np
        self.ids = np.zeros((width, height), dtype=int)
        self.types = np.zeros((width, height), dtype=int)