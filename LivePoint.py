class LivePoint:

    def __init__(self, names, d):
        self.names = names
        self.dimension = len(names)
        if d is not None:
            self.values             = np.array(d, dtype=np.float64)
        else:
            self.values             = np.zeros(self.dimension, dtype=np.float64)
          
            
    def keys(self):
        return self.names
