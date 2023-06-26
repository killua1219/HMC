class straight_line(Model):
    
    def __init__(self, sample, err_x, err_y):
        self.sample = sample
        self.err_x = err_x
        self.err_y = err_y
        self.dim = len(self.sample[0])
        
        self.names = ['x' + str(i) for i in range(1, self.dim+1)]
        self.names = self.names + ['coef', 'intercetta']
        self.bounds = [(sample[0][i]-3*err_x,sample[0][i]+3*err_x) for i in range(self.dim)]
        self.bounds.append((0,100))
        self.bounds.append((0,100))
        
    def log_likelihood(self, parameters):
        
        dim = len(self.sample[0])
        
        x = (self.sample[:][0] - parameters[:dim])
        y = (self.sample[:][1] - (parameters[-2]*parameters[:dim] + parameters[-1]))
        
        return -0.5*np.sum((x/self.err_x)**2 + (y/self.err_y)**2)
