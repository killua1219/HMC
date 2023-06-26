class D1gaussian(Model):
    
    def __init__(self):
        self.names=['mean']
        self.bounds=[(-100,100)]
        
    def log_likelihood(self, parameters):
        return -0.5*np.sum(parameters**2)
    
    def analytical_gradient(self,parameters):
        return -np.sum(parameters)
