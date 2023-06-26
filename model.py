class Model():
    """
    Base class for user's model. User should subclass this
    and implement log_likelihood, names and bounds
    """
    __metaclass__ = ABCMeta
    names=[] # Names of parameters, e.g. ['p1','p2']
    bounds=[] # Bounds of prior as list of tuples, e.g. [(min1,max1), (min2,max2), ...]

    def in_bounds(self,param):
        """
        Checks whether param lies within the bounds
        -----------
        Parameters:
            param: :obj:`raynest.parameter.LivePoint`
        -----------
        Return:
            True: if all dimensions are within the bounds
            False: otherwise
        """
        return all(self.bounds[i][0] < param.values[i] < self.bounds[i][1] for i in range(param.dimension))

    def new_point(self):
        """
        Create a new LivePoint, drawn from within bounds
        -----------
        Return:
            p: :obj:`raynest.parameter.LivePoint`
        """
        
        logP = -inf
        while(logP==-inf):
            p = LivePoint(self.names, d=np.array([np.random.uniform(self.bounds[i][0],self.bounds[i][1]) for i, _ in enumerate(self.names) ]))
            logP=self.log_prior(p)
        return p

    @abstractmethod
    def log_likelihood(self,param):
        """
        returns log likelihood of given parameter
        ------------
        Parameter:
            param: :obj:`raynest.parameter.LivePoint`
        """
        pass

    def log_prior(self,param):
        """
        Returns log of prior.
        Default is flat prior within bounds
        ----------
        Parameter:
            param: :obj:`raynest.parameter.LivePoint`
        ----------
        Return:
            0 if param is in bounds
            -np.inf otherwise
        """
        if self.in_bounds(param):
            return 0.0
        else: return -inf

    
    @abstractmethod
    def force(self,param):
        """
        returns the force (-grad potential)
        Required for Hamiltonian sampling
        ----------
        Parameter:
        param: :obj:`raynest.parameter.LivePoint`
        """
        pass

    @abstractmethod
    def analytical_gradient(self,param):
        """
        returns the gradient of the likelihood (-grad potential)
        Required for Hamiltonian sampling
        ----------
        Parameter:
        param: :obj:`raynest.parameter.LivePoint`
        """
        pass



    def header(self):
        """
        Return a string with the output file header
        """
        return '\t'.join(self.names) + '\tlogL'
