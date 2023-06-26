class log_HMC:
    def __init__(self, user_model, covariance_matrix=None):
        
        self.model = user_model
        self.dim = len(self.model.names)
        
        try:
            if covariance_matrix == None:

                self.covariance_matrix = np.eye(len(user_model.names))
        
        except:
            
            self.covariance_matrix = covariance_matrix
    
    
    def K_energy(self, p):
        
        K = np.dot(p, (1/np.diag(self.covariance_matrix)) * p)
        return np.sum(0.5*K)
    
    
    def U_energy(self, q):
        
        U = -self.model.log_likelihood(q)
        return U
    
    
    def Energy(self, q, p):
        
        return -self.U_energy(q)+self.K_energy(p)
    
    
    def logarithmic_Energy(self, q, p, B):
        
        return -np.log(self.U_energy(q))+log(self.K_energy(p)+B)
    
    
    def derivative(self, q, time_step):
        
        der=np.zeros(self.dim)
        
        for i in range(self.dim):
            
            matrix = np.repeat(q[:, np.newaxis], 3, axis=1)
            matrix[i][0]-=time_step
            matrix[i][2]+=time_step
            gradient = np.gradient([self.U_energy(matrix[:,0]),self.U_energy(matrix[:,1]),self.U_energy(matrix[:,2])], matrix[i][:])
            der[i]=gradient[1]
            
        return der
    
    
        
    def momentum_update(self):
        
        p = np.random.normal(0, np.diag(self.covariance_matrix))
        
        return p
        
    
    
    def logarithmic_leap_frog(self,time_step, q, p, B ,L):
        
        try:
            i = self.model.new_point().names.index("t")
            
            pass
        
        except:
            
            dt = 0.5 * time_step / self.U_energy(q)
            p += dt * self.derivative(q, dt)
            
            for i in range(L-1):
                
                dt = time_step / (self.K_energy(p) + B)
                q += dt * p / np.diag(self.covariance_matrix)
                
                dt = time_step / self.U_energy(q)
                p += dt * self.derivative(q, dt)
                
            dt = time_step / (self.K_energy(p) + B)
            q += dt * p / np.diag(self.covariance_matrix)    
                
            dt = 0.5 * time_step / self.U_energy(q)
            p += dt * self.derivative(q, dt)
            
        return q,p
    
    
        
    def logarithmic_leap_frog_plot(self, q, p, time_step=1, L = 1000, k=0, l=0, true_function = None):
        
        leapfrog_sample = np.zeros((L, 2*self.dim))
        B = - self.Energy(q,p)
        
        for j in range(L):
            
            q,p=self.logarithmic_leap_frog(time_step, q, p, B, 1)
            leapfrog_sample[j,:self.dim] = q
            leapfrog_sample[j,self.dim:] = p
            
        plt.scatter(leapfrog_sample[:,k],leapfrog_sample[:,self.dim+l])
        plt.title("Phase space")
        plt.xlabel(f"q{k}")
        plt.ylabel(f"p{l}")
        if true_function is not None:
            pass
    

    
    def run(self, time_step ,L, iteration=10000, q=np.array(None)):
        
        parameters_sample = np.zeros((iteration, 2*self.dim))
        rejected_points=0
        
        if q.any() == None:
            q = self.model.new_point().values
        
        for i in tqdm(range(iteration)):
            
            p = self.momentum_update()
            B = - self.Energy(q,p)
            new_q, new_p = self.logarithmic_leap_frog(time_step, q, p, B, L)
            
            alpha = self.logarithmic_Energy(q, p, B) - self.logarithmic_Energy(new_q, new_p, B)
            
            if alpha > 0:
                    q = new_q.copy()
            else:
                acceptance=random.random()
                
                if alpha >= np.log(acceptance):
                        q = new_q.copy()
                        
                else: rejected_points+=1
                    
            parameters_sample[i,:self.dim] = q.copy()
            parameters_sample[i,self.dim:] = p.copy()
            
            
        return parameters_sample,rejected_points
