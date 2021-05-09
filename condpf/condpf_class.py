import matplotlib.pyplot as plt
import numpy as np
import torch
tod = torch.distributions

class CondPF(torch.nn.Module):

    def __init__(self, model, param):
        super().__init__()
        self.mu = model.mu
        self.sigma = model.sigma
        self.llg = model.likelihood_logscale
        self.l = param[0]
        self.T = param[1]
        self.N = param[2]
        self.dx = param[3]
        self.dy = param[4]
        self.initial_val = param[5]
        
    # discrete unit update with euler, input x of shape (N,dx)
    def unit_update(self, x):
        hl = 2**(-self.l)
        for dt in range(2**self.l):
            dw = torch.randn(x.shape[0],self.dx,1) * np.sqrt(hl)
            x = x + self.mu(x)*hl + (self.sigma(x)@dw)[...,0]
        return x
        
    # initial path generation, output shape (T+1, dx)
    def initial_path_gen(self):
        un = torch.zeros(self.T+1, 1, self.dx) + self.initial_val        
        for t in range(self.T):
            un[t+1] = self.unit_update(un[t])
        return torch.squeeze(un)
    
    # Resampling input multi-dimensional particle x
    def resampling(self, weight, gn, x):
        N = self.N
        ess = 1/((weight**2).sum())
        if ess <= (N/2):
            ## Sample with uniform dice
            dice = np.random.random_sample(N)
            ## np.cumsum obtains CDF out of PMF
            bins = np.cumsum(weight)
            ## np.digitize gets the indice of the bins where the dice belongs to 
            x_hat = x[:,np.digitize(dice,bins),:]
            ## after resampling we reset the accumulating weight
            gn = torch.zeros(N)
        if ess > (N/2):
            x_hat = x

        return x_hat, gn
    
    # Resampling input multi-dimensional particle x
    def pure_resampling(self, weight, gn, x):
        N = self.N
        ## Sample with uniform dice
        dice = np.random.random_sample(N)
        ## np.cumsum obtains CDF out of PMF
        bins = np.cumsum(weight)
        ## np.digitize gets the indice of the bins where the dice belongs to 
        x_hat = x[:,np.digitize(dice,bins),:]
        ## after resampling we reset the accumulating weight
        gn = torch.zeros(N)
        
        return x_hat, gn
    
    
    # Sampling out according to the weight
    def sample_output(self, weight, x):
        ## Sample with uniform dice
        dice = np.random.random_sample(1)
        ## np.cumsum obtains CDF out of PMF
        bins = np.cumsum(weight)
        ## np.digitize gets the indice of the bins where the dice belongs to 
        x_hat = x[:,np.digitize(dice,bins),:]
        ## return the sampled particle path
        return torch.squeeze(x_hat)

    def kernel(self, input_path, observe_path):
        
        un = torch.zeros(self.T+1,self.N,self.dx) + self.initial_val
        un_hat = torch.zeros(self.T+1,self.N,self.dx) + self.initial_val
        gn = torch.zeros(self.N)

        for t in range(self.T):
            un[:t+1] = un_hat[:t+1]

            un[t+1] = self.unit_update(un[t])
            # Main point for conditional PF is that the last particle is fixed, and it joins the resampling process
            un[:,-1] = input_path

            # Cumulating weight function

            # !! llg takes first x and then y !!
            gn = self.llg(un[t+1],observe_path[t+1]) + gn
            what = torch.exp(gn-torch.max(gn))
            wn = what / torch.sum(what)
            wn = wn.detach().numpy()

            # Resampling
            un_hat[:t+2], gn = self.resampling(wn, gn, un[:t+2])
            un_hat[:,-1] = input_path

        # Sample out a path and output it
        return self.sample_output(wn, un)
    
    def chain_gen(self, num_step, observe_path):
        x_chain = torch.zeros(num_step+1,self.T+1,self.dx)
        x_chain[0] = self.initial_path_gen()
        for step in range(num_step):
            x_chain[step+1] = self.kernel(x_chain[step],observe_path)
        return x_chain