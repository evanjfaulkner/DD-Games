import numpy as np
import ipdb
import sys
sys.path.append("./utils/")
from utils_functions import sample_sphere


class DFOPlayer(object):
    """
    Player that implements derivative free optimization
    bandit gradient descent algorithm from 
    "Online convex optimization in the bandit setting:
    gradient descent without a gradient", Flaxman et. al., 2008
    """

    def __init__(self, delta=1, eta=1e-2):
        self.theta_history = []
        self.u_history = []
        self.risk_history = []

        self.delta = delta  # Radius of the sphere for perturbed evaluations
        self.eta = eta  # Initial step size

    def initialize_theta(self, d):
        if len(self.theta_history) != 0:
            raise ValueError("Theta has already been initialized")
#         theta_init = 0.5*np.random.normal(size = d)
        theta_init = np.zeros(d)
        self.theta_history.append(theta_init)
        return theta_init
    
    def perturb_theta(self):
        u = sample_sphere(self.delta/np.log(len(self.theta_history)+2),
                          len(self.theta_history[-1]))
        self.u_history.append(u)
        return self.theta_history[-1]+u
    
    def update_theta(self,oracle_risk):
        self.risk_history.append(oracle_risk)
        theta_new = self.theta_history[-1]-((self.eta/np.log(len(self.theta_history)+2))*oracle_risk*self.u_history[-1])
        self.theta_history.append(theta_new)
        return theta_new
