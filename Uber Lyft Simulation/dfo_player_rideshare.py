import numpy as np
import ipdb
import sys
sys.path.append("../utils/")
from utils_rideshare import sample_sphere


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

    def initialize_theta(self, d, theta_0=None):
        if len(self.theta_history) != 0:
            raise ValueError("Theta has already been initialized")
        elif theta_0 is not None:
            theta_init = theta_0
            self.theta_history.append(theta_init)
        else:
#             theta_init = np.random.normal(size = d)
            theta_init = np.zeros((d,1))
            self.theta_history.append(theta_init)
        return theta_init
    
    def perturb_theta(self):
        u = sample_sphere(self.delta*np.log((len(self.theta_history)+2))/(len(self.theta_history)+2),
                          len(self.theta_history[-1]))
        self.u_history.append(u)
        return self.theta_history[-1]+u
    
    def update_theta(self,oracle_risk):
        self.risk_history.append(oracle_risk)
        batch_size = len(oracle_risk)
        theta_new = self.theta_history[-1]-(self.eta/np.log(np.log(len(self.theta_history)+2)+2)*
                                            np.dot(np.array(self.u_history[-batch_size:]).reshape(-1,batch_size), np.array(oracle_risk).reshape((batch_size,1)))/batch_size)
#         theta_new = self.theta_history[-1]-((self.eta/np.log10((len(self.theta_history)/10+2))*oracle_risk*self.u_history[-1]))
#         print(f'theta_old = {self.theta_history[-1]}')
#         print(f'theta_new = {theta_new}')
        self.theta_history.append(theta_new)
        return theta_new
