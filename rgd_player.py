import numpy as np
import ipdb
import sys
sys.path.append("./utils/")
from utils_functions import sample_sphere


class RGDPlayer(object):
    """
    Player that implements repeated gradient descent
    Converges to performatively stable strategies
    """

    def __init__(self, delta=1, eta=1e-2):
        self.theta_history = []
        self.risk_history = []

        self.eta = eta  # Step size

    def initialize_theta(self, d):
        if len(self.theta_history) != 0:
            raise ValueError("Theta has already been initialized")
#         theta_init = 0.5*np.random.normal(size = d)
        theta_init = np.zeros((d,1))
        self.theta_history.append(theta_init)
        return theta_init
    
    def update_theta(self, z):
        x = np.array(z[0]).reshape(2,1)
        y = z[1]
        theta_prev = np.reshape(self.theta_history[-1],(2,1))
        theta_new = theta_prev - ((self.eta/np.log(len(self.theta_history)+2))*(x@x.T@theta_prev-2*y*x))
        self.theta_history.append(theta_new)
        return theta_new    
