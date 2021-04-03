import numpy as np
import ipdb
from utils import evaluate_test_performative_risk, solve_theta_PO

"""
Player that implements the 2-stage algorithm
"""

#TODO Make an abstract class that this concrete class extends
class TwoStagePlayer(object):

    def __init__(self):
        self.theta_history = []
        self.theta_other_history = []
        self.data_history = []

    def update_theta_stage_one(self, t, self.num_rounds, z_t, theta_other):
        pass

    def update_theta_stage_one(self, t, self.num_rounds, z_t, theta_other):
        pass

    def update_theta(self, t, self.num_rounds, z_t, theta_other):
        self.data_history.append(z_t)
        self.theta_other_history.append(theta_other)

        if t <= self.num_rounds/2:
            theta_new = update_theta_stage_one(self, t, self.num_rounds, z_t, theta_other)
        else :
            theta_new = update_theta_stage_two(self, t, self.num_rounds, z_t, theta_other)

        return theta_new

