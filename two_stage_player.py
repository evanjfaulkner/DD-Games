import numpy as np
import ipdb
sys.path.append("utils/")
from utils import evaluate_test_performative_risk, solve_theta_PO, solve_distribution_params, find_qs, solve_theta

"""
Player that implements the 3-stage algorithm:
    1. Collect n/2 samples and estimate mu, gamma.
    2. Collect n/2 samples and find qs for training.
    3. Alternating between players without any new data.
"""

#TODO Make an abstract class for player that this concrete class extends
class TwoStagePlayer(object):

    def __init__(self):
        self.theta_history = []
        self.theta_other_history = []
        self.data_history = []

        #Will get filled in at the end of Stage 1
        self.mu_hat = None
        self.gamma_hat = None

        #Will get filled in at the end of Stage 2, before alternating
        self.x_train = None
        self.qs = None

    def update_theta_stage_one(self, t, self.num_rounds, z_t, theta_other):
        d = len(z_t)
        return np.random.normal(size = d)

    def perform_estimation_between_stages(self):
        #TODO Make sure this works for both players and is not sensitive to the order of inputs
        self.mu_hat, self.gamma_hat = solve_distribution_params(self.data_history, self.theta_history, self.theta_other_history)
        return self.mu_hat, self.gamma_hat

    def update_theta_stage_two(self, t, self.num_rounds, z_t, theta_other):
        d = len(z_t)
        return np.random.normal(size = d)

    def find_qs_after_stage_two(self):
        qs = find_qs(self.mu_hat, self.gamma_hat, self.data_history, self.theta_history, self.theta_other_history)

        num_rounds = len(self.data_history)
        num_first_stage = num_rounds/2
        z_train = self.data_history[num_first_stage:]
        x_lst = [e[0] for e in z_train]

        self.x_train = x_lst
        self.qs = qs
        return qs

    def update_theta_without_observations(self, theta_other):
        theta_new = solve_theta(self.x_train, self.qs, self.mu_hat, self.gamma_hat, theta_other)
        return theta_new


    def update_theta_with_observations(self, t, self.num_rounds, z_t, theta_other):
        self.data_history.append(z_t)
        self.theta_other_history.append(theta_other)

        theta_new = None
        if t <= self.num_rounds/2:
            #Stage 1
            theta_new = self.update_theta_stage_one(self, t, self.num_rounds, z_t, theta_other)
        elif t == self.num_rounds/2 - 1:
            #End of stage 1
            self.perform_estimation_between_stages()
            theta_new = self.update_theta_stage_two(self, t, self.num_rounds, z_t, theta_other)
        elif t < self.num_rounds - 1:
            #Stage 2
            theta_new = self.update_theta_stage_two(self, t, self.num_rounds, z_t, theta_other)
        else :
            #End of stage 2
            theta_new = self.update_theta_stage_two(self, t, self.num_rounds, z_t, theta_other)
            self.find_qs_after_stage_two()

        return theta_new
