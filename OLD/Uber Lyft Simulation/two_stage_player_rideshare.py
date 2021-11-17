import numpy as np
import ipdb
import sys
sys.path.append("../utils/")
from utils_rideshare import find_qs, solve_distribution_params, gd_theta


class TwoStagePlayer(object):
    """
    Player that implements the 3-stage algorithm:
        1. Collect n/2 samples and estimate mu, gamma.
        2. Collect n/2 samples and find qs for training.
        3. Alternating between players without any new data.
    """

    def __init__(self):
        self.theta_history = []
        self.theta_other_history = []
        self.data_history = []
        self.risk_history = []

        #Will get filled in at the end of Stage 1
        self.mu_hat = None
        self.gamma_hat = None

        #Will get filled in at the end of Stage 2, before alternating
        self.x_train = None
        self.qs = None

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

    def update_theta_stage_one(self, z_t, d):
        return np.random.normal(size = (d,1))

    def perform_estimation_between_stages(self):
        self.mu_hat, self.gamma_hat = solve_distribution_params(self.data_history,
                                                                self.theta_history,
                                                                self.theta_other_history)
        return self.mu_hat, self.gamma_hat

    def update_theta_stage_two(self, z_t, d):
        return np.random.normal(size = (d,1))

    def find_qs_after_stage_two(self):
        num_rounds = len(self.data_history)
        num_first_stage = int(num_rounds/2)
        z_train = self.data_history[num_first_stage:]
        x_lst = [e for e in z_train]

        qs = find_qs(self.mu_hat, self.gamma_hat,
                     self.data_history[num_first_stage:],
                     self.theta_history[num_first_stage:],
                     self.theta_other_history[num_first_stage:])

        self.x_train = x_lst
        self.qs = qs
#         print(np.mean(qs, axis=0))
        return qs

    def update_theta_without_observations(self, theta_me, theta_other, lambda_r, eta, t, prices_):
        self.theta_other_history.append(theta_other)
        g = np.mean(self.qs, axis=0)
        theta_new = np.clip(gd_theta(g, prices_, eta, lambda_r, t, self.mu_hat, self.gamma_hat, theta_me, theta_other),-5,5)
        self.theta_history.append(theta_new)
        return theta_new

    def update_theta_with_observations(self, t, num_rounds, z_t, d, theta_other):
        self.data_history.append(z_t)
        self.theta_other_history.append(theta_other)

        theta_new = None
        if t < num_rounds/2 - 1:
            #Stage 1
            theta_new = self.update_theta_stage_one(z_t, d)
        elif self.mu_hat is None:
#             print("Stage 1 finished. Performing estimation now")
            #End of stage 1
            self.perform_estimation_between_stages()
            theta_new = self.update_theta_stage_two(z_t, d)
        elif t < num_rounds - 1:
            #Stage 2
            theta_new = self.update_theta_stage_two(z_t, d)
        else :
#             print("Stage 2 finished. Finding qs now.")
            #End of stage 2
            theta_new = self.update_theta_stage_two(z_t, d)
            self.find_qs_after_stage_two()

        self.theta_history.append(theta_new)
        return theta_new