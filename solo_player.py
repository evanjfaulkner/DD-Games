import numpy as np
import ipdb
import sys
sys.path.append("./utils/")
from utils_functions import find_qs, solve_mu, solve_theta


#TODO Make an abstract class for player that this concrete class extends
class SoloPlayer(object):
    """
    Player who ignores the effects of the other players in the game
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

    def initialize_theta(self, d):
        if len(self.theta_history) != 0:
            raise ValueError("Theta has already been initialized")
        theta_init = np.random.normal(size = d)
        self.theta_history.append(theta_init)
        return theta_init

    def update_theta_stage_one(self, z_t):
        d = len(z_t)
        return np.random.normal(size = d)

    def perform_estimation_between_stages(self):
        self.mu_hat = solve_mu(self.data_history,
                               self.theta_history)
        self.gamma_hat = np.zeros(np.shape(self.mu_hat))  # Need to make this the right shape
        return self.mu_hat

    def update_theta_stage_two(self, z_t):
        d = len(z_t)
        return np.random.normal(size = d)

    def find_qs_after_stage_two(self):
        num_rounds = len(self.data_history)
        num_first_stage = int(num_rounds/2)
        z_train = self.data_history[num_first_stage:]
        x_lst = [e[0] for e in z_train]

        qs = find_qs(self.mu_hat, self.gamma_hat,
                     self.data_history[num_first_stage:],
                     self.theta_history[num_first_stage:],
                     self.theta_other_history[num_first_stage:])

        self.x_train = x_lst
        self.qs = qs
        return qs

    def update_theta_without_observations(self, theta_other):
        self.theta_other_history.append(theta_other)
        theta_new = solve_theta(self.x_train, self.qs,
                                self.mu_hat, np.zeros(np.shape(self.mu_hat)),
                                theta_other)
        self.theta_history.append(theta_new)
        return theta_new

    def update_theta_with_observations(self, t, num_rounds, z_t, theta_other):
        self.data_history.append(z_t)
        self.theta_other_history.append(theta_other)

        theta_new = None
        if t < num_rounds/2 - 1:
            #Stage 1
            theta_new = self.update_theta_stage_one(z_t)
        elif self.mu_hat is None:
#             print("Stage 1 finished. Performing estimation now")
            #End of stage 1
            self.perform_estimation_between_stages()
            theta_new = self.update_theta_stage_two(z_t)
        elif t < num_rounds - 1:
            #Stage 2
            theta_new = self.update_theta_stage_two(z_t)
        else :
#             print("Stage 2 finished. Finding qs now.")
            #End of stage 2
            theta_new = self.update_theta_stage_two(z_t)
            self.find_qs_after_stage_two()

        self.theta_history.append(theta_new)
        return theta_new
