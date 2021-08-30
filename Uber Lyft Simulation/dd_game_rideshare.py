import numpy as np
import sys
sys.path.append("./utils/")
from utils_rideshare import solve_theta, evaluate_test_performative_risk


"""
Class to run a decision dependent game with 2 players
"""

class DecisionDependentGame(object):

    def __init__(self, p1, p2, p1_data_params, p2_data_params,
                 p1_data_generating_func, p2_data_generating_func,
                 num_rounds=100000, num_alternate_rounds=100, num_test=1000):
        """
        p1, p2: players. Instances of player abstract class
        p1_data_generating_func, p2_data_generating_func: functions that generate data for p1, p2
        p1_data_params, p2_data_params: configurations to generate performative data for both players. Tuple of (data, dimension, sigma_y, mu, gamma)

        num_rounds: number of training samples
        num_alternate_rounds: how many rounds the players alternate for after all data has been observed
        num_test: number of test samples
        """
        self.p1 = p1
        self.p1_generate_data_func = p1_data_generating_func
        self.p1_data_params = p1_data_params
        self.data_p1 = self.p1_data_params[0]
        self.d1 = self.p1_data_params[1]
        self.theta_p1 = p1.initialize_theta(self.d1, p1_data_params[-1])

        self.p2 = p2
        self.p2_generate_data_func = p2_data_generating_func
        self.p2_data_params = p2_data_params
        self.data_p2 = self.p2_data_params[0]
        self.d2 = self.p2_data_params[1]
        self.theta_p2 = p2.initialize_theta(self.d2, p2_data_params[-1])

        self.num_rounds = num_rounds
        self.num_test = num_test
        self.num_alternate_rounds = num_alternate_rounds

    def get_p1(self):
        return self.p1

    def get_p2(self):
        return self.p2

    def get_num_rounds(self):
        return self.num_rounds

    def get_num_test(self):
        return self.num_test

    def evaluate_test_perf_risk_p1(self):
        data_p1, d1, sigma_y_p1, mu_p1, gamma_p1, theta_1_0 = self.p1_data_params
        mse_avg = evaluate_test_performative_risk(self.p1_generate_data_func,
                                                  data_p1, mu_p1, gamma_p1,
                                                  self.theta_p1, self.theta_p2,
                                                  sigma_y_p1,
                                                  self.num_test)
        return mse_avg

    def evaluate_test_perf_risk_p2(self):
        data_p2, d2, sigma_y_p2, mu_p2, gamma_p2, theta_2_0 = self.p2_data_params
        #The ordering between p1 and p2 gets flipped
        mse_avg = evaluate_test_performative_risk(self.p2_generate_data_func,
                                                  data_p2, mu_p2, gamma_p2,
                                                  self.theta_p2, self.theta_p1,
                                                  sigma_y_p2,
                                                  self.num_test)
        return mse_avg

    def run_post_train_alternating(self):
        for t in range(self.num_alternate_rounds):
            theta_p1_new = self.p1.update_theta_without_observations(self.theta_p2)
            theta_p2_new = self.p2.update_theta_without_observations(self.theta_p1)
            self.theta_p1 = theta_p1_new
            self.theta_p2 = theta_p2_new

    def run_train(self):

        for t in range(self.num_rounds):
            p1_data, d1, sigma_y_p1, mu_p1, gamma_p1, theta_1_0 = self.p1_data_params
            z_p1 = self.p1_generate_data_func(p1_data, sigma_y_p1,
                                              mu_p1, gamma_p1,
                                              self.theta_p1, self.theta_p2)

            p2_data, d2, sigma_y_p2, mu_p2, gamma_p2, theta_2_0 = self.p2_data_params
            z_p2 = self.p2_generate_data_func(p2_data, sigma_y_p2,
                                              mu_p2, gamma_p2,
                                              self.theta_p2, self.theta_p1)

            theta_p1_new = self.p1.update_theta_with_observations(t, self.num_rounds,
                                                                  z_p1, d1, self.theta_p2)
            theta_p2_new = self.p2.update_theta_with_observations(t, self.num_rounds,
                                                                  z_p2, d2, self.theta_p1)
            self.theta_p1 = theta_p1_new
            self.theta_p2 = theta_p2_new

        return self.theta_p1, self.theta_p2
