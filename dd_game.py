import numpy as np
import ipdb
from utils import evaluate_test_performative_risk, solve_theta_PO

"""
Class to run a decision dependent game with 2 players
"""

class DecisionDependentGame(object):

    def __init__(self, p1, p2, p1_data_params, p2_data_params, p1_data_generating_func, p2_data_generating_func, num_rounds, num_alternate_rounds, num_test):
        """
        p1, p2: players. Instances of player abstract class
        p1_data_generating_func, p2_data_generating_func: functions that generate data for p1, p2
        p1_data_params, p2_data_params: configurations to generate performative data for both players. Tuple of (cov_x, sigma_y, beta, mu, gamma)

        num_rounds: number of training samples
        num_alternate_rounds: how many rounds the players alternate for after all data has been observed
        num_test: number of test samples
        """
        self.p1 = p1
        self.p1_generate_data_func = p1_data_generating_func
        self.p1_data_params = p1_data_params
        self.theta_p1 = None

        self.p2 = p2
        self.p2_generate_data_func = p2_data_generating_func
        self.p2_data_params = p2_data_params
        self.theta_p2 = None

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

    def solve_nash(self):
        #TODO Generalize Yonghun's solve_theta_PO function to work for different cov_x, sigma_y
        cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1 = self.p1_data_params
        cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2 = self.p2_data_params
        theta_PO_p1, theta_PO_p2 = solve_theta_PO(mu_p1, mu_p2, gamma_p1, gamma_p2, beta_p1, beta_p2, cov_x_p1)
        return theta_PO_p1, theta_PO_p2

    def evaluate_test_perf_risk_p1(self):
        cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1 = self.p1_data_params
        mse_avg = evaluate_test_performative_risk(beta_p1, mu_p1, gamma_p1, self.theta_p1, self.theta_p2, cov_x_p1, sigma_y_p1, self.num_test)
        return mse_avg

    def evaluate_test_perf_risk_p2(self):
        cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2 = self.p2_data_params
        mse_avg = evaluate_test_performative_risk(beta_p2, mu_p2, gamma_p2, self.theta_p1, self.theta_p2, cov_x_p2, sigma_y_p2, self.num_test)
        return mse_avg

    def run_post_train_alternating(self):
        pass

    def run_train(self):

        for t in range(self.num_rounds):
            cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1 = self.p1_data_params
            z_p1 = self.p1_generate_data_func(cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1, self.theta_p1, self.theta_p2)

            cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2 = self.p2_data_params
            z_p2 = self.p2_generate_data_func(cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2, self.theta_p1, self.theta_p2)

            theta_p1_new = p1.update_theta(t, self.num_rounds, z_p1, theta_p2)
            theta_p2_new  = p2.update_theta(t, self.num_rounds, z_p2, theta_p1)
            self.theta_p1 = theta_p1_new
            self.theta_p2 = theta_p2_new

        return theta_p1, theta_p2
