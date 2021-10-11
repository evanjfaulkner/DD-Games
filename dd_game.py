import numpy as np
import ipdb
import sys
sys.path.append("./utils/")
from utils_functions import solve_theta_PO, solve_theta_SO, solve_theta, evaluate_test_performative_risk, evaluate_closed_performative_risk, sample_from_location_family


"""
Class to run a decision dependent game with 2 players
"""

class DecisionDependentGame(object):

    def __init__(self, p1, p2, p1_data_params, p2_data_params,
                 p1_data_generating_func, p2_data_generating_func,
                 num_rounds=100000, num_alternate_rounds=100, num_test=10000):
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
        cov_x_p1 = self.p1_data_params[0]
        d1 = len(cov_x_p1)
        self.theta_p1 = p1.initialize_theta(d1)

        self.p2 = p2
        self.p2_generate_data_func = p2_data_generating_func
        self.p2_data_params = p2_data_params
        cov_x_p2 = self.p2_data_params[0]
        d2 = len(cov_x_p2)
        self.theta_p2 = p2.initialize_theta(d2)

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
        cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1 = self.p1_data_params
        cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2 = self.p2_data_params
        theta_PO_p1, theta_PO_p2 = solve_theta_PO(mu_p1, mu_p2,
                                                  gamma_p1, gamma_p2,
                                                  beta_p1, beta_p2,
                                                  cov_x_p1, cov_x_p2)
        return theta_PO_p1, theta_PO_p2

    def solve_social_opt(self):
        cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1 = self.p1_data_params
        cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2 = self.p2_data_params
        theta_SO_p1, theta_SO_p2 = solve_theta_SO(mu_p1, mu_p2,
                                                  gamma_p1, gamma_p2,
                                                  beta_p1, beta_p2,
                                                  cov_x_p1, cov_x_p2)
        return theta_SO_p1, theta_SO_p2

    def evaluate_test_perf_risk_p1(self):
        cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1 = self.p1_data_params
        mse_avg = evaluate_test_performative_risk(self.p1_generate_data_func,
                                                  beta_p1, mu_p1, gamma_p1,
                                                  self.theta_p1, self.theta_p2,
                                                  cov_x_p1, sigma_y_p1,
                                                  self.num_test)
        return mse_avg

    def evaluate_test_perf_risk_p2(self):
        cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2 = self.p2_data_params
        #The ordering between p1 and p2 gets flipped
        mse_avg = evaluate_test_performative_risk(self.p2_generate_data_func,
                                                  beta_p2, mu_p2, gamma_p2,
                                                  self.theta_p2, self.theta_p1,
                                                  cov_x_p2, sigma_y_p2,
                                                  self.num_test)
        return mse_avg
    
    def evaluate_closed_perf_risk(self):
        cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1 = self.p1_data_params
        cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2 = self.p2_data_params
        PR_theta_1, PR_theta_2 = evaluate_closed_performative_risk(
            self.theta_p1, self.theta_p2, mu_p1, mu_p2, gamma_p1, gamma_p2,
            beta_p1, beta_p2, cov_x_p1, cov_x_p2, sigma_y_p1, sigma_y_p2)
        return PR_theta_1, PR_theta_2

    def run_post_train_alternating(self):
        for t in range(self.num_alternate_rounds):
            theta_p1_new = self.p1.update_theta_without_observations(self.theta_p2)
            theta_p2_new = self.p2.update_theta_without_observations(self.theta_p1)
            self.theta_p1 = theta_p1_new
            self.theta_p2 = theta_p2_new

    def run_train(self):

        for t in range(self.num_rounds):
            cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1 = self.p1_data_params
            z_p1 = self.p1_generate_data_func(cov_x_p1, sigma_y_p1,
                                              beta_p1, mu_p1, gamma_p1,
                                              self.theta_p1, self.theta_p2)

            cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2 = self.p2_data_params
            z_p2 = self.p2_generate_data_func(cov_x_p2, sigma_y_p2,
                                              beta_p2, mu_p2, gamma_p2,
                                              self.theta_p2, self.theta_p1)

            theta_p1_new = self.p1.update_theta_with_observations(t, self.num_rounds,
                                                                  z_p1, self.theta_p2)
            theta_p2_new = self.p2.update_theta_with_observations(t, self.num_rounds,
                                                                   z_p2, self.theta_p1)
            self.theta_p1 = theta_p1_new
            self.theta_p2 = theta_p2_new

        return self.theta_p1, self.theta_p2
    
    def oracle_grad1(self):
        cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1 = self.p1_data_params
        mu_hat = mu_p1
        gamma_hat = gamma_p1
        theta_me = self.theta_p1
        theta_other = self.theta_p2
        grad = 0
        return grad
    
    def oracle_grad2(self):
        cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2 = self.p2_data_params
        mu_hat = mu_p2
        gamma_hat = gamma_p2
        theta_me = self.theta_p2
        theta_other = self.theta_p1
        grad = 0
        return grad
        
    def oracle_z1(self):
        cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1 = self.p1_data_params
        theta_me = self.theta_p1
        theta_other = self.theta_p2
        z1 = sample_from_location_family(cov_x_p1, sigma_y_p1, beta_p1, mu_p1, gamma_p1, theta_me, theta_other)
        return z1
    
    def oracle_z2(self):
        cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2 = self.p2_data_params
        theta_me = self.theta_p2
        theta_other = self.theta_p1
        z2 = sample_from_location_family(cov_x_p2, sigma_y_p2, beta_p2, mu_p2, gamma_p2, theta_me, theta_other)
        return z2