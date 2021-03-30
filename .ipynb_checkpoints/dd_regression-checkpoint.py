"""
Helper functions for decision dependent linear regression game simulation
"""
import numpy as np

def obtain_data(n, d_1, d_2, mu_1, gamma_1, mu_2, gamma_2,
                cov_x_1, cov_x_2, sigma_y_1, sigma_y_2,
                beta_1, beta_2):
    
    theta_1_lst, theta_2_lst = [], []
    z_1_lst, z_2_lst = [], []
    
    for i in np.arange(2*n):
        # Both players play random actions to gather information
        theta_1_i, theta_2_i = np.random.normal(size=d_1), np.random.normal(size=d_2)
        theta_1_lst.append(theta_1_i)
        theta_2_lst.append(theta_2_i)

        z_1_i = sample_from_distribution(cov_x_1, sigma_y_1, beta_1, mu_1, gamma_1, theta_1_i, theta_2_i)
        z_2_i = sample_from_distribution(cov_x_2, sigma_y_2, beta_2, mu_2, gamma_2, theta_1_i, theta_2_i)
        z_1_lst.append(z_1_i)
        z_2_lst.append(z_2_i)
    return np.array(z_1_lst), np.array(z_2_lst), np.array(theta_1_lst), np.array(theta_2_lst)


def sample_from_distribution(cov_x, sigma_y, beta, mu, gamma, theta_1, theta_2):
    x = np.random.multivariate_normal(np.zeros(len(cov_x)), cov_x)
    U_y = np.random.normal(0, sigma_y)
    y = beta.T @ x + mu.T @ theta_1 + gamma.T @ theta_2 + U_y
    return (x, y)
    
    
def solve_distribution_params(z_lst, theta_1_lst, theta_2_lst):
    y = [e[1] for e in z_lst]
    A = np.hstack((theta_1_lst, theta_2_lst))
    mu_tilde = np.linalg.pinv(A.T @ A) @ A.T @ y
    
    num_each = int(len(mu_tilde)/2)
    mu_hat = mu_tilde[:num_each]
    gamma_hat = mu_tilde[num_each:]
    return mu_hat, gamma_hat


def find_qs(mu_hat, gamma_hat, z_lst, theta_1_lst, theta_2_lst):
    y_lst = [e[1] for e in z_lst]
    q_lst = []
    for (idx, y) in enumerate(y_lst):
        theta_1 = theta_1_lst[idx]
        theta_2 = theta_2_lst[idx]
        q = y - np.dot(mu_hat, theta_1) - np.dot(gamma_hat, theta_2)
        q_lst.append(q)
    return np.array(q_lst)
        
    
def solve_theta(x_lst, q, mu_hat, gamma_hat, theta_other):
    y_mod = q + np.dot(gamma_hat, theta_other)*np.ones(len(q))
    x_arr = np.array(x_lst)
    A = x_arr - mu_hat
    theta = np.linalg.pinv(A.T @ A) @ A.T @ y_mod
    return theta


def run_first_stage(z_1_lst, z_2_lst, theta_1_lst, theta_2_lst):
    mu_hat_1, gamma_hat_1 = solve_distribution_params(z_1_lst, theta_1_lst, theta_2_lst)
    mu_hat_2, gamma_hat_2 = solve_distribution_params(z_2_lst, theta_1_lst, theta_2_lst)
    return mu_hat_1, gamma_hat_1, mu_hat_2, gamma_hat_2


def run_second_stage(d_1, d_2, x_1_lst, q_1, x_2_lst, q_2, mu_hat_1, 
                     gamma_hat_1, mu_hat_2, gamma_hat_2, num_iter):
    
    theta_1_init, theta_2_init = np.random.normal(size=d_1), np.random.normal(size=d_2)
    theta_1_lst, theta_2_lst = [theta_1_init], [theta_2_init]
    
    for k in range(num_iter):
        #Solve theta for player 1
        theta_2_prev = theta_2_lst[-1]
        theta_1_k= solve_theta(x_1_lst, q_1, mu_hat_1, gamma_hat_1, theta_2_prev)

        #Solve theta for player 2
        theta_1_prev = theta_1_lst[-1]
        theta_2_k= solve_theta(x_2_lst, q_2, mu_hat_2, gamma_hat_2, theta_1_prev)

        theta_1_lst.append(theta_1_k)
        theta_2_lst.append(theta_2_k)
    
    return theta_1_lst, theta_2_lst


def evaluate_test_performative_risk(beta, mu, gamma, theta_p1, theta_p2, cov_x, sigma_y, num_test):
    test_set = [sample_from_distribution(cov_x, sigma_y, beta, mu, gamma, theta_p1, theta_p2) for e in range(num_test)]
    x_test = np.array([e[0] for e in test_set])
    y_test = [e[1] for e in test_set]
    y_pred = x_test @ theta_p1
    mse = np.linalg.norm(y_test - y_pred)**2
    mse_avg = mse/len(y_test)
    return mse_avg


#TODO: cov_x and sigma_y could also be different for the two players. Change this later.
def run_game(n, d_1, d_2, mu_1, gamma_1, mu_2, gamma_2,
             num_iter_stage_two, num_test,
             cov_x_1, cov_x_2, sigma_y_1, sigma_y_2,
             beta_1, beta_2): 
    
    #Obtain and split up the data for both stages
    z_1_lst, z_2_lst, theta_1_lst, theta_2_lst = obtain_data(n, d_1, d_2, mu_1, gamma_1, mu_2, gamma_2,
                                                             cov_x_1, cov_x_2, sigma_y_1, sigma_y_2, beta_1, beta_2)
    
    num_first_stage = n
    z_1_lst_sliced = (z_1_lst[:num_first_stage], z_1_lst[num_first_stage:])
    z_2_lst_sliced = (z_2_lst[:num_first_stage], z_2_lst[num_first_stage:])

    theta_1_lst_sliced = (theta_1_lst[:num_first_stage], theta_1_lst[num_first_stage:])
    theta_2_lst_sliced = (theta_2_lst[:num_first_stage], theta_2_lst[num_first_stage:])
    
    #Stage 1
    mu_hat_1, gamma_hat_1, mu_hat_2, gamma_hat_2 = run_first_stage(z_1_lst_sliced[0], z_2_lst_sliced[0],
                                                                   theta_1_lst_sliced[0], theta_2_lst_sliced[0])
    
#     print(mu_p1)
#     print(mu_hat_p1)
#     print(mu_p2)
#     print(mu_hat_p2)
    
    #Obtain qs for stage 2
    q_1_lst = find_qs(mu_hat_1, gamma_hat_1, z_1_lst_sliced[1], theta_1_lst_sliced[1], theta_2_lst_sliced[1])
    q_2_lst = find_qs(mu_hat_2, gamma_hat_2, z_2_lst_sliced[1], theta_1_lst_sliced[1], theta_2_lst_sliced[1])
    
    #Stage 2
    x_lst_1 = [e[0] for e in z_1_lst_sliced[1]]
    x_lst_2 = [e[0] for e in z_2_lst_sliced[1]]
    
    theta_1_final_lst, theta_2_final_lst = run_second_stage(d_1, d_2, x_lst_1, q_1_lst, x_lst_2, q_2_lst,
                                                            mu_hat_1, gamma_hat_1, mu_hat_2, gamma_hat_2, num_iter_stage_two)
    
    perf_risk_1 = evaluate_test_performative_risk(beta_1, mu_1, gamma_1, theta_1_final_lst[-1],
                                                  theta_2_final_lst[-1], cov_x_1, sigma_y_1, num_test)
    
    
    perf_risk_2 = evaluate_test_performative_risk(beta_2, mu_2, gamma_2, theta_1_final_lst[-1],
                                                  theta_2_final_lst[-1], cov_x_2, sigma_y_2, num_test)
    
    return perf_risk_1, perf_risk_2