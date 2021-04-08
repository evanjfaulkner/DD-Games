import numpy as np
import ipdb

"""
Data
"""
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

def sample_from_location_family(cov_x, sigma_y, beta, mu, gamma, theta_me, theta_other):
    x = np.random.multivariate_normal(np.zeros(len(cov_x)), cov_x)
    U_y = np.random.normal(0, sigma_y)
    y = beta.T @ x + mu.T @ theta_me + gamma.T @ theta_other + U_y
    return (x, y)


"""
Nash and Test PR
"""

def evaluate_test_performative_risk(data_generator, beta, mu, gamma, theta_me, theta_other, cov_x, sigma_y, num_test):
    test_set = [data_generator(cov_x, sigma_y, beta, mu, gamma, theta_me, theta_other) for e in range(num_test)]
    x_test = np.array([e[0] for e in test_set])
    y_test = [e[1] for e in test_set]
    y_pred = x_test @ theta_me
    mse = np.linalg.norm(y_test - y_pred)**2
    mse_avg = mse/len(y_test)
    return mse_avg

def solve_theta_PO(mu_1, mu_2, gamma_1, gamma_2, beta_1, beta_2, Sigma_1, Sigma_2):
    """
    Solves for the performative optima/Nash equilibrium of the DD Game
    """
    d_1 = np.size(mu_1)
    d_2 = np.size(mu_2)
    
    mu_Sig_1 = np.linalg.inv(np.outer(mu_1, mu_1) + Sigma_1)
    mu_Sig_2 = np.linalg.inv(np.outer(mu_2, mu_2) + Sigma_2)
    mu_gamma_1 = np.outer(mu_1, gamma_1) 
    mu_gamma_2 = np.outer(mu_2, gamma_2)
    
    A_1 = np.linalg.inv(np.eye(d_1) - mu_gamma_1 @ mu_Sig_2 @ mu_gamma_2)
    A_2 = np.linalg.inv(np.eye(d_2) - mu_gamma_2 @ mu_Sig_1 @ mu_gamma_1)
    
    B_1 = mu_Sig_1 @ (Sigma_1@beta_1 - mu_gamma_1@mu_Sig_2@Sigma_2@beta_2)
    B_2 = mu_Sig_2 @ (Sigma_2@beta_2 - mu_gamma_2@mu_Sig_1@Sigma_1@beta_1)
    
    theta_PO_1 = A_1 @ B_1
    theta_PO_2 = A_2 @ B_2
    return theta_PO_1, theta_PO_2

def solve_theta_SO(mu_1, mu_2, gamma_1, gamma_2, beta_1, beta_2, Sigma_1, Sigma_2):
    """
    Solves for the Socially Optimal strategy of the DD Game
    """
    d_1 = np.size(mu_1)
    d_2 = np.size(mu_2)
    
    mm_1 = np.outer(mu_1,mu_1)
    mm_2 = np.outer(mu_2,mu_2)
    gg_1 = np.outer(gamma_1,gamma_1)
    gg_2 = np.outer(gamma_2,gamma_2)
    mg_1 = np.outer(mu_1,gamma_1)
    mg_2 = np.outer(mu_2,gamma_2)
    
    mgSig_1 = np.linalg.inv(Sigma_1 + mm_1 + gg_2)
    mgSig_2 = np.linalg.inv(Sigma_2 + gg_1 + mm_2)
    
    A_1 = np.linalg.inv(np.eye(d_1) + (mg_1 + mg_2) @ mgSig_2 @ (Sigma_2 @ beta_2 - (mg_1 + mg_2)))
    A_2 = np.linalg.inv(np.eye(d_2) + (mg_2 + mg_1) @ mgSig_1 @ (Sigma_1 @ beta_1 - (mg_2 + mg_1)))
    
    B_1 = mgSig_1 @ (Sigma_1 @ beta_1)
    B_2 = mgSig_2 @ (Sigma_2 @ beta_2)
    
    theta_SO_1 = A_1 @ B_1
    theta_SO_2 = A_2 @ B_2
    return theta_SO_1, theta_SO_2

"""
Helpers for TwoStage Player
"""

def solve_distribution_params(z_lst, theta_me_lst, theta_other_lst):
    y = [e[1] for e in z_lst]
    A = np.hstack((theta_me_lst, theta_other_lst))
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
