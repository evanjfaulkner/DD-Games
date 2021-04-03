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

def sample_from_location_family(cov_x, sigma_y, beta, mu, gamma, theta_1, theta_2):
    x = np.random.multivariate_normal(np.zeros(len(cov_x)), cov_x)
    U_y = np.random.normal(0, sigma_y)
    y = beta.T @ x + mu.T @ theta_1 + gamma.T @ theta_2 + U_y
    return (x, y)

def evaluate_test_performative_risk(beta, mu, gamma, theta_p1, theta_p2, cov_x, sigma_y, num_test):
    test_set = [sample_from_distribution(cov_x, sigma_y, beta, mu, gamma, theta_p1, theta_p2) for e in range(num_test)]
    x_test = np.array([e[0] for e in test_set])
    y_test = [e[1] for e in test_set]
    y_pred = x_test @ theta_p1
    mse = np.linalg.norm(y_test - y_pred)**2
    mse_avg = mse/len(y_test)
    return mse_avg

def solve_theta_PO(mu_1,mu_2,gamma_1,gamma_2, beta_1,beta_2,Sigma_x):
    """
    Solves for the performative optima/Nash equilibrium of the DD Game
    """
    mu_Sig_1 = np.outer(mu_1, mu_1) - Sigma_x
    mu_Sig_2 = np.outer(mu_2, mu_2) - Sigma_x
    mu_gamma_1 = np.outer(mu_1, gamma_1)
    mu_gamma_2 = np.outer(mu_2, gamma_2)
    A_1 = np.linalg.inv(mu_Sig_2 @ np.linalg.inv(mu_gamma_1) @ mu_Sig_1 - mu_gamma_2)
    A_2 = np.linalg.inv(mu_Sig_1 @ np.linalg.inv(mu_gamma_2) @ mu_Sig_2 - mu_gamma_1)

    B_1 = mu_Sig_2 @ np.linalg.inv(mu_gamma_1) @ Sigma_x @ beta_1 - Sigma_x @ beta_2
    B_2 = mu_Sig_1 @ np.linalg.inv(mu_gamma_2) @ Sigma_x @ beta_2 - Sigma_x @ beta_1


    theta_PO_1 = A_1 @ B_1
    theta_PO_2 = A_2 @ B_2
    return theta_PO_1, theta_PO_2
