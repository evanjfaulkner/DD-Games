import numpy as np
import ipdb

"""
Data
"""
def sample_from_location_family_rideshare(df, sigma_y,
                                          mu, gamma,
                                          theta_me, theta_other):
    
    sample = df.sample()
    x = sample.drop('price', axis=1).to_numpy().T
    U_y = np.random.normal(0, sigma_y)
    y = sample.price.to_numpy() + mu.T @ theta_me + gamma.T @ theta_other + U_y
    return (x, y)

def sample_sphere(epsilon,d):
    """
    Returns a point on the sphere in R^d of radius epsilon
    """
    x = np.random.normal(size=d)
    x /= np.linalg.norm(x)
    x *= epsilon
    return x


"""
Test PR
"""

def evaluate_test_performative_risk(data_generator, data, mu, gamma,
                                    theta_me, theta_other,
                                    sigma_y, num_test):
    
    test_set = [data_generator(data, sigma_y,
                               mu, gamma,
                               theta_me, theta_other)
                for e in range(num_test)]
    x_test = np.array([e[0] for e in test_set])
    y_test = [e[1] for e in test_set]
    y_pred = np.dot(x_test.reshape((-1,1,len(theta_me))),theta_me)
    mse = np.linalg.norm(y_test-y_pred)**2
    mse_avg = mse/len(y_test)
    return mse_avg


"""
Helpers for TwoStage Player
"""

def solve_distribution_params(z_lst, theta_me_lst, theta_other_lst):
    y = [e[1] for e in z_lst]
    A = np.hstack((theta_me_lst, theta_other_lst))
    mu_tilde = np.linalg.pinv(A.T @ A) @ A.T @ y
    num_me = int(np.size(theta_me_lst, axis=1))
    mu_hat = mu_tilde[:num_me]
    gamma_hat = mu_tilde[num_me:]
    return mu_hat, gamma_hat

def find_qs(mu_hat, gamma_hat, z_lst, theta_me_lst, theta_other_lst):
    y_lst = [e[1] for e in z_lst]
    q_lst = []
    for (idx, y) in enumerate(y_lst):
        theta_me = theta_me_lst[idx]
        theta_other = theta_other_lst[idx]
        q = y - np.dot(mu_hat.T, theta_me) - np.dot(gamma_hat.T, theta_other)
        q_lst.append(q)
    return np.array(q_lst)

def solve_theta(x_lst, q, mu_hat, gamma_hat, theta_other):
    y_mod = q + np.dot(gamma_hat.T, theta_other)*np.ones((len(q),1))
    x_arr = np.array(x_lst).squeeze()
    A = x_arr - mu_hat.T
    theta = np.linalg.pinv(A.T @ A) @ A.T @ y_mod
    return theta

# def gd_theta(x_lst, q, mu_hat, gamma_hat, theta_other):

def solve_mu(z_lst, theta_me_lst):
    y = [e[1] for e in z_lst]
    A = np.array(theta_me_lst)
    mu_hat = np.linalg.pinv(A.T @ A) @ A.T @ y
    return mu_hat