import numpy as np
import ipdb

"""
Data
"""
def sample_from_location_family_rideshare(g, mu, gamma, theta_me, theta_other):
    
    y = g + mu.T @ theta_me + gamma.T @ theta_other
    return y

def sample_sphere(epsilon,d):
    """
    Returns a point on the sphere in R^d of radius epsilon
    """
    x = np.random.normal(size=d)
    x /= np.linalg.norm(x)
    x *= epsilon
    return x


"""
Evaluate PR
"""

def evaluate_performative_risk(demand_generator, g, mu, gamma, lambda_r,
                               theta_me, theta_other):
    
    demand = demand_generator(g, mu, gamma, theta_me, theta_other)
    risk = (-demand*(theta_me+10))+(lambda_r*np.linalg.norm(theta_me)/2)
    return risk    


"""
Helpers for TwoStage Player
"""

def solve_distribution_params(z_lst, theta_me_lst, theta_other_lst):
    y = np.array([e.squeeze() for e in z_lst])
    A = np.hstack((theta_me_lst, theta_other_lst))
    mu_tilde = np.linalg.pinv(A.T @ A) @ A.T @ y
    num_me = int(np.size(theta_me_lst, axis=1))
    mu_hat = mu_tilde[:num_me]
    gamma_hat = mu_tilde[num_me:]
    return mu_hat, gamma_hat

def find_qs(mu_hat, gamma_hat, z_lst, theta_me_lst, theta_other_lst):
    y_lst = [e for e in z_lst]
    q_lst = []
    for (idx, y) in enumerate(y_lst):
        theta_me = theta_me_lst[idx]
        theta_other = theta_other_lst[idx]
        q = y - np.dot(mu_hat.T, theta_me) - np.dot(gamma_hat.T, theta_other)
        q_lst.append(q)
    return np.array(q_lst)

def solve_theta(g, eta, lambda_r, mu_hat, gamma_hat, theta_me, theta_other):
    theta = ((10*mu_hat) + g + (gamma_hat*theta_other))/(lambda_r-(2*mu_hat))
    return theta

def gd_theta(g, eta, lambda_r, t, mu_hat, gamma_hat, theta_me, theta_other):
    theta = theta_me - (eta/np.log(t+2))*(-g - (10*mu_hat) - (2*mu_hat*theta_me) - (gamma_hat*theta_other) + (lambda_r*theta_me))
    return theta

def solve_mu(z_lst, theta_me_lst):
    y = [e[1] for e in z_lst]
    A = np.array(theta_me_lst)
    mu_hat = np.linalg.pinv(A.T @ A) @ A.T @ y
    return mu_hat