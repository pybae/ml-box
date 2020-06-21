#
# Class containing base functions for Bayesian Linear Regression (BLIR)
#

import math
import matplotlib.pyplot as plt
import numpy as np


# Returns a gaussian function for the specified mean and variance
# variance is defined as sigma ** 2
def univariate_gaussian(mean, variance):
    def pdf(x):
        return (1 / ((2 * math.pi * variance) ** 1/2)) * \
            math.exp((-1 / 2 * variance) * (x - mean) ** 2)
    return pdf

# Returns a mult-variate gaussian function for the specified mean and covariance
def gaussian(means, covariance):
    def pdf(x):
        dim = len(x)
        covariance_det = abs(np.linalg.det(covariance))
        coefficient = (1 / ((2 * math.pi) ** (dim / 2))) * \
            (1 / (covariance_det ** (1/2)))
        mean_diff = (x - means)[np.newaxis]
        exp = math.exp(
            (-1 / 2) * \
            (mean_diff @ np.linalg.inv(covariance)) @ mean_diff.T)
        return coefficient * exp
    return pdf

# Returns a gaussian basis function for the specified mean and scale
def gaussian_basis(mean, scale):
    def basis(x):
        return math.exp((-1 * (x - mean) ** 2) / (2 * (scale ** 2)))
    return basis

# Returns a design matrix given the basis functions and input vector x, which i
def design_matrix(X, basis_funcs):
    return np.array([[basis(x) for basis in basis_funcs] for x in X])

# Returns a linear regression function given basis_funcs that inputs weights and input vector x
def linear_regression(basis_funcs):
    def y(x, w):
        return np.dot(w, [basis(x) for basis in basis_funcs])
    return y

# Blir prior, which is given by a conjugate gaussian
def blir_prior(means, covariance):
    return gaussian(means, covariance)

# Blir posterior, given the prior means/covariance, set of new observations T, and precision parameter beta
def blir_posterior(means, covariance, T, basis_funcs, beta):
    updated_means, updated_covariance = update_weights(means, covariance, T, basis_funcs, beta)
    return gaussian(updated_means, updated_covariance)

# Update weights for the BLIR posterior
def update_weights(means, covariance, X, T, basis_funcs, beta):
    theta = design_matrix(X, basis_funcs)
    updated_covariance= np.linalg.inv(np.linalg.inv(covariance) + (beta * (theta.T @ theta)))
    updated_means = updated_covariance @ \
                           ((np.linalg.inv(covariance) @ means) + \
                           (beta * (theta.T @ T)))
    return (updated_means, updated_covariance)

def theta(x, basis_funcs):
    return [basis(x) for basis in basis_funcs]

# Simple initial prior with 0 mean and variance dependent on alpha
def initial_weights(n, alpha):
    return (np.zeros(n), np.dot((1 / alpha), np.identity(n)))

# returns a predictor given the Gaussian prior
def blir_predictor(weights, covariance, basis_funcs, beta):
    def predictor(x):
        means = np.array(weights)[np.newaxis]
        theta = np.array([basis(x) for basis in basis_funcs])[np.newaxis]
        mean = np.dot(means, theta.T)[0][0]
        variance = ((1 / beta) + np.dot((theta @ covariance),  theta.T))[0][0]
        # print("mean: " + str(mean))
        # print("variance: " + str(variance))
        return (mean, variance)
    return predictor

# p = gaussian_basis(0, 1)

# y = linear_regression([
#     gaussian_basis(0, 1),
#     gaussian_basis(1, 1),
#     gaussian_basis(2, 1)])

# print(y([0, 1, 10], [1, 1, 1, 1]))

# x = np.linspace(-5, 5, 500)
# y = np.array(list(map(p, x)))


# plt.plot(x, y)
# plt.show()
