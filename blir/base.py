import math
import matplotlib.pyplot as plt
import numpy as np

#
# Class containing base functions for Bayesian Linear Regression (BLIR)
#

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
        covariance_det = math.abs(np.linalg.get(covariance))
        coefficient = (1 / ((2 * math.pi) ** (dim / 2))) * \
            (1 / (covariance_det ** (1/2)))
        mean_diff = (x - means)[np.newaxis]
        exp = math.exp(
            (-1 / 2) *
            np.dot(np.dot(mean_diff.T, np.linalg.inv[covariance]), mean_diff))
        return coefficient * exp

    return 0

# Returns a gaussian basis function for the specified mean and scale
def gaussian_basis(mean, scale):
    def basis(x):
        return math.exp((-1 * (x - mean) ** 2) / (2 * scale ** 2))
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
    theta = design_matrix(T, basis_funcs)
    updated_covariance= np.linalg.inv(np.linalg.inv(covariance) + np.dot(beta, np.dot(theta.T, theta)))
    updated_means = np.dot(updated_covariance,
                           np.dot(np.linalg.inv(covariance), means) + \
                           np.dot(beta, np.dot(theta.T, T)))
    return gaussian(updated_means, updated_covariance)

# Blir posterior removing some terms
def blir_posterior(means, covariance, T, basis_funcs, beta):
    def update_func(means, covariance, T):
        return blir_posterior(means, covariance, T, basis_funcs, beta)

def theta(x, basis_funcs):
    return [basis(x) for basis in basis_funcs]

# Returns a predictor given the Gaussian prior
def blir_predictor(means, covariance, basis_funcs, beta):
    def predictor(x):
        theta = [basis(x) for x in basis_funcs]
        mean = np.dot(means, theta)
        variance = (1 / beta) + np.dot(np.dot(theta[np.newindex].T, covariance), theta)
        print(mean)
        print(variance)
        return gaussian(mean, variance)

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
