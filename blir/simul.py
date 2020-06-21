import math

def test_func(x):
    return math.sin(2 * math.pi * x)

def sample(n):
    x = np.linspace(0, 1, n)
    y = np.array(list(map(test_func, x)))
    return (x, y)

def rand_sample(n):
    x = np.random.random_sample((n,))
    y = np.array(list(map(test_func, x)))
    return (x, y)

x = np.linspace(0, 1, 500)
y = np.array(list(map(test_func, x)))

alpha, beta = 0.1, 500000
means, covariance = initial_weights(9, 1)
basis_funcs = [lambda x: 1,
               gaussian_basis(0.125, 1),
               gaussian_basis(0.25, 1),
               gaussian_basis(0.375, 1),
               gaussian_basis(0.5, 1),
               gaussian_basis(0.625, 1),
               gaussian_basis(0.75, 1),
               gaussian_basis(0.875, 1),
               gaussian_basis(1, 1)]

def plot_predictor(predictor, n, l):
    x = np.linspace(0, 1, n)
    y = [list(t) for t in zip(*np.array(list(map(predictor, x))))]
    means = y[0]
    variance = y[1]
    # plt.plot(x, means, 'o', label=l)
    print(variance)
    plt.errorbar(x, means, variance, label=l)

def update(means, covariance, basis_funcs, beta, n):
    (x, t) = sample(n)
    (updated_means, updated_covariance) = update_weights(means, covariance, x, t, basis_funcs, beta)
    return (blir_predictor(updated_means, updated_covariance, basis_funcs, beta), updated_means, updated_covariance)

def update_iter(means, covariance, basis_funcs, beta, n):
    if n == 0:
        return (blir_predictor(means, covariance, basis_funcs, beta), means, covariance)
    x, t = rand_sample(10)
    (updated_means, updated_covariance) = update_weights(means, covariance, x, t, basis_funcs, beta)
    return update_iter(updated_means, updated_covariance, basis_funcs, beta, n-10)

predictor = blir_predictor(means, covariance, basis_funcs, beta)
(predictor, means, covariance) = update(means, covariance, basis_funcs, beta, 5)

plot_predictor(predictor, 20, "N=100")
(predictor, means, variance) = update(means, covariance, basis_funcs, beta, 100)
plot_predictor(predictor, 20, "N=1000")
(predictor, means, variance) = update(means, covariance, basis_funcs, beta, 1000)

# r = rand_sample(100)
# print(r[1])
# plt.plot(r[0], r[1], label='Rand Samples')


ground_truth = sample(1000)
plt.plot(ground_truth[0], ground_truth[1], label='Ground Truth')
plt.legend()
plt.show()
