import numpy as np

def compute_mean(data):
    """Compute the mean of a list of numbers."""
    return np.mean(data)

def compute_variance(data):
    """Compute the variance of a list of numbers."""
    return np.var(data, ddof=1)

def compute_covariance(data1, data2):
    """Compute the covariance between two lists of numbers."""
    return np.cov(data1, data2, ddof=1)[0, 1]

def gaussian_distribution(x, mean, variance):
    """Compute the Gaussian distribution value for a given x, mean, and variance."""
    return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))


data = [75, 85, 95, 65, 70]
data1 = [20, 25, 30, 35, 40]
data2 = [5, 15, 25, 35, 45]

mean = compute_mean(data)
variance = compute_variance(data)
covariance = compute_covariance(data1, data2)
gaussian_value = gaussian_distribution(5, mean, variance)

print(f"Mean: {mean}")
print(f"Variance: {variance}")
print(f"Covariance: {covariance}")
print(f"Gaussian value at x=5: {gaussian_value}")
