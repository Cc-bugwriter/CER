import numpy as np
import matplotlib.pyplot as plt


def linear_features(x):
    """
    Computes the matrix of linear features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        phi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of phi are the features of each data point given in x
    """
    psi = np.hstack([np.ones(x.shape[0]).reshape((-1, 1)), x])
    return psi.T


def quadratic_features(x):
    """
    Computes the matrix of quadratic independent features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        phi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of phi are the features of each data point given in x
    """
    psi = np.hstack([np.ones(x.shape[0]).reshape((-1, 1)), x, x**2])
    return psi.T


def cubic_features(x):
    """
    Computes the matrix of cubic independent features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        phi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of phi are the features of each data point given in x
    """
    psi = np.hstack([np.ones(x.shape[0]).reshape((-1, 1)), x, x**2, x**3])
    return psi.T


def quartic_features(x):
    """
    Computes the matrix of quartic independent features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        phi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of phi are the features of each data point given in x
    """
    psi = np.hstack([np.ones(x.shape[0]).reshape((-1, 1)), x, x**2, x**3, x**4])
    return psi.T


def tenth_features(x):
    """
    Computes the matrix of tenth independent features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        phi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of phi are the features of each data point given in x
    """
    psi = np.hstack([np.ones(x.shape[0]).reshape((-1, 1)),
                     x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10])
    return psi.T


def twentieth_features(x):
    """
    Computes the matrix of twentieth independent features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        phi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of phi are the features of each data point given in x
    """
    psi = np.hstack([np.ones(x.shape[0]).reshape((-1, 1)),
                     x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10,
                     x**11, x**12, x**13, x**14, x**15, x**16, x**17, x**18, x**19, x**20])
    return psi.T


def linear_regression(x, y, feature_function):
    """
    Computes optimal parameters theta for fitting y = theta^T @ psi(x).
    Prints the parameters and Mean Squared Error (MSE) after fitting the data.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
        y: np.ndarray, np.float64, shape (N, 1) -> data y
            y is 1-dimensional
        feature_function: callable function -> computes the feature matrix of the data x, psi(x)
    returns
        theta: np.ndarray, np.float64, shape (D, 1) -> the optimal parameters of linear regression
    """
    Npoints = x.shape[0]

    Phi = feature_function(x)

    theta = np.linalg.inv(Phi @ Phi.T) @ Phi @ y

    mse = 1 / Npoints * (y - Phi.T @ theta).T @ (y - Phi.T @ theta)

    print('---Results')
    print('theta: ', theta.ravel())
    print('mse: ', mse.item())

    return theta


def plot_linear_regression_2d(x, y, theta, feature_function):
    """
    Plots the raw data and the predictions from the linear regression model.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
        y: np.ndarray, np.float64, shape (N, 1) -> data y
            y is 1-dimensional
        theta: np.ndarray, np.float64, shape (D, 1) -> optimal parameters from linear regression
        feature_function: callable function -> computes the feature matrix of the data x, phi(x)

    """
    xplot = np.linspace(np.min(x.ravel()), np.max(x.ravel()), 10000).reshape((-1, 1))
    plt.scatter(x, y, c='blue', label='raw data')
    plt.scatter(x, feature_function(x).T @ theta, c='green', label='predicted data')
    y_hat_xplot = feature_function(xplot).T @ theta
    plt.plot(xplot, y_hat_xplot, c='red', label='model')
    plt.title(feature_function.__name__)
    plt.legend()
    plt.show()


# Task c)
x = np.load('./data/x_a.npy')
y = np.load('./data/y_a.npy')
theta = linear_regression(x, y, linear_features)
plot_linear_regression_2d(x, y, theta, linear_features)

x = np.load('./data/x_b.npy')
y = np.load('./data/y_b.npy')
theta = linear_regression(x, y, linear_features)
plot_linear_regression_2d(x, y, theta, linear_features)

# Task e)
x = np.load('./data/x_c.npy')
y = np.load('./data/y_c.npy')
theta = linear_regression(x, y, linear_features)
plot_linear_regression_2d(x, y, theta, linear_features)

theta = linear_regression(x, y, quadratic_features)
plot_linear_regression_2d(x, y, theta, quadratic_features)

theta = linear_regression(x, y, cubic_features)
plot_linear_regression_2d(x, y, theta, cubic_features)

theta = linear_regression(x, y, quartic_features)
plot_linear_regression_2d(x, y, theta, quartic_features)

theta = linear_regression(x, y, tenth_features)
plot_linear_regression_2d(x, y, theta, tenth_features)

theta = linear_regression(x, y, twentieth_features)
plot_linear_regression_2d(x, y, theta, twentieth_features)
