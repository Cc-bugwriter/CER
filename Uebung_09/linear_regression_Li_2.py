import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def linear_features(x):
    """
    Computes the matrix of linear features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        psi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of psi are the features of each data point given in x
    """
    [h,w] = x.shape
    psi = np.ones((h,2))
    for i in range(h):
        psi[i,1] = x[i]
    psi = psi.T

    return psi

def quadratic_features(x):
    """
    Computes the matrix of quadratic independent features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        psi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of psi are the features of each data point given in x
    """
    [h,w] = x.shape
    psi = []
    for i in range(3):
        psi.append(x**(i))
    psi = np.array(psi).reshape((3,h))
    return psi

def cubic_features(x):
    """
    Computes the matrix of cubic independent features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        psi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of psi are the features of each data point given in x
    """
    [h,w] = x.shape
    psi = []
    for i in range(4):
        psi.append(x**(i))
    psi = np.array(psi).reshape((4,h))
    return psi


def quartic_features(x):
    """
    Computes the matrix of quartic independent features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        psi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of psi are the features of each data point given in x
    """
    [h,w] = x.shape
    psi = []
    for i in range(5):
        psi.append(x**(i))
    psi = np.array(psi).reshape((5,h))
    return psi


def tenth_features(x):
    """
    Computes the matrix of tenth independent features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        psi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of psi are the features of each data point given in x
    """
    [h,w] = x.shape
    psi = []
    for i in range(11):
        psi.append(x**(i))
    psi = np.array(psi).reshape((11,h))
    return psi


def twentieth_features(x):
    """
    Computes the matrix of twentieth independent features of points x.

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
    returns
        psi: np.ndarray, np.float64, shape (D, N) -> feature matrix
            The feature of a single data point is D-dimensional.
            The columns of psi are the features of each data point given in x
    """
    [h,w] = x.shape
    psi = []
    for i in range(21):
        psi.append(x**(i))
    psi = np.array(psi).reshape((21,h))
    return psi





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

    Psi = feature_function(x)
    theta = np.linalg.inv(Psi @ (Psi.T)) @ Psi @ y
    mse = (np.sum((y.T-((theta.T)@Psi).T)**2)/101)

    print('---Results')
    print('theta: ', theta.ravel())
    print('mse: ', mse.item())

    return theta

def plot_linear_regression_2d(x, y, theta, feature_function):
    """
    Plots the raw data, the predictions from the linear regression model at the training points x,
    and the curve resulting from applying the model to xplot (see below in the code).

    inputs
        x: np.ndarray, np.float64, shape (N, d_x) -> data x
            N are the number of data points and d_x the dimension of a single data point
        y: np.ndarray, np.float64, shape (N, 1) -> data y
            y is 1-dimensional
        theta: np.ndarray, np.float64, shape (D, 1) -> optimal parameters from linear regression
        feature_function: callable function -> computes the feature matrix of the data x, psi(x)

    """
    Psi = feature_function(x)
    xplot = np.linspace(np.min(x.ravel()), np.max(x.ravel()), 10000).reshape((-1, 1))
    plt.scatter(x, y, c='blue', label='raw data')
    y_hat = (theta.T)@ Psi
    plt.scatter(x, y_hat, c='green', label='predicted data')
    y_hat_continuous = (theta.T)@ feature_function(xplot)
    plt.plot(xplot, y_hat_continuous.T, c='red', label='model')
    plt.title(feature_function.__name__)
    plt.legend()
    plt.show()
# Task d)
x = np.load('./data/x_a.npy')
y = np.load('./data/y_a.npy')
theta = linear_regression(x, y, linear_features)
plot_linear_regression_2d(x, y, theta, linear_features)

x = np.load('./data/x_b.npy')
y = np.load('./data/y_b.npy')
theta = linear_regression(x, y, linear_features)
plot_linear_regression_2d(x, y, theta, linear_features)

# Task f)
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
