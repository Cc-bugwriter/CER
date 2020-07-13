import math

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
    d = x.shape[1]
    a = np.zeros(d + 1, dtype=float)
    for i in range(d + 1):
        if i == 0:
            a[i] = 1
        else:
            a[i] = x[i]

    psi = np.transpose(a)
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
    d=x.shape[1]
    a = np.zeros(2*d+1,dtype=float)
    for i in range(2*d+1):
        if i == 0:
            a[i]=1
        if 0<i<d+1:
            a[i] = x[i]
        else:
            a[i] = math.pow(x[i],2)

    psi = np.transpose(a)
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
    d = x.shape[1]
    a = np.zeros(3 * d + 1, dtype=float)
    for i in range(3 * d + 1):
        if i == 0:
            a[i] = 1
        if 0 < i < d + 1:
            a[i] = x[i]
        if 0< i <2*d +1:
            a[i] = math.pow(x[i], 2)
        else:
            a[i] = math.pow(x[i], 3)

    psi = np.transpose(a)
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
    d = x.shape[1]
    a = np.zeros(4 * d + 1, dtype=float)
    for i in range(4 * d + 1):
        if i == 0:
            a[i] = 1
        if 0 < i < d + 1:
            a[i] = x[i]
        if d + 1 <= i < 2 * d + 1:
            a[i] = math.pow(x[i], 2)
        if 2 * d + 1 <= i < 3 * d + 1:
            a[i] = math.pow(x[i], 3)
        else:
            a[i] = math.pow(x[i], 4)

    psi = np.transpose(a)
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
    d = x.shape[1]
    a = np.zeros(10 * d + 1, dtype=float)
    a[0]=1
    for i in range(10 * d + 1):
        for j in range(11):
            if j * d +1< i <= (j+1)*d +1:
                a[i] = math.pow(x[i], j+1)

    psi = np.transpose(a)
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
    d = x.shape[1]
    a = np.zeros(20 * d + 1, dtype=float)
    a[0]=1
    for i in range(20 * d + 1):
        for j in range(21):
            if j * d +1< i <= (j+1)*d +1:
                a[i] = math.pow(x[i], j+1)

    psi = np.transpose(a)
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
    y_T =np.transpose(y)

    Psi = feature_function(x)
    n= Psi.shape[0]
    if n == 1:
        Psi_T = Psi.reshape(1,n)
    else:
        Psi_T = Psi.transpose()

    a = Psi @ Psi_T


    theta =  np.linalg.inv(a) @ Psi @ y

    theta_T = np.transpose(theta)

    mse =  (y_T @ y - y_T @ Psi_T @ theta - theta_T @ Psi @ y + theta_T @ Psi @ Psi_T @ theta )/ Npoints

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
    xplot = np.linspace(np.min(x.ravel()), np.max(x.ravel()), 10000).reshape((-1, 1))
    plt.scatter(x, y, c='blue', label='raw data')
    y_hat = ...
    plt.scatter(x, y_hat, c='green', label='predicted data')
    y_hat_continuous = ...
    plt.plot(xplot, y_hat_continuous, c='red', label='model')
    plt.title(f'{feature_function}')
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