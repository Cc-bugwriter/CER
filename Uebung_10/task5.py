import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize as mini
import unittest

'''
Gruppe 183
Yi Cui 2758172
'''


def nonlinear_least_squares(motor_func, pStart, grid, data, approx_model):
    # -----------------------------------------------------------------
    #  Computes the least-squares-parameter for a model function.
    #  N - number of data points
    #  M - number of parameters
    # -----------------------------------------------------------------
    #  input:    
    #   motor_func     func      handle to model function of the motor
    #   pStart         [Mx1]     An initial guess for the parameter vector
    #   grid           [Nx3]     A vector with points at which some measurement
    #                            data exists. The values are
    #                            grid = [uDesired, q, dq]
    #   data           [Nx1]     The measured data at the grid points
    #   approx_model   {dict}    a description of the model
    # -----------------------------------------------------------------
    #  return:
    #   L              func      handle to residual function
    #   dL_dp          func      handle to residual function derivate
    #   p              [Mx1]     the optimized parameter
    #   err_final      float     the sum of the squared error at all gridpoints
    #   success        bool      the return state of the nonlinear solver
    # -----------------------------------------------------------------
    u, _ = motor_func(grid, approx_model, pStart)

    def L(pStart):
        # calculate u_ist
        u_ist, _ = motor_func(grid, approx_model, pStart)

        # initial loss
        loss = 0
        # compute loss function
        for x in range(len(data)):
            lho = data[x - 1] - u_ist[x - 1]
            loss += 0.5 * np.linalg.norm(lho) ** 2
        return loss

    def dL_dp(pStart):
        # calculate u_ist
        u_ist, _ = motor_func(grid, approx_model, pStart)

        # initial loss
        loss = 0
        # compute loss function
        for x in range(len(data)):
            lho = data[x - 1] - u_ist[x - 1]
            loss += -lho
        return loss

    # use scipy to minimize
    results = mini(L, x0=pStart, method='SLSQP', tol=0.00001, options={'maxiter': 1000})
    # results = mini(dL_dp, x0=pStart, method='SLSQP', tol=0.00001, options={'maxiter': 1000})

    # record result
    p = results.x.reshape((-1, 1))
    success = results.success

    err_final = L(p)

    return L, dL_dp, p, err_final, success


def motor_model_function(grid, approx_model, param):
    # -----------------------------------------------------------------
    #  This function is a model of the "real" motor (MOTOR).
    #  N - number of data points
    #  M - number of parameters
    # -----------------------------------------------------------------
    #  input:
    #   grid           [Nx3]     Matrix of data points [u,q,dq], where
    #                            the function has to be evaluated.
    #   approx_model   {dict}    Model specification.
    #                            Check the exercise description for
    #                            more information.
    #   param          [Mx1]     A vector of parameters that define
    #                            this function.
    # -----------------------------------------------------------------
    #  return:
    #   u              [Nx1]     The result, i.e. this function (specified
    #                            by model and parameters) evaluated at the
    #                            grid points.
    #   dudp           [NxM]     The derivative of this function at all
    #                            grid points w.r.t. the parameters.
    # -----------------------------------------------------------------
    # assign motor variables
    u_soll = grid[:, [0]]
    q = grid[:, [1]]
    dq = grid[:, [2]]

    # recall input function
    mu1 = approx_model['mu1']
    mu2 = approx_model['mu2']
    n1 = approx_model['n1']
    n2 = approx_model['n2']
    n = n1 + n2

    # compute u and ud
    u1, ud_p1 = mu1(q, param[0:n1], n1 - 1)
    u2, ud_p2 = mu2(dq, param[n1:n], n2 - 1)

    u_ist = u1 * u_soll + u2 * u_soll

    dudp = np.mat(u_soll).reshape((1, -1)) * np.hstack((ud_p1, ud_p2))

    return u_ist, dudp


def polynom_n(x, param, n):
    # -----------------------------------------------------------------
    #  Polynomial of the order n
    #  N - number of data points
    #  M - number of parameters
    # -----------------------------------------------------------------
    #  input:
    #   x              [Nx1]     Input Vector
    #   param          [Mx1]     A vector of parameters that define
    #                            the function, check the exercise
    #                            description for more information.
    #   n              int       The order of the polynomial
    # -----------------------------------------------------------------
    #  return:
    #   fx             [Nx1]     Output Vector of the polynomial
    #   dfdp           [NxM]     Jacobian Matrix of df/fp
    # -----------------------------------------------------------------
    # initialise
    fx = np.zeros(np.shape(x))
    dfdp = np.zeros((x.shape[0], param.shape[0]))

    # assign polynom substitution
    for i in range(n + 1):
        x_i = np.power(x, i)
        fx += param[i] * x_i
        dfdp[:, i] = x_i.reshape((-1))

    return fx, dfdp


def modelbased_friction_compensation(approx_model, param, u, q, dq):
    # -----------------------------------------------------------------
    #  This function gets a desired force u and computes some force
    #  uCorrection, such that MOTOR(uCorrection) = u.
    # -----------------------------------------------------------------
    #  input:
    #   approx_model   {dict}    Model specification.
    #                            Check the exercise description for
    #                            more information.
    #   param          [mx1]     A vector of parameters that define
    #                            this function.
    #   u              [3x1]     Some desired force u
    #   q              [3x1]     The current joint position.
    #   dq             [3x1]     The current joint velocity.
    # -----------------------------------------------------------------
    #  return:
    #   u              [3x1]     some output force
    # -----------------------------------------------------------------
    grid = []
    for i in range(3):
        grid.append([u[i], q[i], dq[i]])

    u_ist = motor_model_function(grid, approx_model, param)
    u = u / u_ist * u
    return u


def plot_model_differences(time_scale, Q1, Q2, Q3):
    # -----------------------------------------------------------------
    #  This method plots the data collected in runP5.
    # -----------------------------------------------------------------
    #  input:
    #   time_scale    [N]             Time scale (x values)
    #   Q1            [Nx3]           Joint values for first simulation
    #   Q2            [Nx3]           Joint values for second simulation
    #   Q3            [Nx3]           Joint values for third simulation
    # -----------------------------------------------------------------
    fig, axs = plt.subplots(3, 1)

    namespace = ["F1", "F2", "F3"]
    for i in range(3):
        axs[i].plot(time_scale, Q1[:, i], label="first simulation")
        axs[i].plot(time_scale, Q2[:, i], label="second simulation")
        axs[i].plot(time_scale, Q3[:, i], label="third simulation")
        axs[i].set_xlabel('time')
        axs[0].set_ylabel(namespace[i])
        axs[0].grid(True)
    plt.show()

    pass


if __name__ == "__main__":
    # test for task 5
    mfunc = motor_model_function
    p0 = np.array([0, 1, 3, 2, -1]).reshape(-1, 1)
    xres = np.array([3.5, 5.0, 3.5, -1]).reshape(-1, 1)
    grid = np.array([
        [1, 0.5, 0],
        [1, 1, 1],
        [1, -0.25, 1.2],
        [1, -1, -1]
    ])
    model = {
        'mu1': polynom_n,
        'mu2': polynom_n,
        'n1': 2,
        'n2': 3,
    }
    # task 1
    _, _, p_opt, _, success = nonlinear_least_squares(
        mfunc, p0, grid, xres, model)

    p_expected = np.array(
        [-0.04, 1.17, 2.96, 1.83, -0.92]).reshape(-1, 1)

    print(f"error in p_opt: {np.linalg.norm(p_opt - p_expected)}")

    # task 2
    model = {
        'mu1': polynom_n,
        'mu2': polynom_n,
        'n1': 1,
        'n2': 3,
    }

    p = np.array([0, 1, 3, 2]).reshape(-1, 1)
    x = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 0, 1.2],
        [1, 0, -1]
    ])

    u1, du1 = motor_model_function(x, model, p)
    u_expected = np.array([1, 6, 7.48, 0]).reshape(-1, 1)
    print(f"error in u1 :{np.linalg.norm(u_expected - u1)}")

    model['n1'] = 2
    p = np.array([0, 1, 3, 2, -1]).reshape(-1, 1)
    x = np.array([
        [1, 0.5, 0],
        [1, 1, 1],
        [1, -0.25, 1.2],
        [1, -1, -1]
    ])
    u2, du2 = motor_model_function(x, model, p)
    u_expected = np.array([3.5, 5.0, 3.71, -1]).reshape(-1, 1)
    print(f"error in u2 :{np.linalg.norm(u_expected - u2)}")






