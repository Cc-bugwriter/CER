import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


from tasks.task2 import newton


def expl_euler(f_dyn, x_k, dt):
    """
    ------------------------------------------------------------------------
            Compute a simulation step using the Explicit Euler Method
    ------------------------------------------------------------------------
    n: dimensions of the 3D-printer
    Inputs
        f_dyn   f_handle             function handle that returns the time
                                     derivative of a given state at a given time
                                     np.ndarray, np.float64, shape (n, 1) ->
                                     np.ndarray, np.float64, shape (n, 1)
        x_k     np.ndarray((n, 1))   the current state
        dt      float                the time step
    ------------------------------------------------------------------------
    Outputs
        x_k_1   np.ndarray((n, 1))   the next state x_{k+1}
    ------------------------------------------------------------------------
    """
    # x_k+1 = x_k + h*f(x_k)
    # guarantee correct size
    x_k = x_k.reshape((-1, 1))

    # assign intermediate variables (derivative)
    dx_k = f_dyn(x_k)

    # implement explicit Euler Method
    x_k_1 = x_k + dt * dx_k

    return x_k_1


def impl_euler(f_dyn, x_k, dt):
    """
    ------------------------------------------------------------------------
            Compute a simulation step using the implicit Euler method
    ------------------------------------------------------------------------
    n: dimensions of the 3D-printer
    Inputs
        f_dyn   f_handle             function handle that returns the time
                                     derivative of a given state at a given time
                                     np.ndarray, np.float64, shape (n, 1) ->
                                     np.ndarray, np.float64, shape (n, 1)
        x_k     np.ndarray((n, 1))   the current state
        dt      float                the time step
    ------------------------------------------------------------------------
    Outputs
        x_k_1   np.ndarray((n, 1))   the next state x_{k+1}
    ------------------------------------------------------------------------
    """
    import scipy.optimize
    # x_k+1 = x_k + h*f(x_k+1)
    # --> F(x_k+1) = 'x_k+1 - (x_k+1 = x_k + h*f(x_k+1)) == 0'
    # guarantee correct size
    x_k = x_k.reshape((-1, 1))

    # implement implicit Euler method
    def F_impl(x_k_):
        x_k_ = x_k_.reshape((-1, 1))
        return (x_k_ - x_k - dt*f_dyn(x_k_)).flatten()

    # use scipy.opimize to calculate
    x_k_1 = scipy.optimize.fsolve(F_impl, (x_k + dt * f_dyn(x_k)).flatten())

    return np.array(x_k_1).reshape((-1, 1))


def heun(f_dyn, x_k, dt):
    """
    ------------------------------------------------------------------------
            Compute a simulation step using Heun's method
    ------------------------------------------------------------------------
    n: dimensions of the 3D-printer
    Inputs
        f_dyn   f_handle             function handle that returns the time
                                     derivative of a given state at a given time
                                     np.ndarray, np.float64, shape (n, 1) ->
                                     np.ndarray, np.float64, shape (n, 1)
        x_k     np.ndarray((n, 1))   the current state
        dt      float                the time step
    ------------------------------------------------------------------------
    Outputs
        x_k_1   np.ndarray((n, 1))   the next state x_{k+1}
    ------------------------------------------------------------------------
    """
    # s1 = f(x_k) s2 = f(x_k + h*s1) x_k+1 = x_k + h/2(s1 + s2)
    # guarantee correct size
    x_k = x_k.reshape((-1, 1))

    # assign intermediate variables (derivative)
    s_1 = f_dyn(x_k)
    s_2 = f_dyn(x_k + dt*s_1)

    # implement Heun's method
    x_k_1 = x_k + dt/2 * (s_1 + s_2)

    return x_k_1


def rk4(f_dyn, x_k, dt):
    """
    ------------------------------------------------------------------------
     Compute a simulation step using the Runge-Kutta method of order 4
    ------------------------------------------------------------------------
    n: dimensions of the 3D-printer
    Inputs
        f_dyn   f_handle             function handle that returns the time
                                     derivative of a given state at a given time
                                     np.ndarray, np.float64, shape (n, 1) ->
                                     np.ndarray, np.float64, shape (n, 1)
        x_k     np.ndarray((n, 1))   the current state
        dt      float                the time step
    ------------------------------------------------------------------------
    Outputs
        x_k_1   np.ndarray((n, 1))   the next state x_{k+1}
    ------------------------------------------------------------------------
    """
    # s1 = f(x_k) s2 = f(x_k + h/2*s1)
    # s3 = f(x_k + h/2*s2) s4 = f(x_k + h*s3)
    # x_k+1 = x_k + h/6(s1 + 2*s2 + 2*s3 + s4)

    # guarantee correct size
    x_k = x_k.reshape((-1, 1))

    # assign intermediate variables (derivative)
    s_1 = f_dyn(x_k)
    s_2 = f_dyn(x_k + dt/2 * s_1)
    s_3 = f_dyn(x_k + dt/2 * s_2)
    s_4 = f_dyn(x_k + dt * s_3)

    # implement Heun's method
    x_k_1 = x_k + dt / 6 * (s_1 + 2*s_2 + 2*s_3 + s_4)

    return x_k_1


def plot_solver_data(computation_times, actual_trajectories, solver, tmax=12, dt=0.01):
    """
    ------------------------------------------------------------------------
                        Plot Recorded Data from Solvers
    ------------------------------------------------------------------------
    Inputs
        computation_times       pandas.DataFrame    Dictionary like data, stored the
                                                    calculation time of different method
                                                    with different time step size.

                                time_step index (float): 0.05, 0.01, 0.005
                                method index (object): 'VREP', 'ImplEuler', 'ExplEuler', 'Heun', 'RK'
                                The value or array can be called by:
                                computation_times[time_step]            e.g. computation_times[0.05]
                                computation_times[time_step][method]    e.g. computation_times[0.05]['RK'] -> float
        --------------------------------------------------------------------
        actual_trajectories     pandas.DataFrame    Dictionary like data, stored the
                                                    calculation time of different method
                                                    with different time step size.

                                time_step index (float): 0.05, 0.01, 0.005
                                method index (object): 'VREP', 'ImplEuler', 'ExplEuler', 'Heun', 'RK'
                                The value or array can be called by:
                                actual_trajectories[time_step]            e.g. actual_trajectories[0.05]
                                actual_trajectories[time_step][method]    e.g. actual_trajectories[0.05]['RK'] ->
                                                                                2D-list: [d, N]
                                                                                d: dimensions of the 3D-printer
                                                                                N: numbers of time steps
        --------------------------------------------------------------------
        solver                  method              ODE solver for computing exact solution
    ------------------------------------------------------------------------
    """
    time_serires = np.arange(0, tmax / dt + 1) / (1 / dt)
    time_serires_all = np.vstack((time_serires, time_serires, time_serires)).reshape((3, -1))
    q = solver(time_serires_all)

    # plot 1
    fig_1, axs_1 = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    axs_1[0].set_title("Position der drei Gelenke in Gelenkkoordinaten")
    axs_1[0].set_xlabel('time in s')
    axs_1[0].set_ylabel('position in m')
    axs_1[0].plot(time_serires, q[0, :], label="joint 1")
    axs_1[0].plot(time_serires, q[1, :], label="joint 2")
    axs_1[0].plot(time_serires, q[2, :], label="joint 3")
    axs_1[0].legend(bbox_to_anchor=(0.5, -0.15), loc='lower center')

    axs_1[1] = computation_times.T.plot.bar()

    plt.show()

    pass


if __name__ == '__main__':
    # # Differential equations with start value
    # p1 = lambda x: 0.5 * x ** 2 + 2 * x - 3
    #
    # # Aufgaba 1.1
    # x1 = np.ones((1, 1))
    # x1_hist = x1.copy()
    # for i in range(3):
    #     x1 = expl_euler(p1, x1, 1)
    #     x1_hist = np.append(x1_hist, x1)
    #
    # x1_expected = np.array([1, 0.5, -1.375, -6.179687])
    # print(f'explizite Euler-Verfahren error : {np.linalg.norm(x1_hist - x1_expected)}')
    #
    # # Aufgabe 1.2
    # x1 = np.ones((1, 1))
    # x1_hist = x1.copy()
    # for i in range(3):
    #     x1 = impl_euler(p1, x1, 1)
    #     x1_hist = np.append(x1_hist, x1)
    #
    # x1_expected = np.array([1.0, 1.23606798, 1.12787783, 1.17812863])
    # print(f'implizite Euler-Verfahren error : {np.linalg.norm(x1_hist - x1_expected)}')
    #
    # # Aufgabe 1.3
    # x1 = np.ones((1, 1))
    # x1_hist = x1.copy()
    # for i in range(3):
    #     x1 = heun(p1, x1, 1)
    #     x1_hist = np.append(x1_hist, x1)
    # x1_expected = np.array([1.0, -0.1875, -3.76951503, -1.21651473])
    # print(f'Heun-Verfahren error : {np.linalg.norm(x1_hist - x1_expected)}')
    #
    # # Aufgabe 1.4
    # x1 = np.ones((1, 1))
    # x1_hist = x1.copy()
    # for i in range(3):
    #     x1 = rk4(p1, x1, 1)
    #     x1_hist = np.append(x1_hist, x1)
    # x1_expected = np.array([1.0, -0.97578688, -4.49632406, -3.86973236])
    # print(f'Runge-Kutta-Verfahren vierter Ordnung. error : {np.linalg.norm(x1_hist - x1_expected)}')

    # Aufgabe 2
    numberOfJoints = 3
    dt = 0.01
    modelledPartialMasses = np.array([2.5, 2.5, 0.25, 0.15])
    modelledJointDamping = np.array([3.0, 3.0, 3.0])
    gravityConstant = 9.81
    jointLimits_raw = np.array([-0.002, 0.0, 0.0, 0.4, 0.4, 0.4])
