import numpy as np

def inverse_dynamics(ddq, dq, link_masses, joint_damping, gravity_acc):
    """
    Computes the forces for each joint, given the trajectory of joint velocities and accelerations.

    The columns of ddq and dq are the desired joint accelerations and velocities, for each step of the trajectory.
    E.g., if ddq.shape() = (3, 10), we have a trajectory of 10 steps, where for each step we want to achieve the
    given accelerations for the 3 joints.

    When working with numpy arrays, make use of broadcasting instead of using a for loop. E.g, it is possible to
    sum two arrays with different dimensions.
    a = np.array([1., 2.]), b = np.array([[1., 1.], [1., 1.]]), (a+b).shape = (2,2)
    As a tip, first construct the loop version and only then move to numpy broadcasting.

    The general equation for a robot's dynamics is given by
    f = M(q) @ ddq + C(q, dq) + G(q),
    where q are the generalized joint coordinates, dq the joint velocities,
    ddq the joint accelerations, M mass matrix, C the Centrifugal and Coriolis forces,
    G(q) the gravity force.

    n_links: number of links
    n_joints: number of joints
    N: length of the trajectory

    inputs
        ddq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint accelerations
        dq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint velocities
        link_masses: np.ndarray, np.float64, shape (n_links,) -> mass of each link
        joint_damping: np.ndarray, np.float64, shape (n_joints,) -> damping coefficient for each joint
        gravity_acc: float -> gravity acceleration constant
    returns
        force: np.float64, shape (n_joints, N) -> trajectory of forces to apply to each joint

    """
    # Safe way to convert to numpy arrays
    ddq = np.asarray(ddq)
    dq = np.asarray(dq)

    n_links = link_masses.shape[0]
    n_joints = dq.shape[0]
    N = dq.shape[1]

    # Mass matrix
    M = np.array([
        [sum(link_masses[1:4]), 0, 0],
        [0, sum(link_masses[2:4]), 0],
        [0, 0, link_masses[3]]
    ])

    # Damping matrix
    C = np.array([
        [joint_damping[0], 0, 0],
        [0, joint_damping[1], 0],
        [0, 0, joint_damping[2]]
    ])

    # Gravity
    G = np.array([gravity_acc, 0, 0]).reshape(n_joints, 1)

    # Compute the force
    force = np.dot(M, ddq+G) + np.dot(C, dq)

    return force


def cyclic_analytical(sim_time_vect, amplitude, omega):
    """
    Computes the analytical velocity and acceleration trajectories for each joint, for the given vector
    of simulation times sim_time_vect. Each entry of the joint vector, as a function of time, is given by

    qi(t) = a * (1 - cos(omega*t))

    N: length of the trajectory == len(sim_time_vect)
    n_joints: number of joints

    inputs
        sim_time_vect: np.ndarray, np.float64, shape (N,) -> vector with simulation times
        amplitude: int or float -> joint amplitude
        omega: int or float -> angular frequency
    returns
        dq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint velocities
        ddq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint accelerations

    """
    n_joints = 3

    dq = np.array([amplitude*omega*np.sin(omega*sim_time_vect)]).repeat(n_joints, axis=0)
    ddq = np.array([amplitude*(omega**2)*np.cos(omega*sim_time_vect)]).repeat(n_joints, axis=0)

    return dq, ddq


def cyclic_numerical(q, dt):
    """
    Computes the numerical approximation of the velocity and acceleration trajectories for each joint,
    using backward difference quotients (backward differences).

    N: length of the trajectory
    n_joints: number of joints

    inputs
        q: np.ndarray, np.float64, shape (n_joints, N) -> joint trajectory
        dt: float -> time step for numerical derivative computations
    returns
        dq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint velocities
        ddq: np.ndarray, np.float64, shape (n_joints, N) -> trajectory of joint accelerations

    """
    # call back backward difference
    dq = backward_dq(q, dt)
    ddq = backward_dq(dq, dt)

    return dq, ddq


def backward_dq(X, dt):
    """
    Computes backward difference quotient for the time series in X.
    https://de.wikipedia.org/wiki/Differenzenquotient#Rückwärtsdifferenzenquotient

    For time step 0, the backward difference quotient is 0 for all dimensions.

    xdim: X.shape[0]
    N: X.shape[1], length of time series

    inputs
        X: np.ndarray, np.float64, shape (xdim, N) -> time sequence of values
        dt: float -> time step
    returns
        dx: np.ndarray, np.float64, shape (xdim, N) -> backward difference quotient

    """
    # initial list of difference quotient
    dx_list = []

    # convert joint index to trajectory index
    x = X.T  # (N, n_joints)

    # append speed value of each time point
    for i in range(x.shape[0]):
        if i == 0:
            dx_list.append(np.zeros(x.shape[1]).tolist())
        else:
            dx_list.append(((x[i] - x[i - 1]) / dt).tolist())
    # convert list to array
    dx = np.array(dx_list)  # (N, n_joints)

    # convert back to joint index
    return dx.T


def newton(x0, iters, eps, f, df):
    """
    Computes a root of the function f using Newton's method.
    https://en.wikipedia.org/wiki/Newton%27s_method

    Stop the iterative procedure if:
        - The euclidean norm between consecutive points is less than eps
        - The number of iterations (iters) was reached

    xdim: x0.shape[0]

    inputs
        x0: np.ndarray, np.float64, shape (xdim, 1) -> starting vector
        iters: int -> maximum number of iterations
        eps: float -> tolerance
        f: callable function -> f:R^xdim -> R^xdim. f(x0) gives f at x0
        df: callable function -> df:R^xdim -> R^(xdim, xdim). df(x0) computes the jacobian of f at x0
    returns
        x1: np.ndarray, np.float64, shape (xdim, 1) -> the root of f

    """
    x0 = x0.copy()

    # initial list of iteration result
    x_iter = [x0]
    for i in range(iters):
        # newton method
        x_newton = x_iter[0] - np.dot(np.linalg.inv(df(x_iter[0])), f(x_iter[0]))
        # assign in list
        x_iter.append(x_newton)

        if np.linalg.norm(x_iter[1] - x_iter[0]) < eps:
            x_iter[0] = x_iter[1]
            break
        else:
            x_iter[0] = x_iter[1]
            # save memory
            del x_iter[1:]

    x1 = x_iter[0]

    return x1


def optimize_vmax(q_target, beta, vmax0, iters, eps, link_masses, joint_damping, gravity_acc):
    """
    Computes the joint velocities that maximize the specified cost function.

    n_joints: number of joints

    inputs
        q_target: np.ndarray, np.float64, shape (n_joints, 1) -> joint target
        beta: np.ndarray, np.float64, shape (n_joints, 1) -> weighting parameter
        vmax0: np.ndarray, np.float64, shape (n_joints, 1) -> initial guess for the joint velocity
        iters: int -> maximum number of iterations for the optimization routine
        eps: float -> tolerance for the optimization routine
        link_masses: np.ndarray, np.float64, shape (n_links,) -> mass of each link
        joint_damping: np.ndarray, np.float64, shape (n_joints,) -> damping coefficient for each joint
        gravity_acc: float -> gravity acceleration constant
    returns
        vmax_opt: np.ndarray, np.float64, shape (n_joints, 1) -> the joint velocity that maximizes the cost function

    """
    def dJ(x):
        """
        Jacobian Matrix of the scalar function J, TRANSPOSED.
        The Jacobian of a scalar function is by definition a row vector, but here we return it as a column vector.

        inputs
            x: np.ndarray, np.float64, shape (n_joints, 1)
        returns
            jac: np.ndarray, np.float64, shape (n_joints, 1) -> the Jacobian (TRANSPOSED) of J at x

        """
        # initial Jacobian list
        j_list = []

        # assign dimension
        n_joints = x.shape[0]

        # compute intermediate variable
        force = inverse_dynamics(np.zeros((n_joints, 1)), x, link_masses, joint_damping, gravity_acc)
        # compute Jacobian element
        for i in range(n_joints):
            j_element = 2 * force[i] * (q_target[i] ** 2) * joint_damping[i] - beta[i] / q_target[i]
            j_list.append(float(j_element))

        # convert to array, shape (n_joints, 1)
        jac = np.array(j_list).reshape((n_joints, 1))
        return jac

    def ddJ(x):
        """
        Hessian Matrix of the scalar function J.

        inputs
            x: np.ndarray, np.float64, shape (n_joints, 1)
        returns
            hess: np.ndarray, np.float64, shape (n_joints, n_joints) -> the Hessian of J at x

        """
        # initial Jacobian list
        hess_list = []

        # assign dimension
        n_joints = x.shape[0]

        # compute intermediate variable
        force = inverse_dynamics(np.zeros((n_joints, 1)), x, link_masses, joint_damping, gravity_acc)

        # build a diagonal Hessian matrix
        for i in range(n_joints):
            hess_row = []
            for j in range(n_joints):
                if i == j:
                    hess_element = 2 * q_target[i] ** 2 * joint_damping[i] ** 2
                    hess_row.append(float(hess_element))
                else:
                    hess_row.append(0.)
            hess_list.append(hess_row)

        # convert array, shape (n_joints, n_joints)
        hess = np.array(hess_list)
        return hess

    # Find the optimal velocity with Newton's method
    vmax_opt = newton(vmax0, iters, eps, dJ, ddJ)

    return vmax_opt


if __name__ == '__main__':
    # tasks 1
    link_masses = np.array([2.5, 2.5, 0.25, 0.15])
    joint_damping = np.array([3.0, 3.0, 3.0])
    gravity_acc = 9.81

    # tasks 1.1
    ddq = np.array([[0., 1.], [0., 1.], [0., 1.]])
    dq = np.array([[0., 0.5], [0., 0.5], [0., 0.5]])
    force_1 = inverse_dynamics(ddq, dq, link_masses, joint_damping, 0)
    force_expected = np.array([[0., 4.4], [0., 1.9], [0., 1.65]])
    print(f"difference between expect and force 1: {np.linalg.norm(force_expected - force_1)}")

    # tasks 1.2
    ddq = np.array([[0., 1.], [0., 0.], [0., 0.]])
    dq = np.array([[0., 0.5], [0., 0.0], [0., 0.0]])
    force_2 = inverse_dynamics(ddq, dq, link_masses, joint_damping, gravity_acc)
    force_expected = np.array([[28.449, 32.849], [0., 0.0], [0., 0.0]])
    print(f"difference between expect and force 2: {np.linalg.norm(force_expected - force_2)}")

    # tasks 2
    t = np.array([0, 0.1, 0.2])
    q = np.array([np.square(t), 0 * t, 0 * t])
    amplitude = 1
    omega = 1
    dt = 0.01

    # tasks 2.1
    test_dq_1, test_ddq_1 = cyclic_analytical(t, amplitude, omega)

    # tasks 2.2
    dq_expected = np.array([[0, 1, 3], [0, 0, 0], [0, 0, 0]])
    test_dq_2, test_ddq_2 = cyclic_numerical(q, dt=dt)
    print(f"difference between expect and dq: {np.linalg.norm(dq_expected - test_dq_2)}")

    # tasks 3
    # tasks 3.1
    def f(x):
        x = x.reshape(-1)
        return np.array([
            0.1 * x[1] - 0.5 * x[0],
            x[0] - 0.5 * x[1] - 0.1,
            -0.5 * x[2] + x[0] ** 2
        ]).reshape((-1, 1))


    def df(x):
        x = x.reshape(-1)
        return np.array([
            [-0.5, 0.1, 0],
            [1, -0.5, 0],
            [2 * x[0], 0, -0.5]
        ])

    x0 = np.array([0.1, 0.1, 0.1]).reshape((-1, 1))
    xs = newton(x0, 1000, 1e-8, f, df)
    xs_expected = np.array([-1 / 15, -1 / 3, 2 / 225]).reshape((-1, 1))
    print(f"difference between expect and newton methode: {np.linalg.norm(xs_expected - xs)}")

    # tasks 3.1
    q_target = np.array([0.2, 0.4, 0.2]).reshape((-1, 1))
    beta = np.array([0.1, 0.1, 0.1]).reshape((-1, 1))
    vmax0 = np.array([0.1, 0.1, 0.1]).reshape((-1, 1))
    iterations = 1000
    epsilon = 1e-8
    gravity_acc = 0
    vmax_opt = optimize_vmax(q_target, beta, vmax0, iterations, epsilon, link_masses, joint_damping, gravity_acc)
    vmax_opt_expected = np.array([0.6944444444444443, 0.0868055555555555, 0.6944444444444443]).reshape((-1, 1))
    print(f"difference between expect and v max opt: {np.linalg.norm(vmax_opt_expected - vmax_opt)}")