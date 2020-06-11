import numpy as np
from tast.task2 import inverse_dynamics


def compute_torques(ctrl_params, q, q_desired, dq, dq_desired):
    """
    PID controller.
    Computes control torques required for given joint values,
    joint targets, joint velocities and joint target velocities.

    xdim: dimensions of the 3D-printer
    N: length of trajectory

    Controller variables:
        Flags:
        ctrl_params.use_invdyn: bool -> use inverse dynamics in PD controller, for tasks b) set True.

        Parameters:
        ctrl_params.kp: np.ndarray, np.float64, shape (xdim, 1) -> proportional gain
        ctrl_params.ki: np.ndarray, np.float64, shape (xdim, 1) -> integral gain
        ctrl_params.kd: np.ndarray, np.float64, shape (xdim, 1) -> differential gain
        ctrl_params.partial_masses: modelled partial masses
        ctrl_params.joint_damping: np.ndarray, np.float64, shape (xdim,) -> modelled joint damping
        ctrl_params.gravity_constant: float64, gravity acceleration g = 9.81
        ctrl_params.dt: float64, time step size

        Persistence/storage:
        ctrl_params.ctrl_err_int: integral error
        ctrl_params.ctrl_err_past: previous proportional error

    Inputs
        q: np.ndarray, np.float64, shape (xdim, N) -> current joint position
        q_desired: np.ndarray, np.float64, shape (xdim, N) -> desired joint position
        dq: np.ndarray, np.float64, shape (xdim, N) -> current joint velocity
        dq_desired: np.ndarray, np.float64, shape (xdim, N) -> desired joint velocity
    Outputs
        u: np.ndarray, np.float64, shape (xdim, N) -> current joint position

    """
    try:
        N = q.shape[1]
        shape = q.shape
    except IndexError:
        N = 1
        shape = q.shape[0]

    q = q.reshape(-1, N)
    dq = dq.reshape(-1, N)
    q_desired = q_desired.reshape(-1, N)
    dq_desired = dq_desired.reshape(-1, N)

    # assign intermediate variables
    delta_q = q_desired - q
    delta_dq = dq_desired - dq

    # initial output variable
    u = np.empty(shape).reshape(-1, N)

    # PID controller
    for i in range(N):
        ctrl_params.ctrl_err_past = delta_q[:, i].reshape(-1, 1)
        ctrl_params.ctrl_err_int += ctrl_params.dt / 2 * delta_q[:, i].reshape(-1, 1)

        u[0, i] = ctrl_params.ctrl_err_past[0] * ctrl_params.kp[0] + ctrl_params.ctrl_err_int[0] * ctrl_params.ki[
            0] + \
                  delta_dq[0] * ctrl_params.kd[0]
        u[1, i] = ctrl_params.ctrl_err_past[1] * ctrl_params.kp[1] + ctrl_params.ctrl_err_int[1] * ctrl_params.ki[
            1] + \
                  delta_dq[1] * ctrl_params.kd[1]
        u[2, i] = ctrl_params.ctrl_err_past[2] * ctrl_params.kp[2] + ctrl_params.ctrl_err_int[2] * ctrl_params.ki[
            2] + \
                  delta_dq[2] * ctrl_params.kd[2]

    if ctrl_params.use_invdyn:
        # add compensation of gravitation
        for i in range(N):
            gra_com = inverse_dynamics(ddq=np.zeros((3, 1)),
                                       dq=np.zeros((3, 1)),
                                       link_masses=ctrl_params.partial_masses,
                                       joint_damping=ctrl_params.joint_damping,
                                       gravity_acc=ctrl_params.gravity_constant
                                       ).flatten()
            u[:, i] += gra_com

    if N == 1:
        u = u.flatten()

    return u


def setup_coffecient():
    """
    PID controller parameters, used for tasks a) and b)

    Output
        kp: np.ndarray, np.float64, shape (xdim, 1) -> proportional gain
        ki: np.ndarray, np.float64, shape (xdim, 1) -> integral gain
        kd: np.ndarray, np.float64, shape (xdim, 1) -> derivative gain
    """

    # bad initial value, Please try to find a good value
    kp = np.array([1000, 70, 60]).reshape(-1, 1)
    ki = np.array([1000, 5, 1.5]).reshape(-1, 1)
    kd = np.array([90, 9, 2.5]).reshape(-1, 1)
    return kp, ki, kd


def trajectory_generation(q_final, joint_num, vmax, amax, dt):
    """
    Trapezoid-Velocity Trajectory Generator
    Input
        q_final: float -> joint final position
        joint_num: int -> joint index
        vmax: np.ndarray, np.float64, shape (xdim, ) -> maximal velocity
        amax: np.ndarray, np.float64, shape (xdim, ) -> maximal acceleration
        dt: float -> time step size

    Output
        q: np.ndarray, np.float64, shape (N_steps, ) -> position of desired trajectory
        qd: np.ndarray, np.float64, shape (N_steps, ) -> velocity of desired trajectory
        times: np.ndarray, np.float64, shape (N_steps, ) -> time of desired trajectory
    """
    # compute intermediate variables
    t1 = vmax/amax
    q_t1 = 1/2 * amax * t1**2
    delta_t2 = (q_final - 2*q_t1)/vmax
    t2 = t1 + delta_t2
    t_all = t1 + t2

    # compute N_steps
    N_steps = t_all/dt +1

    # assign time series
    times = np.linspace(0, t_all[joint_num], num=int(N_steps[joint_num]))

    # initial trajectory position and velocity
    q = np.empty(times.shape)
    qd = np.empty(times.shape)

    for i, time in enumerate(times):
        if time <= t1[joint_num]:
            # position
            q[i] = 1/2 * amax[joint_num] * time**2
            q_t1 = q[i]
            # velocity
            qd[i] = amax[joint_num] * time
        elif time <= t2[joint_num]:
            # position
            q[i] = vmax[joint_num] * (time - t1[joint_num]) + q_t1
            q_t2 = q[i]
            # velocity
            qd[i] = vmax[joint_num]
        else:
            # position
            q[i] = q_t2 + vmax[joint_num] * (time - t2[joint_num]) - 1/2 * amax[joint_num] * (time - t2[joint_num])**2
            # velocity
            qd[i] = vmax[joint_num] - amax[joint_num] * (time - t2[joint_num])

    return q, qd, times


if __name__ == '__main__':
    class Ctrl_params():
        def __init__(self, invdyn=False):
            num_joints = 3
            self.use_invdyn = invdyn
            self.kp = np.array([1000, 70, 60]).reshape(-1, 1)
            self.ki = np.array([1000, 5, 1.5]).reshape(-1, 1)
            self.kd = np.array([90, 9, 2.5]).reshape(-1, 1)
            self.partial_masses = np.array([2.5, 2.5, 0.25, 0.15])
            self.joint_damping = np.array([3.0, 3.0, 3.0])
            self.gravity_constant = 9.81
            self.dt = 0.01
            self.ctrl_err_int = np.zeros((num_joints, 1))
            self.ctrl_err_past = np.zeros((num_joints, 1))

    ctrl_params = Ctrl_params()
    q = np.ones(3) * 0.02
    q_desired = np.ones(3) * 0.03
    dq = np.ones(3) * 0.01
    dq_desired = np.ones(3) * 0.03

    # tasks 1
    u_expected = np.array([11.85000, 0.88025, 0.650075])
    u = compute_torques(ctrl_params, q, q_desired, dq, dq_desired)
    print(f"test1 u error : {np.linalg.norm(u_expected - u)}")

    # tasks 2
    ctrl_params = Ctrl_params(invdyn=True)
    u_expected = np.array([40.29900, 0.88025, 0.650075])
    u = compute_torques(ctrl_params, q, q_desired, dq, dq_desired)
    print(f"test2 u error : {np.linalg.norm(u_expected - u)}")

    # tasks 3
    joint_num = 0
    q_final = 30e-2
    vmax = np.array([0.06, 0.06, 0.06])
    amax = np.array([0.1, 0.1, 0.1])
    dt = 0.01
    q_target, qd_target, time = trajectory_generation(q_final, joint_num, vmax, amax, dt)

    test_ind = [0, 9, 49]
    # test q
    q_target_expected = np.array([0, 4.0500e-04, 0.012005])
    print(f"test3 q error : {np.linalg.norm(q_target_expected - q_target[test_ind])}")
    # test dq
    qd_target_expected = np.array([0, 0.009, 0.049])
    print(f"test3 dq error : {np.linalg.norm(qd_target_expected - qd_target[test_ind])}")





