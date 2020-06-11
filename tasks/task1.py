import numpy as np


def dictionary(letter):
    """
    For given letter (C, E, R), creates a matrix (path) of via points (x,y,z) and a vector (extrusion) of extrusions
    between consecutive via points. Extrusion is a vector of boolean values, where 1 (True) means extrude,
    and 0 (False) not extrude.

    inputs
        letter: char
    returns
        path: numpy array, float, shape (N_via_points, 3)
        extrusion: numpy array, boolean, shape (N_via_points)
    """
    if letter == 'C':
        path = np.array([[0, 0, 1], [0, 0, 0], [0, 2, 0], [1, 2, 0], [1, 1.8, 0],
                         [.2, 1.8, 0], [.2, .2, 0], [1, .2, 0], [1, 0, 0], [0, 0, 0],
                         [0, 0, 1]])
        extrusion = np.array([0, 1, 1, 1, 1,
                              1, 1, 1, 1, 0,
                              0])
    elif letter == 'E':
        path = np.array([[0, 0, 1], [0, 0, 0], [0, 2, 0], [1, 2, 0], [1, 1.8, 0],
                        [.2, 1.8, 0], [.2, 1.1, 0], [1, 1.1, 0], [1, .9, 0], [.2, .9, 0],
                        [.2, .2, 0], [1, .2, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
        extrusion = np.array([0, 1, 1, 1, 1,
                              1, 1, 1, 1, 1,
                              1, 1, 1, 0, 0])
    elif letter == 'R':
        path = np.array([[0, 0, 1], [0, 0, 0], [0, 2, 0], [1, 2, 0], [1, .9, 0],
                         [.6, .9, 0], [1, 0, 0], [.8, 0, 0], [.4, .9, 0], [.4, 1.1, 0],
                         [.8, 1.1, 0], [.8, 1.8, 0], [.8, 1.8, 0], [.2, 1.8, 0], [.2, 0, 0],
                         [0, 0, 0], [0, 0, 1]])
        extrusion = np.array([0, 1, 1, 1, 1,
                              1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1,
                              0, 0])

    return path, extrusion


def compute_forward_kinematic(q, kinematic_params, joint_limits):
    """
    Computes the end-effector coordinates (x,y,z) with respect to the frame Sp, for the given joint displacements q.
    q is a vector with the joint displacements.
    kinematic_params is a vector with the DH-Parameters kinematic_params = (la, lb, lc, ld, oX, oY, oZ).
    joint_limits is an array with the minimum and maximum joint values for each joint in q.

    inputs
        q: numpy array, float, shape (3)
        kinematic_params: numpy array, float, shape (7)
        joint_limits: numpy array, float, shape (3, 2)
    returns
        position: numpy array, float, shape (3)
    """
    # Check if requested joint configuration does not violate joint limits.
    # If limits are violated return an empty array np.array([])
    epsilon = 1e-6

    if q[0] < joint_limits[0, 0]-epsilon or q[0] > joint_limits[0, 1]+epsilon\
            or q[1] < joint_limits[1, 0]-epsilon or q[1] > joint_limits[1, 1]+epsilon\
            or q[2] < joint_limits[2, 0]-epsilon or q[2] > joint_limits[2, 1]+epsilon:
        position = np.array([])
        return position

    # Compute analytical forward kinematics
    T_01 = trans_matrix(theta=np.pi/2, d=q[0], a=kinematic_params[0], alfa=np.pi/2)
    T_12 = trans_matrix(theta=-np.pi/2, d=q[1]+kinematic_params[1], a=kinematic_params[2], alfa=-np.pi/2)
    T_23 = trans_matrix(theta=0, d=q[2], a=kinematic_params[3], alfa=0)

    T_p0 = np.array([[1, 0, 0, kinematic_params[4]],
                     [0, 1, 0, kinematic_params[5]],
                     [0, 0, 1, kinematic_params[6]],
                     [0, 0, 0, 1]])

    T_p3 = T_p0 @ T_01 @ T_12 @ T_23

    r_33 = np.zeros((4, 1))
    r_33[-1] = 1
    position = np.dot(T_p3, r_33)

    return position[:3]


def compute_inverse_kinematic(position, kinematic_params, joint_limits):
    """
    Computes the joint displacements given the position of the end-effector.
    position is a vector with the (x,y,z) coordinates of the end-effector position in the frame Sp.
    kinematic_params is a vector with the DH-Parameters kinematic_params = (la, lb, lc, ld, oX, oY, oZ).
    joint_limits is an array with the maximum and minimum joint values for each joint in q.

    inputs
        position: numpy array, float, shape (3)
        kinematic_params: numpy array, float, shape (7)
        joint_limits: numpy array, float, shape (3, 2)
    returns
        q: numpy array, float, shape (3)
    """
    # Check if requested position is in workspace. If not, return an empty array np.array([])

    # Compute analytical inverse kinematics
    T_p0 = np.array([[1, 0, 0, kinematic_params[4]],
                     [0, 1, 0, kinematic_params[5]],
                     [0, 0, 1, kinematic_params[6]],
                     [0, 0, 0, 1]])
    T_0p = inverse_trans(T_p0)
    position_0 = np.dot(T_0p, np.append(position, [1]))

    # Compute reference position
    T_01 = trans_matrix(theta=np.pi / 2, d=0, a=kinematic_params[0], alfa=np.pi / 2)
    T_12 = trans_matrix(theta=-np.pi / 2, d=0 + kinematic_params[1], a=kinematic_params[2], alfa=-np.pi / 2)
    T_23 = trans_matrix(theta=0, d=0, a=kinematic_params[3], alfa=0)
    T_03 = T_01 @ T_12 @ T_23
    position_ref = np.dot(T_03, np.append(np.zeros((3, 1)), [1]))

    # Compute delta position
    position_delta = position_0 - position_ref

    # assign q
    q = np.zeros((3, 1))  # initialize
    q[0] = position_delta[2]
    q[1] = position_delta[0]
    q[2] = position_delta[1]

    return q


def estimate_filament_consumption(points, extrusions):
    """
    Computes the total material used for drawing the path implicit in points and extrusions.
    points is an array of via points and extrusions the corresponding use of material between consecutive via points.

    inputs
        points: numpy array, float, with shape (N_via_points, 3)
        extrusions: numpy array, boolean, with shape (N_via_points)
    returns
        consumption: float
    """
    consumption = 0.
    for i, boolen in enumerate(extrusions):
        if i>0 and boolen == 1 and extrusions[i-1] == 1:
            consumption += np.linalg.norm(points[i]-points[i-1], ord=2)
        elif i>0 and boolen == 0 and extrusions[i-1] == 1:
            consumption += np.linalg.norm(points[i] - points[i - 1], ord=2)

    return consumption


def trans_matrix(theta, d, a, alfa):
    """
    Computes the transformation matrix between two coordinates, e.g. n-1 coord to n coord
    which base on DH-Parameter table
    :param theta: [float], rotation angle in z axis, unit in rad
    :param d: [float], translation distance in z axis, unit in m
    :param a: [float], translation distance in x axis, unit in m
    :param alfa: [float], rotation angle in x axis, unit in rad
    :return:
        T: [array], transformation matrix shape(4,4)
    """
    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alfa), np.sin(theta)*np.sin(alfa), a*np.cos(theta)],
                  [np.sin(theta), np.cos(theta)*np.cos(alfa), -np.cos(theta)*np.sin(alfa), a*np.sin(theta)],
                  [0, np.sin(alfa), np.cos(alfa), d],
                  [0, 0, 0, 1]])
    return T


def inverse_trans(T):
    """
    Computes the inverse transformation matrix
    :param T: transformation matrix shape(4,4)
    :return:
        T_inv: inverse transformation matrix shape(4,4)
    """
    T_inv = np.linalg.inv(T)
    return T_inv


if __name__ == "__main__":
    kinematics = np.array([0.02, 0.02, 0.02, 0.048, -0.02, -0.02, 0.07])
    kinematic_params = kinematics
    jointLimits_raw = np.array([-0.002, 0.0, 0.0, 0.4, 0.4, 0.4])
    jointLimits = jointLimits_raw.reshape(2, 3).T
    printStartPosition = np.array([0.02, 0.0, 0.1])
    printEndPosition = np.array([0.02, 0.0, 0.02])

    # test
    (test_path, test_extrusion) = dictionary('C')

    q = np.array([
        0.193046140797008,
        0.149089308797921,
        0.243116195308282])

    test_p = compute_forward_kinematic(q=q, kinematic_params=kinematics, joint_limits=jointLimits)
    test_q = compute_inverse_kinematic(position=test_p, kinematic_params=kinematics, joint_limits=jointLimits)

    test_cost = estimate_filament_consumption(points=test_path, extrusions=test_extrusion)
