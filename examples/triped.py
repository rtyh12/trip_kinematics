from trip_kinematics.KinematicObject import KinematicObject
from trip_kinematics.KinematicGroup import KinematicGroup
from trip_kinematics.Robot import Robot, forward_kinematic
from casadi import Opti
from typing import Dict
from trip_kinematics.HomogenTransformationMartix import Homogenous_transformation_matrix
import numpy as np
from math import radians
from tf.transformations import quaternion_from_euler


def c(rx, ry, rz, opti):
    A_CSS_P = Homogenous_transformation_matrix(
        tx=0.265, ty=0, tz=0.014, rx=rx, ry=ry, rz=rz, conv='xyz')

    T_P_SPH1_2 = np.array([-0.015, -0.029, 0.0965]) * -1
    T_P_SPH2_2 = np.array([-0.015, 0.029, 0.0965]) * -1
    x0, y0, z0 = T_P_SPH1_2
    x1, y1, z1 = T_P_SPH2_2

    A_P_SPH1_2 = Homogenous_transformation_matrix(
        tx=x0, ty=y0, tz=z0, conv='xyz')
    A_P_SPH2_2 = Homogenous_transformation_matrix(
        tx=x1, ty=y1, tz=z1, conv='xyz')

    A_c1 = A_CSS_P * A_P_SPH1_2
    A_c2 = A_CSS_P * A_P_SPH2_2

    c1 = A_c1.get_translation()
    c2 = A_c2.get_translation()

    c1_mx = opti.variable(3, 1)
    c1_mx[0, 0] = c1[0]
    c1_mx[1, 0] = c1[1]
    c1_mx[2, 0] = c1[2]

    c2_mx = opti.variable(3, 1)
    c2_mx[0, 0] = c2[0]
    c2_mx[1, 0] = c2[1]
    c2_mx[2, 0] = c2[2]

    c1 = c1_mx
    c2 = c2_mx
    return c1, c2


def p1(theta, opti):
    A_CCS_lsm_tran = Homogenous_transformation_matrix(
        tx=0.139807669447128, ty=0.0549998406976098, tz=-0.051)

    A_CCS_lsm_rot = Homogenous_transformation_matrix(
        rz=radians(-338.5255), conv='xyz')

    A_CCS_lsm = A_CCS_lsm_tran * A_CCS_lsm_rot

    A_MCS1_JOINT = Homogenous_transformation_matrix(rz=theta, conv='xyz')

    A_CSS_MCS1 = A_CCS_lsm * A_MCS1_JOINT

    A_MCS1_SP11 = Homogenous_transformation_matrix(tx=0.085, ty=0, tz=-0.0245)

    A_CCS_SP11 = A_CSS_MCS1 * A_MCS1_SP11

    p1 = A_CCS_SP11.get_translation()
    p1_mx = opti.variable(3, 1)
    p1_mx[0, 0] = p1[0]
    p1_mx[1, 0] = p1[1]
    p1_mx[2, 0] = p1[2]
    return p1_mx


def p2(theta, opti):
    A_CCS_rsm_tran = Homogenous_transformation_matrix(
        tx=0.139807669447128, ty=-0.0549998406976098, tz=-0.051)

    A_CCS_rsm_rot = Homogenous_transformation_matrix(
        rz=radians(-21.4745), conv='xyz')

    A_CCS_rsm = A_CCS_rsm_tran*A_CCS_rsm_rot

    A_MCS2_JOINT = Homogenous_transformation_matrix(rz=theta, conv='xyz')

    A_CSS_MCS2 = A_CCS_rsm * A_MCS2_JOINT

    A_MCS2_SP21 = Homogenous_transformation_matrix(tx=0.085, ty=0, tz=-0.0245)

    A_CSS_SP21 = A_CSS_MCS2 * A_MCS2_SP21

    p2 = A_CSS_SP21.get_translation()
    p2_mx = opti.variable(3, 1)
    p2_mx[0, 0] = p2[0]
    p2_mx[1, 0] = p2[1]
    p2_mx[2, 0] = p2[2]
    return p2_mx


def mapping_f(state: Dict[str, float]):
    theta_right = state['t1']
    theta_left = state['t2']
    opti = Opti()
    r = 0.11

    gimbal_x = opti.variable()
    gimbal_y = opti.variable()
    gimbal_z = opti.variable()

    c1, c2 = c(rx=gimbal_x, ry=gimbal_y, rz=gimbal_z, opti=opti)
    closing_equation = ((c1-p1(theta_right, opti)).T @ (c1-p1(theta_right, opti)) -
                        r**2)**2+((c2-p2(theta_left, opti)).T @ (c2-p2(theta_left, opti)) - r**2)**2

    opti.minimize(closing_equation)
    p_opts = {"print_time": False}
    s_opts = {"print_level": 0, "print_timing_statistics": "no"}
    opti.solver('ipopt', p_opts, s_opts)
    sol = opti.solve()
    quat = quaternion_from_euler(
        sol.value(gimbal_x), sol.value(gimbal_y), sol.value(gimbal_z))
    return [{'q0': quat[3], 'q1':quat[0], 'q2':quat[1], 'q3':quat[2]}]


if __name__ == '__main__':

    A_CSS_P = KinematicObject(name='A_CSS_P', values={
        'x': 0.265, 'z': 0.014, 'q0': 1, 'q1': 0, 'q2': 0, 'q3': 0}, stateVariables=['q0', 'q1', 'q2', 'q3'])

    gimbal_joint = KinematicGroup(name='gimbal_joint', open_chain=[
        A_CSS_P], initial_state={'t1': 0, 't2': 0}, f_mapping=mapping_f, g_mapping=lambda x: len(x))

    A_P_LL_joint = KinematicObject(name='A_P_LL_joint',
                                   values={'x': 1.640, 'z': -0.037, 'q0': 1, 'q1': 0, 'q2': 0, 'q3': 0}, stateVariables=['q0', 'q1', 'q2', 'q3'], parent=gimbal_joint)

    A_LL_Joint_FCS = KinematicObject(name='A_LL_Joint_FCS', values={
        'x': -1.5}, parent=A_P_LL_joint)

    robot = Robot([gimbal_joint, A_P_LL_joint, A_LL_Joint_FCS])

    print(forward_kinematic(robot))