from trip_kinematics.KinematicGroup import KinematicGroup, Transformation
from trip_kinematics.Robot import Robot, inverse_kinematics, forward_kinematics
from casadi import Opti, sqrt
from typing import Dict, List
from trip_kinematics.HomogenTransformationMatrix import HomogenousTransformationMatrix
import numpy as np
from math import radians, sin, cos


def right_swing_sub_chain(theta1, qw1, qx1, qy1, qz1, qw2, qx2, qy2, qz2):
    A_CCS_rsm_tran = HomogenousTransformationMatrix(
        tx=0.139807669447128, ty=-0.0549998406976098, tz=-0.051)

    A_CCS_rsm_rot = HomogenousTransformationMatrix(
        rz=radians(-21.4745), conv='xyz')  # radians(-21.4745)-34.875251275010434

    A_CCS_rsm = A_CCS_rsm_tran*A_CCS_rsm_rot

    A_MCS2_JOINT = HomogenousTransformationMatrix(
        rz=theta, conv='xyz')

    A_CSS_MCS2 = A_CCS_rsm * A_MCS2_JOINT

    A_MCS2_SP21 = HomogenousTransformationMatrix(
        tx=0.085, ty=0, tz=-0.0245)

    A_CSS_SP21 = A_CSS_MCS2 * A_MCS2_SP21

    A5 = HomogenousTransformationMatrix(
        conv='quat', qw=qw1, qx=qx1, qy=qy1, qz=qz1)

    A6 = HomogenousTransformationMatrix(tx=0.110, ty=0, tz=0)

    A7 = HomogenousTransformationMatrix(
        conv='quat', qw=qw2, qx=qx2, qy=qy2, qz=qz2)

    A8 = HomogenousTransformationMatrix(tx=-0.015, ty=0.0299, tz=0.0965)

    path = A_CSS_SP21 * A5 * A6 * A7 * A8

    return path


def left_swing_sub_chain(theta1, qw1, qx1, qy1, qz1, qw2, qx2, qy2, qz2):
    A_CCS_lsm_tran = HomogenousTransformationMatrix(
        tx=0.139807669447128, ty=0.0549998406976098, tz=-0.051)

    A_CCS_lsm_rot = HomogenousTransformationMatrix(
        rz=radians(-338.5255), conv='xyz')  # radians()34.875251275010434

    A_CCS_lsm = A_CCS_lsm_tran * A_CCS_lsm_rot

    A_MCS1_JOINT = HomogenousTransformationMatrix(
        rz=theta, conv='xyz')

    A_CSS_MCS1 = A_CCS_lsm * A_MCS1_JOINT

    A_MCS1_SP11 = HomogenousTransformationMatrix(
        tx=0.085, ty=0, tz=-0.0245)

    A_CCS_SP11 = A_CSS_MCS1 * A_MCS1_SP11

    A5 = HomogenousTransformationMatrix(
        conv='quat', qw=qw1, qx=qx1, qy=qy1, qz=qz1)

    A6 = HomogenousTransformationMatrix(tx=0.110, ty=0, tz=0)

    A7 = HomogenousTransformationMatrix(
        conv='quat', qw=qw2, qx=qx2, qy=qy2, qz=qz2)

    A8 = HomogenousTransformationMatrix(tx=-0.015, ty=-0.0299, tz=0.0965)

    path = A_CSS_SP21 * A5 * A6 * A7 * A8

    return path


def bridge_sub_chain(qw, qx, qy, qz):

    A_CSS_P_trans = HomogenousTransformationMatrix(
        tx=0.265, ty=0, tz=0.014)

    A_CSS_P_rot = HomogenousTransformationMatrix(
        conv='quat', qw=qw, qx=qx, qy=qy, qz=qz)

    A_CSS_P = A_CSS_P_trans * A_CSS_P_rot

    return A_CSS_P


def norm_constraint(qw, qx, qy, qz):

    return qw**2 + qx**2 + qy**2 + qz**2 - 1


def quad_norm(matrix):
    norm = 0
    for i in range(4):
        for j in range(4):
            norm += matrix[i][j]**2
    return norm


def f_map(state, tip):
    opti = Opti()

    r_qw_1 = opti.variable()
    r_qx_1 = opti.variable()
    r_qy_1 = opti.variable()
    r_qz_1 = opti.variable()
    r_qw_2 = opti.variable()
    r_qx_2 = opti.variable()
    r_qy_2 = opti.variable()
    r_qz_2 = opti.variable()

    l_qw_1 = opti.variable()
    l_qx_1 = opti.variable()
    l_qy_1 = opti.variable()
    l_qz_1 = opti.variable()
    l_qw_2 = opti.variable()
    l_qx_2 = opti.variable()
    l_qy_2 = opti.variable()
    l_qz_2 = opti.variable()

    b_qw = opti.variable()
    b_qx = opti.variable()
    b_qy = opti.variable()
    b_qz = opti.variable()

    path_1 = left_swing_sub_chain(
        state[0]['t1'], l_qw_1, l_qx_1, l_qy_1, l_qz_1, l_qw_2, l_qx_2, l_qy_2, l_qz_2)
    path_2 = right_swing_sub_chain(
        state[0]['t2'], r_qw_1, r_qx_1, r_qy_1, r_qz_1, r_qw_2, r_qx_2, r_qy_2, r_qz_2)
    path_3 = bridge_sub_chain(b_qw, b_qx, b_qy, b_qz)

    opti.subject_to(norm_constraint(r_qw_1, r_qx_1, r_qy_1, r_qz_1))
    opti.subject_to(norm_constraint(r_qw_2, r_qx_2, r_qy_2, r_qz_2))
    opti.subject_to(norm_constraint(l_qw_1, l_qx_1, l_qy_1, l_qz_1))
    opti.subject_to(norm_constraint(l_qw_2, l_qx_2, l_qy_2, l_qz_2))
    opti.subject_to(norm_constraint(b_qw, b_qx, b_qy, b_qz))

    closing_equation = quad_norm(path_1.matrix - path_2.matrix) + quad_norm(
        path_2.matrix - path_3.matrix) + quad_norm(path_3.matrix - path_1.matrix)

    opti.minimize(closing_equation)

    sol = opti.solve()

    gimbal_qw = sol.value(b_qw)
    gimbal_qx = sol.value(b_qx)
    gimbal_qy = sol.value(b_qy)
    gimbal_qz = sol.value(b_qz)

    return [{}, {'qw': gimbal_qw, 'qx': gimbal_qx, 'qy': gimbal_qy, 'qz': gimbal_qz}]


def g_map(state):
    return [{'t1': 0, 't2': 0}]


A_CSS_P_trans = Transformation(name='A_CSS_P_trans',
                               values={'tx': 0.265, 'tz': 0.014})

A_CSS_P_rot = Transformation(name='A_CSS_P_rot',
                             values={'qw': 0, 'qx': 0, 'qy': 0, 'qz': 0}, state_variables=['qw', 'qx', 'qy', 'qz'])

gimbal_joint = KinematicGroup(name='gimbal_joint', virtual_transformations=[A_CSS_P_trans,
                                                                            A_CSS_P_rot], actuated_state=[{'t1': 0, 't2': 0}], f_mapping=f_map, g_mapping=g_map)

A_P_LL = Transformation(name='A_P_LL', values={'tx': 1.640, 'tz': -0.037, })

zero_angle_convention = Transformation(name='zero_angle_convention',
                                       values={'ry': radians(3)})

LL_revolute_joint = Transformation(name='LL_revolute_joint',
                                   values={'ry': 0}, state_variables=['ry'])

A_LL_Joint_FCS = Transformation(name='A_LL_Joint_FCS', values={'tx': -1.5})

extend_motor = KinematicGroup(name='extend_motor', virtual_transformations=[A_P_LL, zero_angle_convention,
                                                                            LL_revolute_joint, A_LL_Joint_FCS], parent=gimbal_joint)

triped_leg = Robot([gimbal_joint, extend_motor])

gimbal_joint.set_actuated_state([{'t1': 0, 't2': 0}])
