from trip_kinematics.KinematicGroup import KinematicGroup, Transformation
from trip_kinematics.Robot import Robot, inverse_kinematics, forward_kinematics
from scipy.optimize import minimize
from typing import Dict, List
from trip_kinematics.HomogenTransformationMatrix import TransformationMatrix
import numpy as np
from math import radians, sin, cos


def c(rx, ry, rz):
    A_CSS_P_trans = TransformationMatrix(
        tx=0.265, ty=0, tz=0.014)

    A_CSS_P_rot = TransformationMatrix(
        conv='xyz', rx=rx, ry=ry, rz=rz)

    A_CSS_P = A_CSS_P_trans * A_CSS_P_rot

    T_P_SPH1_2 = np.array([-0.015, -0.029, 0.0965]) * -1
    T_P_SPH2_2 = np.array([-0.015, 0.029, 0.0965]) * -1
    x0, y0, z0 = T_P_SPH1_2
    x1, y1, z1 = T_P_SPH2_2

    A_P_SPH1_2 = TransformationMatrix(
        tx=x0, ty=y0, tz=z0, conv='xyz')
    A_P_SPH2_2 = TransformationMatrix(
        tx=x1, ty=y1, tz=z1, conv='xyz')

    A_c1 = A_CSS_P * A_P_SPH1_2
    A_c2 = A_CSS_P * A_P_SPH2_2

    c1 = A_c1.get_translation()
    c2 = A_c2.get_translation()


    return c1, c2


def p1(theta):
    A_CCS_lsm_tran = TransformationMatrix(
        tx=0.139807669447128, ty=0.0549998406976098, tz=-0.051)

    A_CCS_lsm_rot = TransformationMatrix(
        rz=radians(-338.5255), conv='xyz')  

    A_CCS_lsm = A_CCS_lsm_tran * A_CCS_lsm_rot

    A_MCS1_JOINT = TransformationMatrix(
        rz=theta, conv='xyz')

    A_CSS_MCS1 = A_CCS_lsm * A_MCS1_JOINT

    A_MCS1_SP11 = TransformationMatrix(
        tx=0.085, ty=0, tz=-0.0245)

    A_CCS_SP11 = A_CSS_MCS1 * A_MCS1_SP11

    p1 = A_CCS_SP11.get_translation()

    return p1


def p2(theta):
    A_CCS_rsm_tran = TransformationMatrix(
        tx=0.139807669447128, ty=-0.0549998406976098, tz=-0.051)

    A_CCS_rsm_rot = TransformationMatrix(
        rz=radians(-21.4745), conv='xyz')  

    A_CCS_rsm = A_CCS_rsm_tran*A_CCS_rsm_rot

    A_MCS2_JOINT = TransformationMatrix(
        rz=theta, conv='xyz')

    A_CSS_MCS2 = A_CCS_rsm * A_MCS2_JOINT

    A_MCS2_SP21 = TransformationMatrix(
        tx=0.085, ty=0, tz=-0.0245)

    A_CSS_SP21 = A_CSS_MCS2 * A_MCS2_SP21

    p2 = A_CSS_SP21.get_translation()

    return p2


def swing_to_gimbal(state: Dict[str, float], tips: Dict[str, float] = None):

    r = float(0.11)

    theta_left = state['swing_left']
    theta_right = state['swing_right']


    x_0 = [0,0,0]
    if tips:
        x_0[0] = tips['rx']
        x_0[1] = tips['ry']
        x_0[2] = tips['rz']


    def closing_equation(x):
        c1, c2 = c(rx=x[0], ry=x[1], rz=x[2])
        closing_eq= (sum((c1-p1(theta_right))**2) - r**2)**2+ (sum((c2-p2(theta_left))**2) - r**2)**2 
        return closing_eq

    sol = minimize(closing_equation,x_0,method='L-BFGS-B')
    return {'gimbal_joint': {'rx': sol.x[0], 'ry': sol.x[1], 'rz': sol.x[2]}}


def gimbal_to_swing(state: Dict[str,Dict[str, float]], tips: Dict[str, float] = None):

    r = float(0.11)

    gimbal_x = state['gimbal_joint']['rx']
    gimbal_y = state['gimbal_joint']['ry']
    gimbal_z = state['gimbal_joint']['rz']

    x_0 = [0,0]
    if tips:
        x_0[0] = tips['swing_left']
        x_0[1] = tips['swing_right']



    def closing_equation(x):
        c1, c2 = c(rx=gimbal_x, ry=gimbal_y, rz=gimbal_z)
        closing_eq= (sum((c1-p1(x[1]))**2)- r**2)**2 + (sum((c2-p2(x[0]))**2)- r**2)**2  
        return closing_eq

    sol = minimize(closing_equation,x_0,method='L-BFGS-B')
    return {'swing_left': sol.x[0], 'swing_right': sol.x[1]}


A_CSS_P_trans = Transformation(name='A_CSS_P_trans',
                               values={'tx': 0.265, 'tz': 0.014})

A_CSS_P_rot = Transformation(name='gimbal_joint',
                             values={'rx': 0, 'ry': 0, 'rz': 0}, state_variables=['rx', 'ry', 'rz'])

closed_chain = KinematicGroup(name='closed_chain', virtual_transformations=[A_CSS_P_trans,A_CSS_P_rot], 
                              actuated_state={'swing_left': 0, 'swing_right': 0}, 
                              actuated_to_virtual=swing_to_gimbal, virtual_to_actuated=gimbal_to_swing)

A_P_LL = Transformation(name='A_P_LL', values={'tx': 1.640, 'tz': -0.037, })

zero_angle_convention = Transformation(name='zero_angle_convention',
                                       values={'ry': radians(-3)})

extend_joint = Transformation(name='extend_joint',
                                   values={'ry': 0}, state_variables=['ry'])

A_LL_Joint_FCS = Transformation(name='A_LL_Joint_FCS', values={'tx': -1.5})

leg_linear_part = KinematicGroup(name='leg_linear_part',
                                 virtual_transformations=[A_P_LL, zero_angle_convention,extend_joint, A_LL_Joint_FCS], 
                                 parent=closed_chain)

triped_leg = Robot([closed_chain, leg_linear_part])

closed_chain.set_actuated_state({'swing_left': 0, 'swing_right': 0})
