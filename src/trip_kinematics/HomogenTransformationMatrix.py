import numpy as np
from casadi import MX, cos, sin


def quat_rotation_matrix(qw, qx, qy, qz) -> np.ndarray:
    """[summary]

    Args:
        a ([type]): [description]
        b ([type]): [description]
        c ([type]): [description]
        d ([type]): [description]

    Returns:
        np.matrix: [description]
    """
    return np.array([[1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz + qy*qw)], [2*(qx*qy + qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz - qx*qw)], [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]], dtype=object)


def x_axis_rotation_matrix(theta):
    """[summary]

    Args:
        theta ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.array([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]], dtype=object)


def y_axis_rotation_matrix(theta):
    """[summary]

    Args:
        theta ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]], dtype=object)


def z_axis_rotation_matrix(theta):
    """[summary]

    Args:
        theta ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]], dtype=object)


class HomogenousTransformationMatrix:
    """[summary]
    """

    def __init__(self, qw=1, qx=0, qy=0, qz=0, tx=0, ty=0, tz=0, conv='quat', rx=0, ry=0, rz=0):
        """[summary]

        Args:
            a (int, optional): [description]. Defaults to 0.
            b (int, optional): [description]. Defaults to 0.
            c (int, optional): [description]. Defaults to [summary] 0.
        """
        self.matrix: np.array = np.array(
            [[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0., 0., 0., 1.]], dtype=object)
        if conv == 'quat':
            self.matrix[:3, :3] = quat_rotation_matrix(qw, qx, qy, qz)
        if conv == 'xyz':
            self.matrix[:3, :3] = x_axis_rotation_matrix(
                rx) @ y_axis_rotation_matrix(ry) @ z_axis_rotation_matrix(rz)

    def get_translation(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.matrix[: 3, 3]

    def get_rotation(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.matrix[: 3, : 3]

    def __mul__(self, other):
        """[summary]

        Args:
            other ([type]): [description]

        Returns:
            [type]: [description]
        """
        new = HomogenousTransformationMatrix()
        new.matrix = self.matrix @ other.matrix
        return new

    def __str__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return str(self.matrix)
