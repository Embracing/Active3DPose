import numpy as np


def make_viewpoint(obj_loc, dist, az, el):
    """obj_loc: [x, y, z]"""

    # not checked !!!
    az = az / 180.0 * np.pi
    el = el / 180.0 * np.pi
    T1 = np.array(
        [
            [1, 0, 0, obj_loc[0]],
            [0, 1, 0, obj_loc[1]],
            [0, 0, 1, obj_loc[2]],
            [0, 0, 0, 1],
        ]
    )
    T2 = np.array(
        [
            [1, 0, 0, np.cos(el) * np.cos(az) * dist],
            [0, 1, 0, np.cos(el) * np.sin(az) * dist],
            [0, 0, 1, np.sin(el) * dist],
            [0, 0, 0, 1],
        ]
    )
    S = np.array([[0], [0], [0], [1]])  # 4x1
    cam = T1.dot(T2).dot(S)
    cam_pose = [
        cam[0][0],
        cam[1][0],
        cam[2][0],
        -np.degrees(el),
        np.degrees(az) + 180,
        0,
    ]
    return cam_pose


def make_location(X, Y, Z):
    T = np.array([[X], [Y], [Z]])
    return T


def make_rotation(pitch_unreal, yaw_unreal, roll_unreal):
    # Convert from degree to radius
    pitch_unreal = pitch_unreal / 180.0 * np.pi
    yaw_unreal = yaw_unreal / 180.0 * np.pi
    roll_unreal = roll_unreal / 180.0 * np.pi

    # new
    roll = roll_unreal
    yaw = yaw_unreal
    pitch = pitch_unreal

    ryaw = [
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ]
    rpitch = [
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ]
    rroll = [
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ]
    R = np.array(ryaw) @ np.array(rpitch) @ np.array(rroll)

    return R


class CameraPose:
    def __init__(self, x, y, z, pitch, yaw, roll, width, height, fov):
        self.x, self.y, self.z = x, y, z
        self.pitch, self.yaw, self.roll = pitch, yaw, roll
        (
            self.width,
            self.height,
        ) = (
            width,
            height,
        )
        self.fov = fov

        # calculate focal length
        self.f = width / (2 * np.tan(fov / 360.0 * np.pi))

        # for axes swap
        self.Rs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

        # for the interchange between left-right handed system
        self.Ri = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

    def __repr__(self):
        message = (
            'x:{x} y:{y} z:{z} pitch:{pitch} yaw:{yaw} roll:{roll}'
            ' width:{width} height:{height} fov:{fov} f:{f:.2f}'
        )
        return message.format(**self.__dict__)

    def project_to_2d(self, points_3d, return_depth=False):
        """
        points_3d: points in 3D world coordinate, [N, 3]
        Return:
            [N, 2] if return_depth == False, else [N, 3] with camera-frame depth
        """
        if not points_3d.shape[1] == 3:
            print('The input shape is not n x 3, but n x %d' % points_3d.shape[1])
            # TODO: replace this with logging
            return

        n_points = points_3d.shape[0]

        C = make_location(self.x, self.y, self.z)  # C, camera origin in world coordinates
        Rc = make_rotation(
            self.pitch, self.yaw, self.roll
        )  # Rc, direction of camera axex in world coordinates.

        Ext_R = self.Rs @ Rc.T @ self.Ri  # [3, 3]
        Ext_t = -Ext_R @ C  # [3, 3] X [3, 1] = [3, 1]

        points_3d_camera = Ext_R @ points_3d.T + Ext_t  # [3, N]
        points_3d_camera = points_3d_camera.T  # [N, 3]

        # TODO: Need to fix this
        half_width = self.width / 2
        half_height = self.height / 2

        np.divide(
            points_3d_camera[:, :2],
            points_3d_camera[:, [2]],
            out=points_3d_camera[:, :2],
            where=points_3d_camera[:, [2]] != 0,
        )
        points_3d_camera[:, 0] = points_3d_camera[:, 0] * self.f + half_width
        points_3d_camera[:, 1] = points_3d_camera[:, 1] * self.f + half_height
        # points_3d_camera[:, 0] = points_3d_camera[:, 0] / points_3d_camera[:, 2] * self.f + half_width
        # points_3d_camera[:, 1] = points_3d_camera[:, 1] / points_3d_camera[:, 2] * self.f + half_height

        if return_depth:
            return points_3d_camera
        else:
            return points_3d_camera[:, :2]

    def world_to_cam(self, points_3d):
        """
        points_3d: points in 3D world coordinate, [N, 3]
        """
        if points_3d.ndim == 1 and len(points_3d) == 3:
            points_3d = points_3d[None, ...]  # [1, 3]

        if not points_3d.shape[1] == 3:
            print('The input shape is not n x 3, but n x %d' % points_3d.shape[1])
            # TODO: replace this with logging
            return

        n_points = points_3d.shape[0]

        C = make_location(self.x, self.y, self.z)
        Rc = make_rotation(self.pitch, self.yaw, self.roll)
        # cam_R  = cam_R * -1
        # print(np.linalg.det(cam_R))

        Ext_R = self.Rs @ Rc.T @ self.Ri  # [3, 3]
        Ext_t = -Ext_R @ C  # [3, 3] X [3, 1] = [3, 1]

        points_3d_camera = Ext_R @ points_3d.T + Ext_t  # [3, N]
        points_3d_camera = points_3d_camera.T  # [N, 3]

        return points_3d_camera

    def cam_to_world(self, points_3d_camera):
        """
        Used to plot camera
        Return: [N x 3]
        """
        assert points_3d_camera.shape[1] == 3, 'shape should be n x 3'

        C = make_location(self.x, self.y, self.z)
        Rc = make_rotation(self.pitch, self.yaw, self.roll)

        points_3d_world = self.Ri.T @ Rc @ self.Rs.T @ points_3d_camera.T + C
        points_3d_world = points_3d_world.T  # [N, 3]

        return points_3d_world

    def update_camera_parameters(self, **kwargs):
        """
        Update x,y,z, pitch,yaw,roll,fov,
        Do not update width height focal here !!!!
        """
        self.__dict__.update(kwargs)
        if 'fov' in kwargs:
            # calculate focal length
            self.f = self.width / (2 * np.tan(self.fov / 360.0 * np.pi))

    def update_camera_parameters_list(self, params):
        """
        Update x,y,z, pitch,yaw,roll,fov
        params: [7]
        Do not update width height focal here !!!!
        """
        self.x, self.y, self.z = params[0], params[1], params[2]
        self.pitch, self.yaw, self.roll = params[3], params[4], params[5]
        self.fov = params[6]
        self.f = self.width / (2 * np.tan(self.fov / 360.0 * np.pi))

    def get_intrinsic(self, homo=True):
        intrinsic = np.array([[self.f, 0, self.width / 2], [0, self.f, self.height / 2]])
        if homo:
            intrinsic = np.vstack((intrinsic, np.array((0, 0, 1))))  # [3, 3]
        return intrinsic

    def get_extrinsic(self, homo=True):
        C = make_location(self.x, self.y, self.z)
        Rc = make_rotation(self.pitch, self.yaw, self.roll)

        Ext_R = self.Rs @ Rc.T @ self.Ri  # [3, 3]
        Ext_t = -Ext_R @ C  # [3, 3] X [3, 1] = [3, 1]

        extrinsic = np.concatenate((Ext_R, Ext_t), axis=1)  # [3, 4]
        if homo:
            extrinsic = np.vstack((extrinsic, np.array((0, 0, 0, 1))))  # [4, 4]
        return extrinsic
