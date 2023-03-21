import numpy as np
import torch


class CameraPoseTorch:
    def __init__(self, width, height, fov):
        """
        width, height: img resolution
        fov: default 90
        """
        (
            self.width,
            self.height,
        ) = (
            width,
            height,
        )
        self.fov = fov

        # calculate focal length as standard scalar
        self.f = width / (2 * np.tan(fov / 360.0 * np.pi))

        # for axes swap
        self.Rs_dict = {
            'cpu': torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32),
        }

        # for the interchange between left-right handed system
        self.Ri_dict = {
            'cpu': torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32),
        }

        self.intrinsic_dict = {
            'cpu': torch.tensor(
                [[self.f, 0, self.width / 2], [0, self.f, self.height / 2]],
                dtype=torch.float32,
            )
        }

    def __repr__(self):
        message = f'width:{self.width} height:{self.height} fov:{self.fov} f:{self.f}'
        return message

    def world_to_cam_batch(self, xyz, py, points_3d, py_type='radians'):
        """
        xyz: torch.Tensor of shape [B, 3]
        py: torch.Tensor scalar [B, 2]
        points_3d: torch.Tensor of shape [B, N, 3], 3d points of num N
        Return: [B, N, 3], the last dim is depth
        """
        assert points_3d.ndim == 3, 'shape [B, N, 3] is required'
        device = xyz.device

        C = xyz.unsqueeze(-1)  # [B, 3, 1]
        Rc = self.make_rotation_batch(py, py_type)  # [B, 3, 3]

        Rs = self.Rs_dict.get(device)
        if Rs is None:
            Rs = self.Rs_dict.setdefault(device, self.Rs_dict['cpu'].to(device))  # [3, 3]

        Ri = self.Ri_dict.get(device)
        if Ri is None:
            Ri = self.Ri_dict.setdefault(device, self.Ri_dict['cpu'].to(device))  # [3, 3]

        Ext_R = Rs @ Rc.transpose(1, 2) @ Ri  # [B, 3, 3]
        Ext_t = -Ext_R @ C  # [B, 3, 3] X [B, 3, 1] = [B, 3, 1]

        points_3d_camera = Ext_R @ points_3d.transpose(1, 2) + Ext_t  # [B, 3, N]
        points_3d_camera = points_3d_camera.transpose(1, 2)  # [B, N, 3]

        return points_3d_camera

    def cam_to_world_batch(self, xyz, py, points_3d_camera, py_type='radians'):
        """
        xyz: torch.Tensor of shape [B, 3]
        py: torch.Tensor scalar [B, 2]
        points_3d_camera: torch.Tensor of shape [B, N, 3], 3d points of num N
        Return: [B, N, 3], the last dim is depth
        """
        assert points_3d_camera.ndim == 3, 'shape [B, N, 3] is required'
        device = xyz.device

        C = xyz.unsqueeze(-1)  # [B, 3, 1]
        Rc = self.make_rotation_batch(py, py_type)

        Rs = self.Rs_dict.get(device)  # [3, 3]
        if Rs is None:
            Rs = self.Rs_dict.setdefault(device, self.Rs_dict['cpu'].to(device))

        Ri = self.Ri_dict.get(device)  # [3, 3]
        if Ri is None:
            Ri = self.Ri_dict.setdefault(device, self.Ri_dict['cpu'].to(device))

        points_3d_world = (
            Ri.transpose(0, 1) @ Rc @ Rs.transpose(0, 1) @ points_3d_camera.transpose(1, 2) + C
        )
        points_3d_world = points_3d_world.transpose(1, 2)  # [B, N, 3]
        return points_3d_world

    def cam_to_2d_batch(self, points_3d):
        """
        points_3d: [B, N, 3]
        Return: [B, N, 2]
        """
        assert points_3d.ndim == 3, 'shape [B, N, 3] is required'
        device = points_3d.device
        batch_size, num_points = points_3d.shape[:2]

        # 0 if depth is zero, to prevent nan gradients
        org_den = points_3d[..., [-1]]  # [B, N, 1]
        invalid_index = org_den < self.f  # [B, N, 1]
        ones = torch.ones_like(org_den)  # [B, N, 1]
        zeros = torch.zeros_like(org_den)  # [B, N, 1]
        denominator = torch.where(invalid_index, ones, org_den)
        points_2d = torch.div(points_3d[..., :2], denominator)  # [B, N, 2]
        points_2d = points_2d * torch.where(invalid_index, zeros, ones)  # [B, N, 2]

        ones = torch.ones(
            (batch_size, num_points, 1), dtype=points_2d.dtype, device=device
        )  # [B, N, 1]
        points_3d = torch.cat((points_2d, ones), -1)  # [B, N, 3]

        points_3d = points_3d.unsqueeze(-1)  # [B, N, 3, 1]
        intrinsic = self.intrinsic_dict.get(device)
        if intrinsic is None:
            # [2, 3]
            intrinsic = self.intrinsic_dict.setdefault(
                device, self.intrinsic_dict['cpu'].to(device)
            )  # [3, 3]

        points_2d = intrinsic @ points_3d  # [B, N, 2, 1]
        points_2d = points_2d.squeeze(-1)  # [B, N, 2]

        # for points with invalid depth, corresponding projections are at the center of img. with 0 gradients

        return points_2d

    def world_to_2d_batch(self, xyz, py, points_3d, py_type='radians'):
        points_3d_camera = self.world_to_cam_batch(xyz, py, points_3d, py_type)
        points_2d = self.cam_to_2d_batch(points_3d_camera)
        return points_2d

    def make_rotation_batch(self, input_py, input_type='radians'):
        """
        input_py: pitch, yaw [B, 2]
        input_type: cos or radians or degrees
        Return: [B, 3, 3]
        """
        assert input_py.ndim == 2, 'shape [B, 2] is required'
        device = input_py.device
        batch_size = input_py.shape[0]
        if input_type == 'cos':
            cos_py = input_py  # [B, 2]
            sin_py = torch.acos(input_py).sin()  # [B, 2]
        elif input_type == 'degrees':
            py = input_py / 180.0 * np.pi
            cos_py = py.cos()
            sin_py = py.sin()
        elif input_type == 'radians':
            cos_py = input_py.cos()
            sin_py = input_py.sin()
        else:
            raise RuntimeError

        cos_pitch = cos_py[:, [0]]  # [B, 1]
        cos_yaw = cos_py[:, [1]]  # [B, 1]
        sin_pitch = sin_py[:, [0]]  # [B, 1]
        sin_yaw = sin_py[:, [1]]  # [B, 1]

        zeros = torch.zeros_like(cos_pitch)  # [B, 1]
        ones = torch.ones_like(cos_pitch)  # [B, 1]

        ryaw_flat = (
            cos_yaw,
            -sin_yaw,
            zeros,
            sin_yaw,
            cos_yaw,
            zeros,
            zeros,
            zeros,
            ones,
        )
        ryaw = torch.cat(ryaw_flat, -1).view(batch_size, 3, 3)  # [B, 3, 3]

        rpitch_flat = (
            cos_pitch,
            zeros,
            sin_pitch,
            zeros,
            ones,
            zeros,
            -sin_pitch,
            zeros,
            cos_pitch,
        )
        rpitch = torch.cat(rpitch_flat, -1).view(batch_size, 3, 3)  # [B, 3, 3]

        # rroll is identity matrix
        R = ryaw @ rpitch  # [B, 3, 3]

        return R
