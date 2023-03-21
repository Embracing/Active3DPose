import json
import os
import re
import sys
import time
from functools import wraps
from io import BytesIO

import numpy as np
import PIL.Image

from .client import Client


class Interaction:
    def __init__(self, env_dir, ip='127.0.0.1', port=9000, unix_socket_path=None):
        self.closed = False

        # self.client = Client((ip, port))

        self.env_dir = env_dir
        self.ip = ip
        self.port = port
        self.unix_socket_path = unix_socket_path

        self.cam_params = {}  # {cam_id(int): x, y, z, pitch, yaw, roll, fov}
        self.color_dict = {}
        self.map_name = 'Blank'

        self.build_connection()
        self.init_unrealcv()
        self.init_viewmode_meta()

    def build_connection(self):
        """
        Take platform-specific actions to build client
        """
        # common inet connection for both linux and windows
        self.client = Client((self.ip, self.port))
        self.check_connection()

        if 'linux' in sys.platform and self.unix_socket_path is not None:
            self.client.disconnect()
            print('Disconnect Inet connection. Waiting for UDS service starts(6s)...')
            time.sleep(6)

            # delete this if-clause to force UDS connection on Linux
            if os.path.exists(self.unix_socket_path):
                print('Changing to UDS communication...')
                self.client = Client(self.unix_socket_path, 'unix')
                self.check_connection()
                print('=>Info: Use UDS communication')
            else:
                print('UDS service does not start. Change back to Inet again')
                self.check_connection()
        else:
            print('=>Info: Use inet communication')

    def init_unrealcv(self):
        # self.client.connect()
        # self.check_connection()

        # self.client.request('vrun setres {w}x{h}w'.format(w=resolution[0], h=resolution[1]))
        # self.client.request('DisableAllScreenMessages')
        self.client.request('vrun sg.ShadowQuality 0')

        # default value is 3
        # self.client.request('vrun sg.TextureQuality 0')
        # self.client.request('vrun sg.EffectsQuality 0')
        # self.client.request('vrun sg.ViewDistanceQuality 0')
        # self.client.request('vrun sg.AntiAliasingQuality 0')
        # self.client.request('vrun sg.FoliageQuality 0')
        # self.client.request('vrun sg.ShadingQuality 0')
        # default value is 100
        # self.client.request('vrun sg.ResolutionQuality 100')

        # self.client.request('vrun r.ScreenPercentage 10')
        # self.client.request('vrun t.maxFPS 62')
        # time.sleep(1)

        self.client.message_handler = self.message_handler
        self.config = self.parse_config()
        self.resolution = (int(self.config['width']), int(self.config['height']))

    def init_viewmode_meta(self):
        bmp_pair = ('bmp', self.decode_bmp)
        npy_pair = ('npy', self.decode_npy)
        self.viewmode_meta = {
            'lit': bmp_pair,
            'normal': bmp_pair,
            'object_mask': bmp_pair,
            'depth': npy_pair,
        }

    def message_handler(self, message):
        msg = message

    def parse_config(self):
        res = None
        while res is None:
            res = self.client.request('vget /unrealcv/status')
        lines = res.split('\n')

        config = {}
        for line in lines:
            if ':' in line:
                semiclone = line.find(':')
                key, value = line[:semiclone], line[semiclone + 1 :]
                key, value = key.strip(), value.strip()
                config[key.lower()] = value
        return config

    def check_connection(self):
        while self.client.isconnected() is False:
            print('UnrealCV server is not running. Trying connecting again...')
            self.client.connect()
            time.sleep(1)
        else:
            print('Connection Established')

    def close(self):
        self.client.disconnect()
        print('Disconnected!')
        self.closed = True

    def __del__(self):
        if self.closed:
            return

        self.close()

    # wrapper func =================================================
    def _check_setcmd_success(expected_return):
        def decorator(func):
            @wraps(func)
            def decorated(self, *args, **kargs):
                count = 0
                success_flag = False
                while True:
                    res = func(self, *args, **kargs)
                    if isinstance(res, list):
                        if isinstance(expected_return, list):
                            exp_len = sum([res.count(item) for item in expected_return])
                        else:
                            exp_len = res.count(expected_return)
                        if exp_len == len(res):
                            success_flag = True
                    else:
                        # tricky, 'ok' in 'ok', or 'ok' in ['ok', 'error']
                        if res in expected_return:
                            success_flag = True

                    if success_flag:
                        break
                    else:
                        count += 1
                        if count > 5:
                            self.close()
                            print('set failed: %s' % func.__name__)
                            print('result:', res)
                            # sys.exit(0)
                            break
                return res

            return decorated

        return decorator

    # ============= util func ========================================
    @_check_setcmd_success('ok')
    def open_map(self, map_name):
        self.map_name = map_name

        # clear state
        self.cam_params = {}  # {cam_id(int): x, y, z, pitch, yaw, roll, fov}
        self.color_dict = {}

        return self.client.request(f'vset /action/game/level {map_name}')

    def get_map_name(self):
        self.map_name = self.client.request('vget /level/name')
        return self.map_name

    @_check_setcmd_success('ok')
    def keyboard(self, key, duration=0.01):  # Up Down Left Right
        cmd = 'vset /action/keyboard {key} {duration}'
        return self.client.request(cmd.format(key=key, duration=duration))

    def decode_bmp(self, res):
        # return RGBA ndarray
        img = np.frombuffer(res, dtype=np.uint8)
        img = img[-self.resolution[1] * self.resolution[0] * 4 :]
        img = img.reshape(self.resolution[1], self.resolution[0], 4)  # [h, w, c]
        img = img[:, :, [2, 1, 0, 3]]
        return img

    def decode_png(self, res):
        img = PIL.Image.open(BytesIO(res))
        return np.asarray(img)

    def decode_npy(self, res):
        img = np.fromstring(res, np.float32)
        img = img[-self.resolution[1] * self.resolution[0] :]
        img = img.reshape(self.resolution[1], self.resolution[0], 1)
        return img

    def get_img(self, cam_id, viewmode):
        """
        Return: [h, w, 4], RGBA
        """
        cmd = f'vget /camera/{cam_id}/{viewmode} bmp'
        res = self.client.request(cmd)
        img = self.decode_bmp(res)
        return img

    def get_depth_img(self, cam_id, inverse=False):
        """
        Return: [h, w, 1]
        """
        cmd = f'vget /camera/{cam_id}/depth npy'
        res = self.client.request(cmd)
        depth = self.decode_npy(res)
        if inverse:
            depth = 1 / depth
        return depth

    # object related =====================================================
    def get_objects(self):
        objects = None
        while objects is None:
            objects = self.client.request('vget /objects')
        objects = objects.split()
        return objects

    @_check_setcmd_success('ok')
    def hide_obj(self, obj):
        return self.client.request(f'vset /object/{obj}/hide')

    @_check_setcmd_success('ok')
    def hide_objs(self, obj_list):
        cmd_list = [f'vset /object/{obj}/hide' for obj in obj_list]
        return self.client.request(cmd_list)

    @_check_setcmd_success('ok')
    def show_obj(self, obj):
        return self.client.request(f'vset /object/{obj}/show')

    @_check_setcmd_success('ok')
    def show_objs(self, obj_list):
        cmd_list = [f'vset /object/{obj}/show' for obj in obj_list]
        return self.client.request(cmd_list)

    @_check_setcmd_success('ok')
    def create_obj(self, obj_type, obj):
        """
        type:
            'CvCharacter': Human
            'FusionCameraActor': Camera
        Flow:
            For a human object, create -> set mesh -> set mask color
            For a camera object, create -> set location -> set rotation
        """
        return self.client.request(f'vset /objects/spawn {obj_type} {obj}')

    @_check_setcmd_success('ok')
    def set_mask_color(self, obj, rgb):
        return self.client.request('vset /object/{obj}/color {} {} {}'.format(*rgb, obj=obj))

    @_check_setcmd_success(['ok', 'error Can not find object'])
    def destroy_obj(self, obj):
        return self.client.request(f'vset /object/{obj}/destroy')

    @_check_setcmd_success(['ok', 'error Can not find object'])
    def destroy_objs(self, obj_list):
        cmd_list = [f'vset /object/{obj}/destroy' for obj in obj_list]
        return self.client.request(cmd_list)

    def get_obj_location(self, obj):
        location = self.client.request(f'vget /object/{obj}/location')
        return [float(i) for i in location.split()]

    def get_obj_rotation(self, obj):
        rotation = self.client.request(f'vget /object/{obj}/rotation')
        return [float(i) for i in rotation.split()]

    def get_obj_pose(self, obj):
        cmd_list = [
            f'vget /object/{obj}/location',
            f'vget /object/{obj}/rotation',
        ]
        loc, rot = self.client.request(cmd_list)
        loc = list(map(float, loc.split()))
        rot = list(map(float, rot.split()))
        return loc + rot

    def get_objects_pose_dict(self, objects):
        pose_dict = {}
        cmd_list = []
        for obj in objects:
            cmd_list.extend(
                [
                    f'vget /object/{obj}/location',
                    f'vget /object/{obj}/rotation',
                ]
            )
        res = self.client.request(cmd_list)
        res = ' '.join(res)  # join all res string
        res_list = list(map(float, res.split()))
        for idx, obj in enumerate(objects):
            pose_dict[obj] = res_list[idx : idx + 6]
        return pose_dict

    @_check_setcmd_success('ok')
    def set_obj_location(self, obj, loc):
        cmd = 'vset /object/{obj}/location {} {} {}'.format(*loc, obj=obj)
        return self.client.request(cmd)

    @_check_setcmd_success('ok')
    def set_obj_rotation(self, obj, rot):
        # rot: [pitch, yaw, roll]
        cmd = 'vset /object/{obj}/rotation {} {} {}'.format(*rot, obj=obj)
        return self.client.request(cmd)

    def set_obj_scale(self, obj, scale):
        # scale factor: [x, y, z]
        cmd = 'vset /object/{obj}/scale {} {} {}'.format(*scale, obj=obj)
        return self.client.request(cmd)

    def get_obj_scale(self, obj):
        # scale factor: [x, y, z]
        cmd = f'vget /object/{obj}/scale'
        scale = None
        while scale is None:
            scale = self.client.request(cmd)
        scale = list(map(float, scale.split()))
        return scale

    def get_obj_color(self, obj):
        color = self.color_dict.get(obj)
        if color is None:
            object_rgba = self.client.request(f'vget /object/{obj}/color')
            object_rgba = re.findall(r'\d+\.?\d*', object_rgba)
            color = [int(i) for i in object_rgba]  # [r,g,b,a]
            color = color[:-1]  # [r,g,b]
            self.color_dict[obj] = color
        return color

    def get_obj_color_batch(self, obj_list):
        color_list = [self.color_dict.get(obj) for obj in obj_list]

        cmd_list = []
        request_obj_list = []
        for idx, col in enumerate(color_list):
            if col is None:
                cmd_list.append(f'vget /object/{obj_list[idx]}/color')
                request_obj_list.append(obj_list[idx])

        if len(cmd_list) == 0:
            return color_list

        object_rgba_list = self.client.request(cmd_list)
        object_rgb_list = [
            list(map(int, re.findall(r'\d+\.?\d*', object_rgba)))[:-1]
            for object_rgba in object_rgba_list
        ]

        assert len(object_rgb_list) == len(request_obj_list)

        ptr = 0
        for idx in range(len(color_list)):
            if color_list[idx] is None:
                this_color = object_rgb_list[ptr]
                color_list[idx] = this_color
                self.color_dict[request_obj_list[ptr]] = this_color
                ptr += 1

        return color_list

    def _get_bb(self, img_mask, color, margin):
        """
        img_mask: [h,w, 3 or 4] RGB or RGBA
        color: list [R, G, B]
        Return: list of [x1, y1, w, h], maybe negative
        """
        r, g, b = color
        obj_mask = (img_mask[:, :, 0] == r) & (img_mask[:, :, 1] == g) & (img_mask[:, :, 2] == b)
        ys, xs = np.where(obj_mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        w = xmax - xmin
        h = ymax - ymin
        bb = [
            xmin - w * margin,
            ymin - h * margin,
            w * (1 + 2 * margin),
            h * (1 + 2 * margin),
        ]
        bb = [int(v) for v in bb]
        return bb

    def get_bb(self, cam_id, obj, margin=0):
        """
        Return: list of [x1, y1, w, h], maybe negative
        """
        img_mask = self.get_img(cam_id, 'object_mask')  # RGBA
        # img_mask = img_mask[:-1]  # [h, w, c] RGB
        obj_color = self.get_obj_color(obj)  # list
        bb = self._get_bb(img_mask, obj_color, margin)
        return bb

    def get_bb_batch(self, img_mask_list, obj_list, margin=0.1):
        """
        Use this function when img_mask is already in hand.
        Return:
            nested list: bb of [img1_obj1, img2_obj1, ..., img1_obj2, img2_obj2]
        """
        res = []
        for obj in obj_list:
            obj_color = self.get_obj_color(obj)
            for img_mask in img_mask_list:
                bb = self._get_bb(img_mask, obj_color, margin)
                res.append(bb)
        return res

    def _get_kp_visibility_from_mask(self, img_mask, color, kp2d):
        """
        Get kp visibility from mask
        img_mask: [h, w, 3 or 4] RGB or RGBA
        color: list [R, G, B]
        kp2d: [J, 2]
        Return: [J], visible: 1, invisible: 0
        """
        r, g, b = color
        h, w = img_mask.shape[:2]
        vis = np.zeros((kp2d.shape[0],), dtype=np.int32)
        for idx, kp in enumerate(kp2d):
            if 0 <= kp[0] < w and 0 <= kp[1] < h:
                kp_color = img_mask[kp[1], kp[0], :3]
                if np.all(kp_color == color):
                    vis[idx] = 1
                # else:
                #     print(f'kp_color {kp_color}, obj color {color}')
        return vis

    def get_kp_visibility_from_mask(self, img_mask_list, obj_list, proj2d):
        """
        Use this function when img_mask and proj2d is already in hand.
        img_mask_list: [img1, img2, ...]
        obj_list: [human0, human1, human2, ...]
        proj2d: [C, N, J, 2]
        Return: [C, N, J], visible: 1, invisible: 0
        """
        proj2d = np.swapaxes(proj2d, 0, 1).astype(np.int32)  # [N, C, J, 2]
        batch_size, num_cameras = proj2d.shape[:2]
        res = []
        for obj_idx, obj in enumerate(obj_list):
            obj_color = self.get_obj_color(obj)
            for view_idx, img_mask in enumerate(img_mask_list):
                kp2d = proj2d[obj_idx, view_idx]  # [J, 2]
                vsb = self._get_kp_visibility_from_mask(img_mask, obj_color, kp2d)  # [J]
                res.append(vsb)
        res = np.array(res).reshape((batch_size, num_cameras, -1))  # [N, C, J]
        res = np.swapaxes(res, 0, 1)  # [C, N, J]
        return res

    def _get_obj_mask(self, img_mask, color):
        """
        img_mask: [h, w, 3 or 4] RGB or RGBA
        color: list [R, G, B]
        Return: [h, w] binary mask
        """
        r, g, b = color
        obj_mask = (img_mask[:, :, 0] == r) & (img_mask[:, :, 1] == g) & (img_mask[:, :, 2] == b)
        return obj_mask

    def get_obj_mask(self, cam_id, obj):
        """
        Return: [h, w] binary mask
        """
        img_mask = self.get_img(cam_id, 'object_mask')  # RGBA
        obj_color = self.get_obj_color(obj)  # list
        obj_mask = self._get_obj_mask(img_mask, obj_color)
        return obj_mask

    @_check_setcmd_success('ok')
    def set_obj_color(self, obj, color):
        self.color_dict[obj] = color
        cmd = 'vset /object/{obj}/color {r} {g} {b}'.format(
            obj=obj, r=color[0], g=color[1], b=color[2]
        )
        return self.client.request(cmd)

    def init_color_dict(self, objects):
        for obj in objects:
            color = self.get_obj_color(obj)
            self.color_dict[obj] = color
        return self.color_dict

    # camera related ====================================================
    def get_cameras(self):
        cmd = 'vget /cameras'
        res = self.client.request(cmd)
        cam_names = res.split()
        return cam_names

    @_check_setcmd_success('ok')
    def set_cam_loc(self, cam_id, loc):
        # loc: [x, y, z]
        # input loc is compatible with list or ndarray
        self.cam_params[cam_id][:3] = loc
        cmd = 'vset /camera/{cam_id}/location {} {} {}'.format(*loc, cam_id=cam_id)
        return self.client.request(cmd)

    def set_cam_loc_physics(self, cam_id, loc):
        # loc: [x, y, z]
        # input loc is compatible with list or ndarray
        self.cam_params[cam_id][:3] = loc
        cmd = 'vset /camera/{cam_id}/moveto {} {} {}'.format(*loc, cam_id=cam_id)
        return self.client.request(cmd)

    def get_cam_loc(self, cam_id, mode='hard'):
        # return list
        if mode == 'soft':
            return self.cam_params[cam_id][:3].tolist()
        if mode == 'hard':
            cmd = f'vget /camera/{cam_id}/location'
            loc = None
            while loc is None:
                loc = self.client.request(cmd)
            loc = list(map(float, loc.split()))
            # self.cam_params[cam_id][:3] = loc
            return loc

    @_check_setcmd_success('ok')
    def set_cam_rot(self, cam_id, rot):  # rot = [pitch, yaw, roll], inverse of org version
        self.cam_params[cam_id][3:6] = rot
        cmd = 'vset /camera/{cam_id}/rotation {} {} {}'.format(*rot, cam_id=cam_id)
        return self.client.request(cmd)

    def get_cam_rot(self, cam_id, mode='hard'):
        # return list
        if mode == 'soft':
            return self.cam_params[cam_id][3:6].tolist()
        if mode == 'hard':
            cmd = f'vget /camera/{cam_id}/rotation'
            rot = None
            while rot is None:
                rot = self.client.request(cmd)
            rot = list(map(float, rot.split()))
            # self.cam_params[cam_id][3:6] = rot
            return rot

    @_check_setcmd_success('ok')
    def set_fov(self, cam_id, fov):
        cmd = f'vset /camera/{cam_id}/fov {fov}'
        return self.client.request(cmd)

    def get_fov(self, cam_id, fov):
        cmd = f'vget /camera/{cam_id}/fov'
        res = self.client.request(cmd)
        return float(res)

    def get_cam_param(self, cam_id):
        """
        rot, loc : list, [x,y,z] [pitch,yaw,roll]
        fov: float
        """
        cmd_list = [
            'vget /camera/%d/location' % cam_id,
            'vget /camera/%d/rotation' % cam_id,
        ]
        cmd_list.append('vget /camera/%d/fov' % cam_id)
        loc, rot, fov = self.client.request(cmd_list)
        loc = list(map(float, loc.split()))
        rot = list(map(float, rot.split()))
        fov = float(fov)
        return loc, rot, fov

    # not in use, we don't change camera resolution on the fly.
    def set_cam_size(self, cam_id, resolution):
        self.client.request('vset /camera/{cam_id}/size {} {}'.format(*resolution, cam_id=cam_id))

    def set_existing_cam_size(self, resolution):
        """
        resolution: tuple, int, [w, h]
        """
        cam_id_list = list(self.cam_params.keys())
        cmd_list = [
            'vset /camera/{cam_id}/size {} {}'.format(*resolution, cam_id=cam_id)
            for cam_id in cam_id_list
        ]
        cmd_list.append('vset /camera/0/size {} {}'.format(*resolution))
        self.client.request(cmd_list)
        self.resolution = resolution

    # Human related ==========================================
    def get_kp3d(self, human_name):
        # Return ndarray of [J, 3]
        res = self.client.request(f'vget /human/{human_name}/3d_keypoint')
        kp_3d = json.loads(res)
        kp_3d = np.asarray(
            list(
                map(
                    lambda x: [x['KpWorld']['X'], x['KpWorld']['Y'], x['KpWorld']['Z']],
                    kp_3d,
                )
            )
        )
        return kp_3d

    @_check_setcmd_success('ok')
    def set_mesh(self, human_name, mesh_name):
        return self.client.request(f'vset /human/{human_name}/mesh /Game/human_mesh/{mesh_name}')

    # only work for main view point
    # @_check_setcmd_success('ShowFlag.Bones = 1')
    def show_bones(self):
        cmd = 'ShowFlag.Bones 1'
        return self.client.request(cmd)

    # @_check_setcmd_success('ShowFlag.Bones = 0')
    def hide_bones(self):
        cmd = 'ShowFlag.Bones 0'
        return self.client.request(cmd)

    def get_keypoint_names(self, human_name):
        cmd = f'vget /human/{human_name}/3d_keypoint'
        results = self.client.request(cmd)
        kp_3d_list = json.loads(results)
        names = [n['Name'] for n in kp_3d_list]
        return names

    # Anim ==================================================
    def get_anim_length(self, human_name, anim_name):
        # only work for existing human
        # Assume human1 exists
        cmd = f'vget /human/{human_name}/animation/frames /Game/human_anim/{anim_name}'
        return int(self.client.request(cmd))

    @_check_setcmd_success('ok')
    def set_anim_ratio(self, human_name, anim_name, ratio):
        cmd = f'vset /human/{human_name}/animation/ratio /Game/human_anim/{anim_name} {ratio}'
        return self.client.request(cmd)

    def change_to_walk_mode(self, human_name):
        """
        In Anim mode, we can play anim with move to another place, but two legs do not act.
        In Walk mode, we cannot play any anim, but huamn can walk normally.
        """
        cmd = f'vbp {human_name} set_anim_bp'
        return self.client.request(cmd)

    def change_to_anim_mode(self, human_name):
        cmd = f'vbp {human_name} set_anim_asset'
        return self.client.request(cmd)

    def move_to(self, human_name, loc):
        """
        Note that the last value should be the smallest z value for human,
        Usually initial z value for human is OK
        """
        cmd = 'vbp {human_name} move_to {} {} {}'.format(*loc, human_name=human_name)
        return self.client.request(cmd)

    def set_walk_speed(self, human_name, speed):
        """
        50 ~ 200, for me 125 is appropriate
        """
        cmd = f'vbp {human_name} set_speed {speed}'
        return self.client.request(cmd)

    def set_rot_speed(self, human_name, rot_speed):
        """
        200-300 looks normal, may be smaller
        """
        cmd = f'vbp {human_name} set_rot_speed {rot_speed}'
        return self.client.request(cmd)

    def set_walk_param(self, human_name, speed, rot_speed):
        """
        50 ~ 200, for me 125 is appropriate
        """
        cmd = [
            f'vbp {human_name} set_speed {speed}',
            f'vbp {human_name} set_rot_speed {rot_speed}',
        ]
        return self.client.request(cmd)

    def set_dilation_async(self, human_name, dilation):
        cmd = f'vbp {human_name} set_dilation {dilation}'
        return self.client.request_async(cmd)

    # Static func ============================================
    @staticmethod
    def get_joints_visibility_from_mask(obj_name_list, kp2d, img_mask_list):
        """
        obj_name_list : [N]
        kp2d : [C, N, J, 2]
        img_mask_list : [C]
        assert len(kp2d) == len(img_mask_list)
        Return:
            Visibility: [C, N, J]
        """

        # get obj color
        return

    # Special func ===========================================
    def get_floor_location(self):
        loc = self.client.request('vget /object/Floor/location')
        loc = list(map(float, loc.split()))
        return loc

    def request(self, cmd):
        return self.client.request(cmd)

    def request_async(self, cmd):
        return self.client.request_async(cmd)

    def pause_game(self):
        return self.client.request('vset /action/game/pause')

    def resume_game(self):
        return self.client.request('vset /action/game/resume')

    def pause_game_async(self):
        return self.client.request_async('vset /action/game/pause')

    def resume_game_async(self):
        return self.client.request_async('vset /action/game/resume')

    # ================= my func =======================================
    def get_concurrent(self, human_list=[10], camera_list=[0], viewmode_list=['lit']):
        """
        Get 3d keypoint of several targets and imgs from several cameras concurrently
        viewmode: list of ['lit', 'normal', 'object_mask', 'depth']
        Return:
            [kp3d_1, ..., kp3d_n] [param1, param2, param3]
            [[norm1, norm2, norm_m], [mask1, mask2, maskm], [depth1, depth2, depthm]]
            Return is an empty list if corresponding input is null.
        """
        num_human, num_cam, num_vm = (
            len(human_list),
            len(camera_list),
            len(viewmode_list),
        )
        cmd_list = [f'vget /human/human{human_id}/3d_keypoint' for human_id in human_list]
        len_res_list = [num_human]

        cmd_param_list = [
            'vget /camera/%d/%s' % (cam_id, param)
            for cam_id in camera_list
            for param in ['location', 'rotation', 'fov']
        ]
        cmd_list.extend(cmd_param_list)
        len_res_list.append(3 * num_cam)

        cmd_img_list = [
            f'vget /camera/{cam_id}/{viewmode} {self.viewmode_meta[viewmode][0]}'
            for viewmode in viewmode_list
            for cam_id in camera_list
        ]
        cmd_list.extend(cmd_img_list)
        len_res_list.append(num_vm * num_cam)

        # print(cmd_list)
        # start = time.time()
        results = self.client.request(cmd_list)
        # print(f'get data time: {time.time() - start}')
        res_endpoint = np.cumsum(len_res_list).tolist()
        res_endpoint.insert(0, 0)
        # print(res_endpoint)

        kp_3d_list = [json.loads(kp_3d) for kp_3d in results[res_endpoint[0] : res_endpoint[1]]]
        kp_3d_list = [
            np.asarray(
                list(
                    map(
                        lambda x: [
                            x['KpWorld']['X'],
                            x['KpWorld']['Y'],
                            x['KpWorld']['Z'],
                        ],
                        kp_3d,
                    )
                )
            )
            for kp_3d in kp_3d_list
        ]

        param_list = []
        for i, cam_id in zip(range(res_endpoint[1], res_endpoint[2], 3), camera_list):
            param_str = ' '.join(results[i : i + 3])
            # print(param_str)
            param = list(map(float, param_str.split()))
            param_list.append(param)

            # update camera params here, x,y,z,pitch,yaw,roll,fov
            if cam_id in self.cam_params:
                self.cam_params[cam_id][:] = param

        img_list = []
        for idx, viewmode in enumerate(viewmode_list):
            start = idx * num_cam
            end = (idx + 1) * num_cam
            img_list.append(
                [
                    self.viewmode_meta[viewmode][1](img)
                    for img in results[res_endpoint[2] + start : res_endpoint[2] + end]
                ]
            )

        return kp_3d_list, param_list, img_list

    def create_cameras(self, cam_id_list=[11, 12, 13, 14]):
        # no check, return value is camera name
        res_list = self.client.request(
            ['vget /camera/%d/location' % cam_id for cam_id in cam_id_list]
        )
        cmd_create_cam = [
            'vset /objects/spawn FusionCameraActor %d' % cam_id_list[idx]
            for idx in range(len(res_list))
            if 'error' in res_list[idx]
        ]
        if len(cmd_create_cam) > 0:
            self.client.request(cmd_create_cam)
        for cam_id in cam_id_list:
            self.cam_params[cam_id] = np.zeros((7,), dtype=np.float32)  # xyz,pitch,yaw,roll,fov

    def create_cameras_async(
        self, create_cam_id_list=[11, 12, 13, 14], all_cam_id_list=[11, 12, 13, 14]
    ):
        # no check, return value is camera name
        # res_list = self.client.request(['vget /camera/%d/location' % cam_id for cam_id in cam_id_list])
        assert len(all_cam_id_list) >= len(create_cam_id_list)

        cmd_create_cam = [
            'vset /objects/spawn FusionCameraActor %d' % cam_id for cam_id in create_cam_id_list
        ]

        if len(cmd_create_cam) > 0:
            self.client.request_async(cmd_create_cam)

        for cam_id in all_cam_id_list:
            self.cam_params[cam_id] = np.zeros((7,), dtype=np.float32)  # xyz,pitch,yaw,roll,fov

    @_check_setcmd_success(['ok', 'error Can not find object'])
    def destroy_existing_cameras(self):
        """
        Only destroy newly added cameras
        cameras hung on the human are considered
        """
        cam_id_list = list(self.cam_params.keys())
        self.cam_params = {}
        cmd_destroy_cam = [f'vset /object/{cam_id}/destroy' for cam_id in cam_id_list]
        if len(cmd_destroy_cam) > 0:
            return self.client.request(cmd_destroy_cam)
        else:
            return 'ok'

    @_check_setcmd_success('ok')
    def update_all_camera_parameters(self, param_dict):
        # update all camera parameters, initialize all cameras
        # {cam_id: np.ndarray([x,y,z,pitch,yaw,roll,fov]), ...}
        cmd_list = []
        for k, v in param_dict.items():
            cmd_list.extend(
                [
                    'vset /camera/{id}/location {} {} {}'.format(*v[:3], id=k),
                    'vset /camera/{id}/rotation {} {} {}'.format(*v[3:6], id=k),
                    f'vset /camera/{k}/fov {v[-1]}',
                ]
            )
            self.cam_params[k][:] = v
        return self.client.request(cmd_list)

    def update_all_camera_parameters_async(self, param_dict):
        # update all camera parameters, initialize all cameras
        # {cam_id: np.ndarray([x,y,z,pitch,yaw,roll,fov]), ...}
        cmd_list = []
        for k, v in param_dict.items():
            cmd_list.extend(
                [
                    'vset /camera/{id}/location {} {} {}'.format(*v[:3], id=k),
                    'vset /camera/{id}/rotation {} {} {}'.format(*v[3:6], id=k),
                    f'vset /camera/{k}/fov {v[-1]}',
                ]
            )
            self.cam_params[k][:] = v
        return self.client.request_async(cmd_list)

    @_check_setcmd_success('ok')
    def control_camera(self, param_dict):
        """
        Pass in Incremental value of pitch, yaw, fov
        If exceeding boundary, then do what? !!!!!!!!!!! keep consistent with env.value?

        pitch: -89.9 - 89.9, env exceeds clip
        yaw: -180 - 180, env exceeds auto change
        When use yaw as camera rot description, change to uniform range manually in advance.
        """

        cmd_list = []
        for k, v in param_dict.items():
            self.cam_params[k][[3, 4, 6]] += v  # What to do if exceeds boundary !!!!!!
            cmd_list.extend(
                [
                    'vset /camera/{id}/rotation {} {} 0'.format(*self.cam_params[k][3:5], id=k),
                    f'vset /camera/{k}/fov {self.cam_params[k][-1]}',
                ]
            )
        res = self.client.request(cmd_list)

        # update cam_params according to env param values
        cmd_list = []
        for k in param_dict.keys():
            cmd_list.extend(
                [
                    f'vget /camera/{k}/rotation',
                    f'vget /camera/{k}/fov',
                ]
            )

        update_value = None
        while update_value is None:
            update_value = self.client.request(cmd_list)

        for idx, k in enumerate(param_dict.keys()):
            rot, fov = update_value[idx * 2 : idx * 2 + 2]
            rot = list(map(float, rot.split()))
            fov = float(fov)
            self.cam_params[k][[3, 4, 6]] = [rot[0], rot[1], fov]

        return res

    # @_check_setcmd_success('ok')
    def rotate_camera(self, param_dict):
        """
        Pass in Incremental value of pitch, yaw
        If exceeding boundary, then do what? !!!!!!!!!!! keep consistent with env.value?

        pitch: -89.9 - 89.9, env exceeds clip
        yaw: -180 - 180, env exceeds auto change
        When use yaw as camera rot description, change to uniform range manually in advance.
        """

        cmd_list = []
        for k, v in param_dict.items():
            self.cam_params[k][[3, 4]] += v  # What to do if exceeds boundary !!!!!!
            cmd_list.extend(
                ['vset /camera/{id}/rotation {} {} 0'.format(*self.cam_params[k][3:5], id=k)]
            )
        res = self.client.request(cmd_list)

        # # update cam_params according to env param values
        # cmd_list = []
        # for k in param_dict.keys():
        #     cmd_list.extend(['vget /camera/{id}/rotation'.format(id=k)])

        # update_value = None
        # while update_value is None:
        #     update_value = self.client.request(cmd_list)

        # for idx, k in enumerate(param_dict.keys()):
        #     rot = update_value[idx]
        #     rot = list(map(float, rot.split()))
        #     self.cam_params[k][[3, 4]] = rot[:2]

        return res

    # @_check_setcmd_success('ok')
    def move_camera(self, param_dict):
        # Pass in Incremental value of x, y, z
        # If exceeding boundary, then do what? !!!!!!!!!!! keep consistent with env.value?
        cmd_list = []
        for k, v in param_dict.items():
            self.cam_params[k][:3] += v
            cmd_list.extend(
                ['vset /camera/{id}/location {} {} {}'.format(*self.cam_params[k][:3], id=k)]
            )
        res = self.client.request(cmd_list)

        # # update cam_params according to env param values
        # cmd_list = []
        # for k in param_dict.keys():
        #     cmd_list.extend(['vget /camera/{id}/location'.format(id=k)])

        # update_value = None
        # while update_value is None:
        #     update_value = self.client.request(cmd_list)

        # for idx, k in enumerate(param_dict.keys()):
        #     loc = update_value[idx]
        #     loc = list(map(float, loc.split()))
        #     self.cam_params[k][:3] = loc

        return res

    # @_check_setcmd_success('ok')
    def rotate_and_move_camera(self, param_dict):
        """
        param_dict:
            {
                'move': {cam_id: delta_xyz},
                'rotate': {cam_id: delta_pitch,yaw}
            }

        Error will accumulate in this step, when camera rotate to boundary or collided
        Call get_concurrent() after calling this func, it updates self.cam_params.
        """
        cmd_list = []
        for mode, mode_dict in param_dict.items():
            if mode == 'move_policy':
                for k, v in mode_dict.items():
                    self.cam_params[k][:3] += v
                    cmd_list.extend(
                        [
                            'vset /camera/{id}/moveto {:.2f} {:.2f} {:.2f}'.format(
                                *self.cam_params[k][:3], id=k
                            )
                        ]
                    )
            elif mode == 'rotate_policy':
                for k, v in mode_dict.items():
                    self.cam_params[k][[3, 4]] += v  # What to do if exceeds boundary !!!!!!
                    if np.abs(np.abs(self.cam_params[k][3]) - 90) < 1:
                        if self.cam_params[k][3] < 0:
                            self.cam_params[k][3] = -89
                        else:
                            self.cam_params[k][3] = 89
                    cmd_list.extend(
                        [
                            'vset /camera/{id}/rotation {:.2f} {:.2f} 0'.format(
                                *self.cam_params[k][3:5], id=k
                            )
                        ]
                    )
        res = self.client.request(cmd_list)
        return res

    def rotate_and_move_camera_async(self, param_dict):
        """
        param_dict:
            {
                'move': {cam_id: delta_xyz},
                'rotate': {cam_id: delta_pitch,yaw}
            }

        Error will accumulate in this step, when camera rotate to boundary or collided
        Call get_concurrent() after calling this func, it updates self.cam_params.
        """
        cmd_list = []
        for mode, mode_dict in param_dict.items():
            if mode == 'move_policy':
                for k, v in mode_dict.items():
                    self.cam_params[k][:3] += v
                    cmd_list.extend(
                        [
                            'vset /camera/{id}/moveto {:.2f} {:.2f} {:.2f}'.format(
                                *self.cam_params[k][:3], id=k
                            )
                        ]
                    )
            elif mode == 'rotate_policy':
                for k, v in mode_dict.items():
                    self.cam_params[k][[3, 4]] += v  # What to do if exceeds boundary !!!!!!
                    if np.abs(np.abs(self.cam_params[k][3]) - 90) < 1:
                        if self.cam_params[k][3] < 0:
                            self.cam_params[k][3] = -89
                        else:
                            self.cam_params[k][3] = 89
                    cmd_list.extend(
                        [
                            'vset /camera/{id}/rotation {:.2f} {:.2f} 0'.format(
                                *self.cam_params[k][3:5], id=k
                            )
                        ]
                    )
        res = self.client.request_async(cmd_list)
        return res
