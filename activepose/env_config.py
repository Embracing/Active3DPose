import numpy as np


map_config_dict = {
    'Blank': {
        'map_center': np.array([0, 0, 0], dtype=np.float32),
        'human_z': 86.15,
        # 'changable_mesh': True,
    },
    'SchoolGymDay': {
        'map_center': np.array([-1120, 330, 130], dtype=np.float32),
        'human_z': 216.15,  # specific to env, not human model, wierd
        # 'changable_mesh': False,
    },
    'Building': {
        'map_center': np.array([430, -270, -30], dtype=np.float32),
        'human_z': 56.15,  # specific to env, not human model, wierd
        # 'changable_mesh': False,
    },
    'Building_small': {
        'map_center': np.array([430, -270, -30], dtype=np.float32),
        'human_z': 56.15,  # specific to env, not human model, wierd
        # 'changable_mesh': False,
    },
    'grass_plane': {
        'map_center': np.array([0, 0, -104.8], dtype=np.float32),
        'human_z': -18.7,  # specific to env, not human model, wierd
        # 'changable_mesh': True,
    },
}

human_model_config_dict = {
    'Default': {
        'changable_mesh': True,
    },
    'Sportman': {
        'changable_mesh': False,
    },
    'Businessman': {
        'changable_mesh': False,  # can be true, howeverneed to change anim_class accordingly
    },
    'grass_plane': {
        'changable_mesh': True,
    },
}

training_setups = {
    'C2_training': {
        'env_description': '',
        'area': np.array([[500.0, 500.0], [500.0, -500.0], [-500.0, -500.0], [-500.0, 500.0]]),
        'area_shape': 'rectangle',  # used to sample points, sample rate = 0.9
        'num_cameras': 2,
        'camera_param_list': [
            np.array((150.0, 500.0, 300.0, -35.0, -135.0, 0, 90.0), dtype=np.float32),
            np.array((-150.0, 500.0, 300.0, -35.0, -45.0, 0, 90.0), dtype=np.float32),
        ],
        'human_mesh_list': [f'ch{i}' for i in range(1, 7)],
        'floor_z_of_human': 0,  # z location for human object
        'lower_bound_for_camera': [-650, -650, 90],  # x, y, z, camera move area
        'higher_bound_for_camera': [650, 650, 300],  # x, y, z
    },
    'C3_training': {
        'env_description': '',
        'area': np.array([[500.0, 500.0], [500.0, -500.0], [-500.0, -500.0], [-500.0, 500.0]]),
        'area_shape': 'rectangle',  # used to sample points, sample rate = 0.9
        'num_cameras': 3,
        'camera_param_list': [
            np.array((150.0, 500.0, 300.0, -35.0, -135.0, 0, 90.0), dtype=np.float32),
            np.array((-150.0, 500.0, 300.0, -35.0, -45.0, 0, 90.0), dtype=np.float32),
            np.array((0, -300.0, 300.0, -35.0, 90.0, 0, 90.0), dtype=np.float32),
        ],
        'human_mesh_list': [f'ch{i}' for i in range(1, 7)],
        'floor_z_of_human': 0,  # z location for human object
        'lower_bound_for_camera': [-650, -650, 90],  # x, y, z, camera move area
        'higher_bound_for_camera': [650, 650, 300],  # x, y, z
    },
    'C4_training': {
        'env_description': '10X10 square, cameras height: 3.0m, with a pitch of 35',
        'area': np.array([[500.0, 500.0], [500.0, -500.0], [-500.0, -500.0], [-500.0, 500.0]]),
        'area_shape': 'rectangle',  # used to sample points, sample rate = 0.9
        'num_cameras': 4,
        'camera_param_list': [
            np.array((500.0, 500.0, 300.0, -35.0, -135.0, 0, 90.0), dtype=np.float32),
            np.array((500.0, -500.0, 300.0, -35.0, 135.0, 0, 90.0), dtype=np.float32),
            np.array((-500.0, -500.0, 300.0, -35.0, 45.0, 0, 90.0), dtype=np.float32),
            np.array((-500.0, 500.0, 300.0, -35.0, -45.0, 0, 90.0), dtype=np.float32),
        ],
        'human_mesh_list': [f'ch{i}' for i in range(1, 7)],
        'floor_z_of_human': 0,  # z location for human object
        'lower_bound_for_camera': [-650, -650, 90],  # x, y, z, camera move area
        'higher_bound_for_camera': [650, 650, 300],  # x, y, z
    },
    'C5_training': {
        'env_description': '10X10 square, cameras height: 3.0m, with a pitch of 35',
        'area': np.array([[500.0, 500.0], [500.0, -500.0], [-500.0, -500.0], [-500.0, 500.0]]),
        'area_shape': 'rectangle',  # used to sample points, sample rate = 0.9
        'num_cameras': 5,
        'camera_param_list': [
            np.array((0.0, -500.0, 300.0, -35.0, 90.0, 0, 90.0), dtype=np.float32),
            np.array((475.5, -154.5, 300.0, -35.0, 162.0, 0, 90.0), dtype=np.float32),
            np.array((293.9, 404.5, 300.0, -35.0, 234.0, 0, 90.0), dtype=np.float32),
            np.array((-293.9, 404.5, 300.0, -35.0, 306.0, 0, 90.0), dtype=np.float32),
            np.array((-475.5, -154.5, 300.0, -35.0, 18.0, 0, 90.0), dtype=np.float32),
        ],
        'human_mesh_list': [f'ch{i}' for i in range(1, 7)],
        'floor_z_of_human': 90.15,  # z location for human object
        'lower_bound_for_camera': [-650, -650, 90],  # x, y, z, camera move area
        'higher_bound_for_camera': [650, 650, 300],  # x, y, z
    },
}

special_setups = {
    'C6_10x10_c3h30_c3h15_dual_triangle': {
        'env_description': '10X10 square, cameras height: 3.0m, with a pitch of 35',
        'area': np.array([[500.0, 500.0], [500.0, -500.0], [-500.0, -500.0], [-500.0, 500.0]]),
        'area_shape': 'rectangle',  # used to sample points, sample rate = 0.9
        'num_cameras': 6,
        'camera_param_list': [  # x,y,z,p,y,r,f
            np.array((0.0, -500.0, 300.0, -35.0, 90.0, 0, 90.0), dtype=np.float32),
            np.array(
                (433.0 + 100, -250.0 - 100, 150.0, 0.0, 150.0, 0, 90.0),
                dtype=np.float32,
            ),
            np.array((433.0, 250.0, 300.0, -35.0, 210.0, 0, 90.0), dtype=np.float32),
            np.array((0.0, 500.0 + 100, 150.0, 0.0, 270.0, 0, 90.0), dtype=np.float32),
            np.array((-433.0, 250.0, 300.0, -35.0, 330.0, 0, 90.0), dtype=np.float32),
            np.array(
                (-433.0 - 100, -250.0 - 100, 150.0, 0.0, 30.0, 0, 90.0),
                dtype=np.float32,
            ),
        ],
        'human_mesh_list': [f'ch{i}' for i in range(1, 7)],
        'floor_z_of_human': 90.15,  # z location for human object
        'lower_bound_for_camera': [-650, -650, 90],  # x, y, z, camera move area
        'higher_bound_for_camera': [650, 650, 300],  # x, y, z
    },
    'C16_10x10_dual_octagon': {
        'env_description': '10X10 square, cameras height: 3.0m, with a pitch of 35',
        'area': np.array([[500.0, 500.0], [500.0, -500.0], [-500.0, -500.0], [-500.0, 500.0]]),
        'area_shape': 'rectangle',  # used to sample points, sample rate = 0.9
        'num_cameras': 8 * 2,
        'camera_param_list': [
            np.array((0.0, -500.0, 300.0, -35.0, 90.0, 0, 90.0), dtype=np.float32),
            np.array((354.0, -354.0, 300.0, -35.0, 135.0, 0, 90.0), dtype=np.float32),
            np.array((500.0, 0.0, 300.0, -35.0, 180.0, 0, 90.0), dtype=np.float32),
            np.array((354.0, 354.0, 300.0, -35.0, 225.0, 0, 90.0), dtype=np.float32),
            np.array((0.0, 500.0, 300.0, -35.0, 270.0, 0, 90.0), dtype=np.float32),
            np.array((-354.0, 354.0, 300.0, -35.0, 315.0, 0, 90.0), dtype=np.float32),
            np.array((-500.0, 0.0, 300.0, -35.0, 0.0, 0, 90.0), dtype=np.float32),
            np.array((-354.0, -354.0, 300.0, -35.0, 45.0, 0, 90.0), dtype=np.float32),
            np.array((0.0, -719.0, 150.0, 0.0, 90.0, 0, 90.0), dtype=np.float32),
            np.array((508.0, -508.0, 150.0, 0.0, 135.0, 0, 90.0), dtype=np.float32),
            np.array((719.0, 0.0, 150.0, 0.0, 180.0, 0, 90.0), dtype=np.float32),
            np.array((508.0, 508.0, 150.0, 0.0, 225.0, 0, 90.0), dtype=np.float32),
            np.array((0.0, 719.0, 150.0, 0.0, 270.0, 0, 90.0), dtype=np.float32),
            np.array((-508.0, 508.0, 150.0, 0.0, 315.0, 0, 90.0), dtype=np.float32),
            np.array((-719.0, 0.0, 150.0, 0.0, 0.0, 0, 90.0), dtype=np.float32),
            np.array((-508.0, -508.0, 150.0, 0.0, 45.0, 0, 90.0), dtype=np.float32),
        ],
        'human_mesh_list': [f'ch{i}' for i in range(1, 7)],
        'floor_z_of_human': 90.15,  # z location for human object
        'lower_bound_for_camera': [-1000, -1000, 90],  # x, y, z, camera move area
        'higher_bound_for_camera': [1000, 1000, 300],  # x, y, z
    },
    'C16_10x10_two_rings': {
        'env_description': '10X10 square, cameras height: 3.0m, with a pitch of 35',
        'area': np.array([[500.0, 500.0], [500.0, -500.0], [-500.0, -500.0], [-500.0, 500.0]]),
        'area_shape': 'rectangle',  # used to sample points, sample rate = 0.9
        'num_cameras': 8 * 2,
        'camera_param_list': [
            np.array((0.0, -500.0, 300.0, -35.0, 90.0, 0, 90.0), dtype=np.float32),
            np.array((354.0, -354.0, 300.0, -35.0, 135.0, 0, 90.0), dtype=np.float32),
            np.array((500.0, 0.0, 300.0, -35.0, 180.0, 0, 90.0), dtype=np.float32),
            np.array((354.0, 354.0, 300.0, -35.0, 225.0, 0, 90.0), dtype=np.float32),
            np.array((0.0, 500.0, 300.0, -35.0, 270.0, 0, 90.0), dtype=np.float32),
            np.array((-354.0, 354.0, 300.0, -35.0, 315.0, 0, 90.0), dtype=np.float32),
            np.array((-500.0, 0.0, 300.0, -35.0, 0.0, 0, 90.0), dtype=np.float32),
            np.array((-354.0, -354.0, 300.0, -35.0, 45.0, 0, 90.0), dtype=np.float32),
            np.array((179.0, -433.0, 250.0, -45.0, 112.5, 0, 90.0), dtype=np.float32),
            np.array((433.0, -179.0, 250.0, -45.0, 157.5, 0, 90.0), dtype=np.float32),
            np.array((433.0, 179.0, 250.0, -45.0, 202.5, 0, 90.0), dtype=np.float32),
            np.array((179.0, 433.0, 250.0, -45.0, 247.5, 0, 90.0), dtype=np.float32),
            np.array((-179.0, 433.0, 250.0, -45.0, 292.5, 0, 90.0), dtype=np.float32),
            np.array((-433.0, 179.0, 250.0, -45.0, 337.5, 0, 90.0), dtype=np.float32),
            np.array((-433.0, -179.0, 250.0, -45.0, 22.5, 0, 90.0), dtype=np.float32),
            np.array((-179.0, -433.0, 250.0, -45.0, 67.5, 0, 90.0), dtype=np.float32),
        ],
        'human_mesh_list': [f'ch{i}' for i in range(1, 7)],
        'floor_z_of_human': 90.15,  # z location for human object
        'lower_bound_for_camera': [-1000, -1000, 90],  # x, y, z, camera move area
        'higher_bound_for_camera': [1000, 1000, 300],  # x, y, z
    },
    'C16_10x10_dual_octagon_high_outer': {
        'env_description': '10X10 square, cameras height: 3.0m, with a pitch of 35',
        'area': np.array([[500.0, 500.0], [500.0, -500.0], [-500.0, -500.0], [-500.0, 500.0]]),
        'area_shape': 'rectangle',  # used to sample points, sample rate = 0.9
        'num_cameras': 8 * 2,
        'camera_param_list': [
            np.array((0.0, -500.0, 300.0, -35.0, 90.0, 0, 90.0), dtype=np.float32),
            np.array((354.0, -354.0, 300.0, -35.0, 135.0, 0, 90.0), dtype=np.float32),
            np.array((500.0, 0.0, 300.0, -35.0, 180.0, 0, 90.0), dtype=np.float32),
            np.array((354.0, 354.0, 300.0, -35.0, 225.0, 0, 90.0), dtype=np.float32),
            np.array((0.0, 500.0, 300.0, -35.0, 270.0, 0, 90.0), dtype=np.float32),
            np.array((-354.0, 354.0, 300.0, -35.0, 315.0, 0, 90.0), dtype=np.float32),
            np.array((-500.0, 0.0, 300.0, -35.0, 0.0, 0, 90.0), dtype=np.float32),
            np.array((-354.0, -354.0, 300.0, -35.0, 45.0, 0, 90.0), dtype=np.float32),
            np.array((0.0, -719.0, 400.0, -30.0, 90.0, 0, 90.0), dtype=np.float32),
            np.array((508.0, -508.0, 400.0, -30.0, 135.0, 0, 90.0), dtype=np.float32),
            np.array((719.0, 0.0, 400.0, -30.0, 180.0, 0, 90.0), dtype=np.float32),
            np.array((508.0, 508.0, 400.0, -30.0, 225.0, 0, 90.0), dtype=np.float32),
            np.array((0.0, 719.0, 400.0, -30.0, 270.0, 0, 90.0), dtype=np.float32),
            np.array((-508.0, 508.0, 400.0, -30.0, 315.0, 0, 90.0), dtype=np.float32),
            np.array((-719.0, 0.0, 400.0, -30.0, 0.0, 0, 90.0), dtype=np.float32),
            np.array((-508.0, -508.0, 400.0, -30.0, 45.0, 0, 90.0), dtype=np.float32),
        ],
        'human_mesh_list': [f'ch{i}' for i in range(1, 7)],
        'floor_z_of_human': 90.15,  # z location for human object
        'lower_bound_for_camera': [-1000, -1000, 90],  # x, y, z, camera move area
        'higher_bound_for_camera': [1000, 1000, 300],  # x, y, z
    },
    'C24_10x10_three_rings': {
        'env_description': '10X10 square, cameras height: 3.0m, with a pitch of 35',
        'area': np.array([[500.0, 500.0], [500.0, -500.0], [-500.0, -500.0], [-500.0, 500.0]]),
        'area_shape': 'rectangle',  # used to sample points, sample rate = 0.9
        'num_cameras': 8 * 3,
        'camera_param_list': [
            np.array((0.0, -500.0, 300.0, -35.0, 90.0, 0, 90.0), dtype=np.float32),
            np.array((354.0, -354.0, 300.0, -35.0, 135.0, 0, 90.0), dtype=np.float32),
            np.array((500.0, 0.0, 300.0, -35.0, 180.0, 0, 90.0), dtype=np.float32),
            np.array((354.0, 354.0, 300.0, -35.0, 225.0, 0, 90.0), dtype=np.float32),
            np.array((0.0, 500.0, 300.0, -35.0, 270.0, 0, 90.0), dtype=np.float32),
            np.array((-354.0, 354.0, 300.0, -35.0, 315.0, 0, 90.0), dtype=np.float32),
            np.array((-500.0, 0.0, 300.0, -35.0, 0.0, 0, 90.0), dtype=np.float32),
            np.array((-354.0, -354.0, 300.0, -35.0, 45.0, 0, 90.0), dtype=np.float32),
            np.array((179.0, -433.0, 250.0, -45.0, 112.5, 0, 90.0), dtype=np.float32),
            np.array((433.0, -179.0, 250.0, -45.0, 157.5, 0, 90.0), dtype=np.float32),
            np.array((433.0, 179.0, 250.0, -45.0, 202.5, 0, 90.0), dtype=np.float32),
            np.array((179.0, 433.0, 250.0, -45.0, 247.5, 0, 90.0), dtype=np.float32),
            np.array((-179.0, 433.0, 250.0, -45.0, 292.5, 0, 90.0), dtype=np.float32),
            np.array((-433.0, 179.0, 250.0, -45.0, 337.5, 0, 90.0), dtype=np.float32),
            np.array((-433.0, -179.0, 250.0, -45.0, 22.5, 0, 90.0), dtype=np.float32),
            np.array((-179.0, -433.0, 250.0, -45.0, 67.5, 0, 90.0), dtype=np.float32),
            np.array((125.0, -302.0, 400.0, -50.0, 112.5, 0, 90.0), dtype=np.float32),
            np.array((302.0, -125.0, 400.0, -50.0, 157.5, 0, 90.0), dtype=np.float32),
            np.array((302.0, 125.0, 400.0, -50.0, 202.5, 0, 90.0), dtype=np.float32),
            np.array((125.0, 302.0, 400.0, -50.0, 247.5, 0, 90.0), dtype=np.float32),
            np.array((-125.0, 302.0, 400.0, -50.0, 292.5, 0, 90.0), dtype=np.float32),
            np.array((-302.0, 125.0, 400.0, -50.0, 337.5, 0, 90.0), dtype=np.float32),
            np.array((-302.0, -125.0, 400.0, -50.0, 22.5, 0, 90.0), dtype=np.float32),
            np.array((-125.0, -302.0, 400.0, -50.0, 67.5, 0, 90.0), dtype=np.float32),
        ],
        'human_mesh_list': [f'ch{i}' for i in range(1, 7)],
        'floor_z_of_human': 90.15,  # z location for human object
        'lower_bound_for_camera': [-1000, -1000, 90],  # x, y, z, camera move area
        'higher_bound_for_camera': [1000, 1000, 300],  # x, y, z
    },
}

human_config_dict = {
    # 'walk_speed_range': [20, 30],  # [10, 20], [50, 75]
    # 'rot_speed_range': [80, 100],  # [80, 100]
    1: {
        'human_id_list': [1],
        'human_location_list': [[0, 0, 0]],
        'human_rotation_list': [[0, 0, 0]],
        'mask_color': [[126, 12, 250]],  # RGB
        'scale': [1],
        'action': ['walk'],
    },
    2: {
        'human_id_list': [1, 2],
        'human_location_list': [[0, 0, 0], [60, 60, 0]],
        'human_rotation_list': [[0, 0, 0], [0, 0, 0]],
        'mask_color': [[126, 12, 250], [1, 17, 250]],  # RGB
        'scale': [1, 1],
        'action': ['walk' for _ in range(2)],
    },
    3: {
        'human_id_list': [1, 2, 3],
        # 'human_location_list': [[0, 0, 0], [60, 60, 0], [120, 120, 0]],
        'human_location_list': [[0, 0, 0], [180, 180, 0], [-180, 180, 0]],
        'human_rotation_list': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        'mask_color': [[126, 12, 250], [1, 17, 250], [12, 153, 250]],  # RGB
        'scale': [1, 1, 1],
        'action': ['walk' for _ in range(3)],
    },
    4: {
        'human_id_list': [1, 2, 3, 4],
        'human_location_list': [
            [0, 0, 0],
            [-200, 200, 0],
            [-200, -200, 0],
            [200, 0, 0],
        ],
        'human_rotation_list': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        'mask_color': [
            [126, 12, 250],
            [1, 17, 250],
            [12, 153, 250],
            [250, 249, 25],
        ],  # RGB
        'scale': [1, 1, 1, 1],
        'action': ['walk' for _ in range(4)],
    },
    5: {
        'human_id_list': [1, 2, 3, 4, 5],
        'human_location_list': [
            [0, 0, 0],
            [0, 400, 0],
            [0, -400, 0],
            [400, 0, 0],
            [-400, 0, 0],
        ],
        'human_rotation_list': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        'mask_color': [
            [126, 12, 250],
            [1, 17, 250],
            [12, 153, 250],
            [250, 249, 25],
            [250, 158, 25],
        ],  # RGB
        'scale': [1, 1, 1, 1, 1],
        'action': ['walk' for _ in range(5)],
    },
    6: {
        'human_id_list': [1, 2, 3, 4, 5, 6],
        'human_location_list': [
            [0, 0, 0],
            [-300, 300, 0],
            [-300, -300, 0],
            [300, -300, 0],
            [300, 300, 0],
            [-180, 0, 0],
        ],
        'human_rotation_list': [[0, 0, 0] for _ in range(6)],
        'mask_color': [
            [126, 12, 250],
            [1, 17, 250],
            [12, 153, 250],
            [250, 249, 25],
            [250, 158, 25],
            [12, 250, 226],
        ],  # RGB
        'scale': [1, 1, 1, 1, 1, 1],
        'action': ['walk' for _ in range(6)],
    },
    7: {
        'human_id_list': [1, 2, 3, 4, 5, 6, 7],
        'human_location_list': [
            [0, 300, 0],
            [-300, 300, 0],
            [-300, -300, 0],
            [300, -300, 0],
            [300, 300, 0],
            [-180, 0, 0],
            [180, 0, 0],
        ],
        'human_rotation_list': [[0, 0, 0] for _ in range(7)],
        'mask_color': [
            [126, 12, 250],
            [1, 17, 250],
            [12, 153, 250],
            [250, 249, 25],
            [250, 158, 25],
            [12, 250, 226],
            [0, 250, 82],
        ],  # RGB
        'scale': [1, 1, 1, 1, 1, 1, 1],
        'action': ['walk' for _ in range(7)],
    },
    10: {
        'human_id_list': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'human_location_list': [
            [0, 0, 0],
            [-240, -240, 0],
            [-180, -180, 0],
            [-120, -120, 0],
            [-60, -60, 0],
            [60, 60, 0],
            [120, 120, 0],
            [180, 180, 0],
            [240, 240, 0],
            [300, 300, 0],
        ],
        'human_rotation_list': [[0, 0, 0] for _ in range(10)],
        'mask_color': [
            [250, 249, 25],
            [139, 0, 139],
            [255, 0, 255],
            [238, 130, 238],
            [153, 50, 204],
            [147, 112, 219],
            [120, 12, 250],
            [119, 12, 250],
            [118, 12, 250],
            [117, 12, 250],
        ],
        # pedestrian all purple
        # 'mask_color': [[136, 250, 12], [126, 12, 250], [139, 37, 250], [152, 62, 250], [165, 87, 250],
        #                [178, 112, 250], [191, 137, 250], [119, 12, 250], [118, 12, 250], [117, 12, 250]],
        # 'mask_color': [[126, 12, 250], [1, 17, 250], [12, 153, 250], [250, 249, 25], [250, 158, 25],
        #                [12, 250, 226], [0, 250, 82], [81, 250, 12], [250, 96, 25], [250, 25, 230]],  # RGB
        'scale': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'action': ['walk' for _ in range(10)],
    },
}


env_config_dict = training_setups | special_setups


def get_human_config(num_humans=None):
    if num_humans is None or num_humans <= 0:
        config_dict = np.random.choice(list(human_config_dict.values()))
        num_humans = len(config_dict['human_id_list'])
    else:
        config_dict = human_config_dict[num_humans]

    # config_dict['walk_speed_range'] = walk_speed_range if walk_speed_range is not None else human_config_dict['walk_speed_range']
    # config_dict['rot_speed_range'] = human_config_dict['rot_speed_range']
    config_dict['num_humans'] = num_humans
    return num_humans, config_dict


def get_env_config(env_name=None):
    if env_name is None:
        return np.random.choice(list(env_config_dict.items()))
    else:
        return env_name, env_config_dict[env_name]


def get_map_config(map_name='default'):
    return map_config_dict[map_name]


def get_human_model_config(model_name='default'):
    return model_name, human_model_config_dict[model_name]


def baseline_env_builder(C, S, H=2, P=0.35):
    key = f'Baseline_C{C}_{S}x{S}_h{H}0_p35'
    size = S * 100.0 / 2
    height = H * 100.0
    pitch = P * 100.0
    D = {
        'env_description': f'{S}X{S} square, cameras height: {H}.0m, with a pitch of {pitch}',
        'area': np.array([[size, size], [size, -size], [-size, -size], [-size, size]]),
        'area_shape': 'rectangle',  # used to sample points, sample rate = 0.9
        'num_cameras': C,
        'human_mesh_list': [f'ch{i}' for i in range(1, 7)],
        'floor_z_of_human': 0,  # z location for human object
        'lower_bound_for_camera': [
            -size - 150.0,
            -size - 150.0,
            90,
        ],  # x, y, z, camera move area
        'higher_bound_for_camera': [size + 150.0, size + 150.0, 300],  # x, y, z
    }

    if C == 2:
        D['camera_param_list'] = [
            np.array((size, size, height, -pitch, -135.0, 0, 90.0), dtype=np.float32),
            np.array((-size, size, height, -pitch, -45.0, 0, 90.0), dtype=np.float32),
        ]
    elif C == 3:
        D['camera_param_list'] = [
            np.array((0, -size, height, -pitch, 90.0, 0, 90.0), dtype=np.float32),
            np.array((-size, size, height, -pitch, -45.0, 0, 90.0), dtype=np.float32),
            np.array((size, size, height, -pitch, -135.0, 0, 90.0), dtype=np.float32),
        ]
    elif C == 4:
        D['camera_param_list'] = [
            np.array((size, size, height, -pitch, -135.0, 0, 90.0), dtype=np.float32),
            np.array((size, -size, height, -pitch, 135.0, 0, 90.0), dtype=np.float32),
            np.array((-size, -size, height, -pitch, 45.0, 0, 90.0), dtype=np.float32),
            np.array((-size, size, height, -pitch, -45.0, 0, 90.0), dtype=np.float32),
        ]

    return {key: D}


for c in [2, 3, 4]:
    for s in [10]:
        for h in [3]:
            env_config_dict.update(baseline_env_builder(c, s, h))
