from collections import OrderedDict


# h36m dataset skeletion, 17 joints
neighbour_dict_set_3d = [
    # ################### 0 ##############################
    # original linkage
    {
        0: [1, 4, 7],
        1: [0, 7, 2],
        2: [1, 3],
        3: [2],
        4: [0, 7, 5],
        5: [4, 6],
        6: [5],
        7: [1, 0, 4, 14, 8, 11],
        8: [7, 9, 11, 14],
        9: [8, 10],
        10: [9],
        11: [8, 7, 12],
        12: [11, 13],
        13: [12],
        14: [8, 7, 15],
        15: [14, 16],
        16: [15],
    },
]

# mpii dataset skeleton, 16 joints
neighbour_dict_set_2d_16 = [
    {
        0: [1],
        1: [0, 2],
        2: [1, 6],
        3: [4, 6],
        4: [3, 5],
        5: [4],
        6: [2, 3, 7],
        7: [6, 8, 12, 13],
        8: [7, 9],
        9: [8],
        10: [11],
        11: [10, 12],
        12: [7, 11],
        13: [7, 14],
        14: [13, 15],
        15: [14],
    }
]

# mpii dataset skeleton, 16+2(foot) joints
neighbour_dict_set_2d_18 = [
    {
        0: [1, 16],
        1: [0, 2],
        2: [1, 6],
        3: [4, 6],
        4: [3, 5],
        5: [4, 17],
        6: [2, 3, 7],
        7: [6, 8, 12, 13],
        8: [7, 9],
        9: [8],
        10: [11],
        11: [10, 12],
        12: [7, 11],
        13: [7, 14],
        14: [13, 15],
        15: [14],
        16: [0],
        17: [5],
    }
]

# mpii dataset skeleton, 16+6(foot) joints
neighbour_dict_set_2d_22 = [
    {
        0: [1, 16, 17, 18],
        1: [0, 2],
        2: [1, 6],
        3: [4, 6],
        4: [3, 5],
        5: [4, 19, 20, 21],
        6: [2, 3, 7],
        7: [6, 8, 12, 13],
        8: [7, 9],
        9: [8],
        10: [11],
        11: [10, 12],
        12: [7, 11],
        13: [7, 14],
        14: [13, 15],
        15: [14],
        16: [0],
        17: [0],
        18: [0],
        19: [5],
        20: [5],
        21: [5],
    }
]

joints_names_22_dict = {
    0: 'rank',
    1: 'rkne',
    2: 'rhip',
    3: 'lhip',
    4: 'lkne',
    5: 'lank',
    6: 'root',
    7: 'thorax',
    8: 'upper neck',
    9: 'head top',
    10: 'rwri',
    11: 'relb',
    12: 'rsho',
    13: 'lsho',
    14: 'lelb',
    15: 'lwri',
    16: 'rbtoe',
    17: 'rstoe',
    18: 'rheel',
    19: 'lheel',
    20: 'lstoe',
    21: 'lbtoe',
}

# unreal human skeleton related
# mpii - 7:thorax
neighbour_dict_set_2d_15 = [
    {
        0: [1],
        1: [0, 2],
        2: [1, 6],
        3: [4, 6],
        4: [3, 5],
        5: [4],
        6: [2, 3, 8],
        7: [6, 8, 11, 12],
        8: [7],
        9: [10],
        10: [11],
        11: [7, 10],
        12: [7, 13],
        13: [12, 14],
        14: [13],
    }
]

# exclude 7--thorax, 16:22--foot, total:15
pred22_to_unreal_id_dict = {
    0: 83,
    1: 82,
    2: 81,
    3: 74,
    4: 75,
    5: 76,
    6: 1,
    8: 6,
    9: 7,
    10: 51,
    11: 50,
    12: 49,
    13: 23,
    14: 24,
    15: 25,
}

# total: 15
used_unreal_joints_dict = OrderedDict(
    [
        (83, 'foot_r'),
        (82, 'calf_r'),
        (81, 'thigh_r'),
        (74, 'thigh_l'),
        (75, 'calf_l'),
        (76, 'foot_l'),
        (1, 'pelvis'),
        (6, 'Head'),
        (7, 'head_end'),
        (51, 'hand_r'),
        (50, 'lowerarm_r'),
        (49, 'upperarm_r'),
        (23, 'upperarm_l'),
        (24, 'lowerarm_l'),
        (25, 'hand_l'),
    ]
)

# COCO-WholeBody, starts from index #0
coco_wholebody_dict = {
    'head': [i for i in range(5)],
    'body': [i for i in range(5, 17)],
    'left_foot': [i for i in range(17, 20)],
    'right_foot': [i for i in range(20, 23)],
    'face': [i for i in range(23, 91)],
    'left_hand': [i for i in range(91, 112)],
    'right_hand': [i for i in range(112, 133)],
}

# total:54, 12 body, 21 left hand, 21 right hand.
# (coco whobody #0, unreal_id)
body_list = [
    (no, unreal_id)
    for no, unreal_id in zip(range(5, 17), [23, 49, 24, 50, 25, 51, 74, 81, 75, 82, 76, 83])
]
left_hand_list = [(i + 91, i + 25) for i in range(21)]
right_hand_list = [(i + 112, i + 51) for i in range(21)]
head_list = [(1, 11), (2, 13)]
# whole_body_list = body_list + left_hand_list + right_hand_list
# whole_body_list = body_list
whole_body_list = body_list + head_list

wholebody_to_unreal_id_dict = OrderedDict(whole_body_list)
body_idx_after_mapping = list(range(12))
lhand_idx_after_mapping = list(range(12, 12 + 21))
rhand_idx_after_mapping = list(range(12 + 21, 12 + 21 * 2))
head_idx_after_mapping = list(range(12, 12 + 2))

wholebody_kpinfo_dict = {
    'mapping_dict_wholebody2unreal': wholebody_to_unreal_id_dict,
    'body_idx_after_mapping': body_idx_after_mapping,
    'lhand_idx_after_mapping': lhand_idx_after_mapping,
    'rhand_idx_after_mapping': rhand_idx_after_mapping,
    'head_idx_after_mapping': head_idx_after_mapping,
}


# 14 joints, 12 body + 2eyes
# bone skeleton plot corresponds to this
coco_k1 = list(wholebody_to_unreal_id_dict.keys())

# unreal corresponding joints
unreal_v1 = list(wholebody_to_unreal_id_dict.values())  # tingyun human model
unreal_v2 = [13, 34, 14, 35, 15, 36, 81, 87, 82, 88, 84, 90, 60, 63]  # sport NPC
unreal_v3 = [27, 6, 28, 7, 29, 8, 61, 55, 62, 56, 64, 58, 51, 52]  # businessman 74j
unreal_v4 = [6, 27, 7, 28, 8, 29, 56, 62, 57, 63, 59, 65, 54, 55]  # businessman 75j

mapping_tuple = (
    coco_k1,
    {
        'Default': unreal_v1,
        'Sportman': unreal_v2,
        'Businessman': {74: unreal_v3, 75: unreal_v4},
    },
)


def get_unreal_dummy_joint_names():
    return [
        'ik_hand_gun',
        'ik_hand_l',
        'ik_hand_r',
        'Root',
        'ik_foot_r',
        'ik_foot_l',
        'ik_foot_root',
        'ik_hand_root',
    ]


def get_skeleton(name):
    res = {}
    skeleton = []
    if name == 'LCN_3D' or name == 'LCN_2D_17':
        neighbour_dict_set_3d = neighbour_dict_set_3d
        for k, v in neighbour_dict_set_3d[0].items():
            for neighbour in v:
                if neighbour > k:
                    skeleton.append([k, neighbour])
        res['skeleton'] = skeleton
        res['joints_right'] = [1, 2, 3, 14, 15, 16]
        res['joints_left'] = [4, 5, 6, 11, 12, 13]
    elif name == 'LCN_2D_16':
        neighbour_dict_set_2d = neighbour_dict_set_2d_16
        for k, v in neighbour_dict_set_2d[0].items():
            for neighbour in v:
                if neighbour > k:
                    skeleton.append([k, neighbour])
        res['skeleton'] = skeleton
        res['joints_right'] = [2, 1, 0, 12, 11, 10]
        res['joints_left'] = [3, 4, 5, 13, 14, 15]
        res['parents'] = {
            2: 6,
            1: 2,
            0: 1,
            3: 6,
            4: 3,
            5: 4,
            12: 7,
            11: 12,
            10: 11,
            13: 7,
            14: 13,
            15: 14,
        }
        res['ankle_right'] = [0]
        res['ankle_left'] = [5]
        res['hand_right'] = [10]
        res['hand_left'] = [15]
    elif name == 'LCN_2D_18':
        # 16 + 2 (foot) joints
        neighbour_dict_set_2d = neighbour_dict_set_2d_18
        for k, v in neighbour_dict_set_2d[0].items():
            for neighbour in v:
                if neighbour > k:
                    skeleton.append([k, neighbour])
        res['skeleton'] = skeleton
        res['joints_right'] = [2, 1, 0, 12, 11, 10, 16]
        res['joints_left'] = [3, 4, 5, 13, 14, 15, 17]
        res['parents'] = {
            2: 6,
            1: 2,
            0: 1,
            3: 6,
            4: 3,
            5: 4,
            12: 7,
            11: 12,
            10: 11,
            13: 7,
            14: 13,
            15: 14,
            16: 0,
            17: 5,
        }
        res['ankle_right'] = [0]
        res['ankle_left'] = [5]
        res['hand_right'] = [10]
        res['hand_left'] = [15]
        res['tiptoe_right'] = [16]
        res['tiptoe_left'] = [17]
    elif name == 'LCN_2D_22':
        # 16 + 6 (foot) joints
        neighbour_dict_set_2d = neighbour_dict_set_2d_22
        for k, v in neighbour_dict_set_2d[0].items():
            for neighbour in v:
                if neighbour > k:
                    skeleton.append([k, neighbour])
        res['skeleton'] = skeleton
        res['joints_right'] = [2, 1, 0, 12, 11, 10, 16, 17, 18]
        res['joints_left'] = [3, 4, 5, 13, 14, 15, 19, 20, 21]
        res['parents'] = {
            2: 6,
            1: 2,
            0: 1,
            3: 6,
            4: 3,
            5: 4,
            12: 7,
            11: 12,
            10: 11,
            13: 7,
            14: 13,
            15: 14,
            16: 0,
            17: 0,
            18: 0,
            19: 5,
            20: 5,
            21: 5,
        }
        res['ankle_right'] = [0]
        res['ankle_left'] = [5]
        res['hand_right'] = [10]
        res['hand_left'] = [15]
        res['bigtoe_right'] = [16]
        res['smalltoe_right'] = [17]
        res['heel_right'] = [18]
        res['bigtoe_left'] = [21]
        res['smalltoe_left'] = [20]
        res['heel_left'] = [19]
    elif name == 'LCN_2D_15':
        neighbour_dict_set_2d = neighbour_dict_set_2d_15
        for k, v in neighbour_dict_set_2d[0].items():
            for neighbour in v:
                if neighbour > k:
                    skeleton.append([k, neighbour])
        res['skeleton'] = skeleton
        res['joints_right'] = [2, 1, 0, 9, 10, 11]
        res['joints_left'] = [3, 4, 5, 12, 13, 14]
        res['parents'] = {
            2: 6,
            1: 2,
            0: 1,
            3: 6,
            4: 3,
            5: 4,
            6: 7,
            7: 8,
            9: 10,
            10: 11,
            11: 7,
            12: 7,
            13: 12,
            14: 13,
        }
        res['ankle_right'] = [0]
        res['ankle_left'] = [5]
        res['hand_right'] = [9]
        res['hand_left'] = [14]
    elif name == 'COCO-WholeBody':
        skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
            [16, 18],
            [16, 19],
            [16, 20],
            [17, 21],
            [17, 22],
            [17, 23],
            [92, 93],
            [93, 94],
            [94, 95],
            [95, 96],
            [92, 97],
            [97, 98],
            [98, 99],
            [99, 100],
            [92, 101],
            [101, 102],
            [102, 103],
            [103, 104],
            [92, 105],
            [105, 106],
            [106, 107],
            [107, 108],
            [92, 109],
            [109, 110],
            [110, 111],
            [111, 112],
            [113, 114],
            [114, 115],
            [115, 116],
            [116, 117],
            [113, 118],
            [118, 119],
            [119, 120],
            [120, 121],
            [113, 122],
            [122, 123],
            [123, 124],
            [124, 125],
            [113, 126],
            [126, 127],
            [127, 128],
            [128, 129],
            [113, 130],
            [130, 131],
            [131, 132],
            [132, 133],
            [10, 92],
            [11, 113],
        ]
        valid_id = sorted(wholebody_to_unreal_id_dict.keys())
        res['skeleton'] = [
            [valid_id.index(a - 1), valid_id.index(b - 1)]
            for a, b in skeleton
            if a - 1 in valid_id and b - 1 in valid_id
        ]
        res['joints_right'] = [
            valid_id.index(i)
            for i in range(133)
            if i in coco_wholebody_dict['right_hand']
            or (i in coco_wholebody_dict['body'] and i % 2 == 0)
        ]
        res['joints_left'] = [
            valid_id.index(i)
            for i in range(133)
            if i in coco_wholebody_dict['left_hand']
            or (i in coco_wholebody_dict['body'] and i % 2 == 1)
        ]
        res['joints_left_hand'] = [
            valid_id.index(i) for i in range(133) if i in coco_wholebody_dict['left_hand']
        ]
        res['joints_right_hand'] = [
            valid_id.index(i) for i in range(133) if i in coco_wholebody_dict['right_hand']
        ]
        res.update(coco_wholebody_dict)
    elif name == 'COCO-WholeBody-onlybody':
        skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
        ]
        valid_id = list(wholebody_to_unreal_id_dict.keys())
        res['skeleton'] = [
            [valid_id.index(a - 1), valid_id.index(b - 1)]
            for a, b in skeleton
            if a - 1 in valid_id and b - 1 in valid_id
        ]
        res['joints_right'] = [
            valid_id.index(i) for i in range(133) if i in coco_wholebody_dict['body'] and i % 2 == 0
        ]
        res['joints_left'] = [
            valid_id.index(i) for i in range(133) if i in coco_wholebody_dict['body'] and i % 2 == 1
        ]
        res.update(coco_wholebody_dict)
    elif name == 'COCO-WholeBody-body+head':
        skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [6, 2],
            [7, 3],
            [2, 3],
        ]  # in whole-body order, start from index #1,
        valid_id = list(wholebody_to_unreal_id_dict.keys())
        res['skeleton'] = [
            [valid_id.index(a - 1), valid_id.index(b - 1)]
            for a, b in skeleton
            if a - 1 in valid_id and b - 1 in valid_id
        ]
        res['joints_right'] = [
            valid_id.index(i) for i in range(133) if i in coco_wholebody_dict['body'] and i % 2 == 0
        ]
        res['joints_left'] = [
            valid_id.index(i) for i in range(133) if i in coco_wholebody_dict['body'] and i % 2 == 1
        ]
        res.update(coco_wholebody_dict)
    else:
        assert 0, f'invalid skeleton type {name:s}'
    return res
