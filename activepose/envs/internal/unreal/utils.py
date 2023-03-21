import numpy as np


def get_view_depth(depth_img, projected_2d, depth, thresh=15, neighbor=2):
    """
    View depth is not accurate. Since there is quantization error for img resolution.
    Input should exlcude points out of the view
    """
    projected_2d_int = np.rint(projected_2d).astype(np.int32)
    # projected_2d[45] = (418, 193)
    depth_img = depth_img.squeeze(-1)
    view_depth = depth_img[projected_2d_int[:, 1], projected_2d_int[:, 0]]  # [N]
    view_depth = view_depth.copy()

    # if diff < 0, take the wrong point.
    diff = depth - view_depth
    odd_point_idx = (diff < -5).nonzero()[0]
    # print(odd_point_idx)

    if len(odd_point_idx) > 0:
        height, width = depth_img.shape
        # odd_projected_2d = projected_2d[odd_point_idx]
        odd_projected_2d_int = projected_2d_int[odd_point_idx]
        print(odd_projected_2d_int)

        lower_bound = np.clip(odd_projected_2d_int - neighbor, 0, None)
        x_higher = np.clip(odd_projected_2d_int[:, 0] + neighbor + 1, None, width)
        y_higher = np.clip(odd_projected_2d_int[:, 1] + neighbor + 1, None, height)
        for idx, org_idx in enumerate(odd_point_idx):
            values = depth_img[
                lower_bound[idx, 1] : y_higher[idx], lower_bound[idx, 0] : x_higher[idx]
            ]
            value = np.amin(values)
            print('================')
            print(projected_2d[org_idx])
            print(view_depth[org_idx], value)
            print(values)
            view_depth[org_idx] = value

    vis = np.abs(depth - view_depth) < thresh

    return view_depth, vis
