import numpy as np


def generate_random_points(shape, points):
    """
    shape: [N, 2] vertices
    N:
        3 points: triangles
        4 points: rectangles
        >4 points: polygons, decompose into triangles, then sample in triangles
    """
    assert len(points) > 2, 'area must be a polygon with more than 3 vertices'
    if shape == 'triangle':
        assert len(points) == 3
        res = points_on_triangle(points, 1).squeeze(0)  # [2]
    elif shape == 'rectangle':
        assert len(points) >= 4
        x_min, y_min = np.amin(points, axis=0)
        x_max, y_max = np.amax(points, axis=0)
        res = np.array((x_max - x_min, y_max - y_min)) * np.random.rand(2)
        res = res + np.array((x_min, y_min))
    elif shape == 'covex_polygon':
        # points should be in order
        assert len(points) >= 4
        triangles = []
        for idx in range(1, len(points) - 1):
            triangles.append(np.array((points[0], points[idx], points[idx + 1])))
        areas = list(map(get_area, triangles))  # N-2
        p = np.array(areas) / np.sum(areas)
        tri_idx = np.random.choice(len(p), size=1, p=p)
        res = points_on_triangle(triangles[tri_idx], 1).squeeze(0)  # [2]
    elif shape == 'polygon':
        assert 0, 'not implemented yet!'
    return res


def points_on_triangle(v, n):
    """
    Give n random points uniformly on a triangle.
    The vertices of the triangle are given by the shape
    v: [N, 2]
    Return: [N, 2]
    """
    x = np.sort(np.random.rand(2, n), axis=0)
    return np.column_stack([x[0], x[1] - x[0], 1.0 - x[1]]) @ v


def get_area(points):
    """
    points: [3, 2]
    """
    area = 0.5 * (
        points[0][0] * (points[1][1] - points[2][1])
        + points[1][0] * (points[2][1] - points[0][1])
        + points[2][0] * (points[0][1] - points[1][1])
    )
    return area


class PointSampler:
    def __init__(self, shape, points):
        """
        Random sample points in a given area.
        """
        self.shape = shape
        self.points = points
        assert len(points) >= 1, 'area must have more than 1 vertex'

        if shape == 'triangle':
            assert len(points) == 3
            # res = points_on_triangle(points, 1).squeeze(0)  # [2]
        elif shape == 'rectangle':
            assert len(points) >= 4
            x_min, y_min = np.amin(points, axis=0)
            x_max, y_max = np.amax(points, axis=0)
            self.edge = np.array((x_max - x_min, y_max - y_min))
            self.lower_bound = np.array((x_min, y_min))
            # res = np.array((x_max - x_min, y_max - y_min)) * np.random.rand(2)
            # res = res + np.array((x_min, y_min))
        elif shape == 'covex_polygon':
            # points should be in order
            assert len(points) >= 4
            self.triangles = []
            for idx in range(1, len(points) - 1):
                self.triangles.append(np.array((points[0], points[idx], points[idx + 1])))
            areas = list(map(get_area, self.triangles))  # N-2
            self.p = np.array(areas) / np.sum(areas)
            # tri_idx = np.random.choice(len(p), size=1, p=p)
            # res = points_on_triangle(triangles[tri_idx], 1).squeeze(0)  # [2]
        elif shape == 'polygon':
            raise NotImplementedError
        elif shape == 'points':
            assert len(points) >= 1
            # sample from points
        else:
            assert 0, f'unkonwn shape: {shape:s}'

    def sample(self):
        if self.shape == 'triangle':
            res = points_on_triangle(self.points, 1).squeeze(0)  # [2]
        elif self.shape == 'rectangle':
            res = self.edge * np.random.rand(2) + self.lower_bound
        elif self.shape == 'covex_polygon':
            tri_idx = np.random.choice(len(self.points), size=1, p=self.p)
            res = points_on_triangle(self.triangles[tri_idx], 1).squeeze(0)  # [2]
        elif self.shape == 'polygon':
            raise NotImplementedError
        elif self.shape == 'points':
            idx = np.random.randint(0, len(self.points))
            res = self.points[idx]
        return res


class TrajectorySampler:
    def __init__(self, points):
        """
        Random sample points in a given area.
        """
        assert len(points) >= 1, 'area must have more than 1 vertex'
        self.points = points

        self.ptr = 0
        self.count = 0  # count trajectories

    def sample(self):
        res = self.points[self.ptr]
        self.ptr = (self.ptr + 1) % len(self.points)
        self.count += 1

        return res

    def is_finished(self):
        """
        1->2->3->4->1(finished)
        """
        if len(self.points) > 1:
            if self.ptr == 1 and self.count != self.ptr:
                return True
            else:
                return False
        else:
            if self.count > 1:
                return True
            else:
                return False

    def reset(self):
        self.ptr = 0
        self.count = 0
