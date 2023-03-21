from .base import WrapperBase


# from .utils import bind


class PartialTriangulation(WrapperBase):
    """
    Can be placed in any order
    1: Stationary scene.
    """

    def __init__(
        self,
        env,
    ):
        super().__init__(env)

        env.unwrapped.partial_triangulation = True
