# Copyright The Plan-Rect Contributors.
#
# This file is part of Plan-Rect.
#
# Plan-Rect is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# Plan-Rect is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with
# Plan-Rect. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import numpy as np
from orthority.camera import Camera


class PerspectiveCamera(Camera):
    """
    Perspective camera for projecting between planes.

    :param im_size:
        Image (width, height) in pixels.
    :param tform:
        Perspective transform from world to pixel coordinates, as a 3-by-3 array.
    """

    def __init__(self, im_size: tuple[int, int], tform: np.ndarray):
        self._im_size = im_size
        self._tform = tform

    def world_to_pixel(self, xyz: np.ndarray) -> np.ndarray:
        xyz_ = np.vstack((xyz[:2], np.ones((1, xyz.shape[1]))))
        ji_ = self._tform.dot(xyz_)
        ji = ji_[:2] / ji_[2]
        return ji

    def pixel_to_world_z(self, ji: np.ndarray, z: float | np.ndarray) -> np.ndarray:
        # only allow projecting to the world z=0 plane
        assert np.all(z == 0)
        ji_ = np.vstack((ji, np.ones((1, ji.shape[1]))))
        xyz = np.linalg.inv(self._tform).dot(ji_)
        xyz /= xyz[2]
        xyz[2] = 0
        return xyz
