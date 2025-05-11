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

from os import PathLike

import click
import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from orthority.camera import Camera
from orthority.common import OpenRaster
from orthority.enums import Interp


def _area_poly(coords: np.ndarray) -> float:
    """Area of the polygon defined by (x, y) ``coords`` with (x, y) along the 2nd
    dimension.
    """
    # uses "shoelace formula": https://en.wikipedia.org/wiki/Shoelace_formula
    return 0.5 * np.abs(
        coords[:, 0].dot(np.roll(coords[:, 1], -1))
        - np.roll(coords[:, 0], -1).dot(coords[:, 1])
    )


def rectify(
    src_file: str | PathLike | OpenFile,
    camera: Camera,
    resolution: tuple[float, float] | None = None,
    interp: str | Interp = Interp.cubic,
    nodata: int | float | None = None,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """
    Rectify an image onto the plane at z=0.

    :param src_file:
        Source image to rectify.  Can be a path or URI string, an
        :class:`~fsspec.core.OpenFile` object in binary mode (``'rb'``), or a dataset
        reader.
    :param camera:
        Source image camera model.
    :param resolution:
        Rectified image pixel (x, y) size in meters.  Defaults to an approximate
        ground sampling distance when set to ``None``.
    :param interp:
        Interpolation method for remapping the source to rectified image.
    :param nodata:
        Nodata value of the rectified image.

    :return:
        - Rectified image as NumPy array with bands along the first dimension.
        - Georeferencing transform of the rectified image.
    """
    # find a default resolution when not supplied
    xy = camera.world_boundary(z=0)[:2]
    if not resolution:
        ji = camera.pixel_boundary()
        pixel_area = _area_poly(ji.T)
        world_area = _area_poly(xy.T)
        gsd = np.sqrt(world_area / pixel_area)
        resolution = (gsd, gsd)
        click.echo(f'Using auto resolution: {gsd:.4e} (m)')

    # create grids for remapping
    bounds = np.array([*xy.min(axis=1), *xy.max(axis=1)])
    # find x,y ranges that include all pixels on, or inside the bounds
    nxy = np.floor((bounds[2:] - bounds[:2]) / resolution) + 1
    x = bounds[0] + np.arange(nxy[0]) * resolution[0]
    # orient the y grid so that it increases bottom to top
    y = bounds[3] - np.arange(nxy[1]) * resolution[1]
    xgrid, ygrid = np.meshgrid(x, y, indexing='xy')
    zgrid = np.zeros_like(xgrid)

    # create a georeferencing transform for the rectified image (-0.5 translation
    # shifts the transform to the GDAL / Rasterio convention where integer
    # coordinates refer to pixel TL corners)
    transform = rio.Affine(resolution[0], 0, bounds[0], 0, -resolution[1], bounds[3])
    transform *= rio.Affine.translation(-0.5, -0.5)

    # remap
    with OpenRaster(src_file, 'r') as src_im:
        src_array = src_im.read()
    if nodata is None:
        # set the default nodata value based on the image data type
        dtype = src_array.dtype
        nodata = (
            np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else float('nan')
        )
    rect_array, _ = camera.remap(
        src_array, xgrid, ygrid, zgrid, interp=interp, nodata=nodata
    )

    return rect_array, transform[:6]
