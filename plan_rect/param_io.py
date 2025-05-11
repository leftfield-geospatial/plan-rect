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
from typing import Any

from fsspec.core import OpenFile
from orthority.common import Open


def write_rectification_data(
    file: str | PathLike | OpenFile,
    src_name: str,
    im_size: tuple[int, int],
    markers: list[dict[str, Any]],
    overwrite: bool = False,
):
    """
    Write camera parameters and markers to a NedCAD format file.

    :param file:
        File to write.  Can be a path or URI string,
        an :class:`~fsspec.core.OpenFile` object or a file object, opened in text
        mode (``'wt'``).
    :param src_name:
        Source image file name.
    :param im_size:
        Image (width, height) in pixels.
    :param markers:
        Markers as a list of dictionaries with ``id``: <marker name>, and ``ji``:
        <pixel coordinate> items.
    :param overwrite:
        Whether to overwrite the file if it exists.
    """
    with Open(file, 'wt', overwrite=overwrite) as f:
        f.write(f'Photo;{src_name}\n')
        f.write(f'Size;{im_size[0]},{im_size[1]};px\n')

        for m in markers:
            f.write(f'{m["id"]};{m["ji"][0]:.4f},{m["ji"][1]:.4f},0\n')
