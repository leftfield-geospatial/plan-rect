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

import posixpath
import warnings
from functools import partial
from pathlib import Path

import click
import cv2
import fsspec
import numpy as np
import rasterio as rio
from fsspec.core import OpenFile
from orthority import param_io
from orthority.common import OpenRaster, join_ofile
from orthority.enums import Interp
from rasterio.errors import NotGeoreferencedWarning

from plan_rect.camera import PerspectiveCamera
from plan_rect.param_io import write_rectification_data
from plan_rect.rectify import rectify
from plan_rect.version import __version__


def _file_cb(
    ctx: click.Context, param: click.Parameter, path_uri: str, mode: str = 'rb'
) -> OpenFile:
    """Click callback to convert a file path / URI to an OpenFile instance."""
    ofile = None
    if path_uri:
        try:
            ofile = fsspec.open(path_uri, mode)
        except Exception as ex:
            raise click.BadParameter(str(ex), param=param) from None
    return ofile


def _resolution_cb(
    ctx: click.Context, param: click.Parameter, resolution: tuple
) -> tuple[float, float]:
    """Click callback to validate and parse the resolution."""
    if len(resolution) == 1:
        resolution *= 2
    elif len(resolution) > 2:
        raise click.BadParameter(
            'At most two resolution values should be specified.', param=param
        )
    return resolution


def _dir_cb(ctx: click.Context, param: click.Parameter, uri_path: str) -> OpenFile:
    """Click callback to validate a directory path / URI, and convert it to an
    OpenFile instance.
    """
    try:
        ofile = fsspec.open(uri_path)
    except Exception as ex:
        raise click.BadParameter(str(ex)) from ex

    # isdir() requires a trailing slash on some file systems (e.g. gcs)
    if not ofile.fs.isdir(posixpath.join(ofile.path, '')):
        raise click.BadParameter(
            f"'{uri_path}' is not a directory or cannot be accessed."
        )
    return ofile


@click.command()
@click.option(
    '-im',
    '--image',
    'image_file',
    type=click.Path(dir_okay=False),
    required=True,
    default=None,
    callback=partial(_file_cb, mode='rb'),
    help='Path / URI of the source image.',
)
@click.option(
    '-m',
    '--marker',
    'markers',
    type=(str, float, float, float, float),
    metavar='ID X Y COL ROW',
    default=(),
    multiple=True,
    help='Marker ID and location in world and pixel coordinates, with pixel '
    'coordinate origin at the bottom left image corner.',
)
@click.option(
    '-g',
    '--gcp',
    'gcp_file',
    type=click.Path(dir_okay=False),
    default=None,
    callback=partial(_file_cb, mode='rt'),
    help='Path / URI of an Orthority GCP file defining marker locations.',
)
@click.option(
    '-r',
    '--res',
    'resolution',
    type=click.FLOAT,
    default=None,
    show_default='ground sampling distance',
    multiple=True,
    callback=_resolution_cb,
    help='Rectified pixel size in meters.  Can be used twice for non-square pixels: '
    "'--res WIDTH --res HEIGHT'.",
)
@click.option(
    '-i',
    '--interp',
    type=click.Choice(Interp, case_sensitive=True),
    default=Interp.cubic,
    show_default=True,
    help='Interpolation method for remapping source to rectified image.',
)
@click.option(
    '-n',
    '--nodata',
    type=click.FLOAT,
    default=None,
    show_default='auto',
    help='Nodata value for the rectified image.',
)
@click.option(
    '-ep',
    '--export-params',
    is_flag=True,
    type=click.BOOL,
    default=False,
    show_default=True,
    help='Export markers to an Orthority GCP file and exit.',
)
@click.option(
    '-od',
    '--out-dir',
    type=click.Path(file_okay=False),
    default=str(Path.cwd()),
    show_default='current working',
    callback=_dir_cb,
    help='Path / URI of the output file directory.',
)
@click.option(
    '-o',
    '--overwrite',
    is_flag=True,
    type=click.BOOL,
    default=False,
    show_default=True,
    help='Overwrite existing output(s).',
)
@click.version_option(version=__version__, message='%(version)s')
@click.pass_context
def cli(
    ctx: click.Context,
    image_file: OpenFile,
    markers: tuple[tuple[str, float, float, float, float]],
    gcp_file: OpenFile,
    export_params: bool,
    out_dir: OpenFile,
    overwrite: bool,
    **kwargs,
):
    """Rectify an image onto a plane.

    Marker locations are required with either '-m' / '--marker' or '-g' / '--gcp'.
    The '-m' / '--marker' option can be provided multiple times.  At least four
    markers are required.
    """
    # silence not georeferenced warnings
    warnings.simplefilter('ignore', category=NotGeoreferencedWarning)
    # enter rasterio environment
    ctx.with_resource(rio.Env(GDAL_NUM_THREADS='ALL_CPUS', GTIFF_FORCE_RGBA=False))

    # form a GCP dictionary
    image_path = Path(image_file.path)
    if not markers and not gcp_file:
        raise click.UsageError(
            "Marker locations are required with either '-m' / '--marker' or '-g' / "
            "'--gcp'."
        )
    with OpenRaster(image_file, 'r') as src_im:
        im_size = src_im.shape[::-1]

    if markers:
        # convert marker locations to GCPs with pixel coordinates converted from BL to
        # TL origin convention
        gcps = [
            dict(
                id=m[0],
                ji=(m[3], im_size[1] - 1 - m[4]),
                xyz=(m[1], m[2], 0.0),
                info=None,
            )
            for m in markers
        ]
        gcp_dict = {image_path.name: gcps}
    else:
        gcp_dict = param_io.read_oty_gcps(gcp_file)
        # limit the GCP dictionary to image_file's GCPs
        gcps = gcp_dict.get(image_path.name) or gcp_dict.get(image_path.stem)
        if not gcps:
            raise click.BadParameter(
                f"There are no GCPs for '{image_path.name}' in '"
                f"{Path(gcp_file.path).name}'",
                param_hint="'-g' / '--gcp'.",
            )
        gcp_dict = {image_path.name: gcps}

    if len(gcps) < 4:
        raise click.UsageError('At least four markers are required.')

    if export_params:
        # export interior parameters and GCPs as orthority format files, and exit
        gcp_file = join_ofile(out_dir, 'gcps.geojson', mode='wt')
        param_io.write_gcps(gcp_file, gcp_dict, overwrite=overwrite)
        click.echo(f"Orthority GCP file written to: '{gcp_file.path}'.")
        return

    # fit perspective transform & create camera
    gcp_ji = np.array([gcp['ji'] for gcp in gcps]).T
    gcp_xyz = np.array([gcp['xyz'] for gcp in gcps]).T
    tform, _ = cv2.findHomography(
        gcp_xyz[:2].T.astype('float32'), gcp_ji.T.astype('float32')
    )
    cam = PerspectiveCamera(im_size, tform)

    # find & print fitting error
    cam_ji = cam.world_to_pixel(gcp_xyz)
    err = np.sqrt(np.sum((gcp_ji - cam_ji) ** 2, axis=0)).mean()
    click.echo(f'RMS fitting error: {err:.4e} (pixels).')

    # rectify
    rect_array, transform = rectify(image_file, cam, **kwargs)

    # write rectified image
    profile = dict(
        driver='png',
        transform=transform,
        width=rect_array.shape[2],
        height=rect_array.shape[1],
        count=rect_array.shape[0],
        dtype=rect_array.dtype,
    )
    rect_im_file = join_ofile(out_dir, 'rect.png', mode='wb')
    with OpenRaster(rect_im_file, 'w', overwrite=overwrite, **profile) as rect_ds:
        rect_ds.write(rect_array)

    # find pixel coordinates of markers in the rectified image (0.5 translation
    # shifts the transform from the GDAL / Rasterio convention to give integer
    # coordinates that refer to pixel centers)
    inv_transform = ~(rio.Affine(*transform) * rio.Affine.translation(0.5, 0.5))
    rect_ji = np.array(inv_transform * gcp_xyz[:2])

    # create a rectified marker list, converting pixel coordinates from TL to BL
    # origin convention
    rect_ji[1] = rect_array.shape[1] - 1 - rect_ji[1]
    rect_markers = [
        dict(id=gcp.get('id'), ji=ji_mkr)
        for i, (gcp, ji_mkr) in enumerate(zip(gcps, rect_ji.T))
    ]

    # write rectification data
    rect_data_file = join_ofile(out_dir, 'pixeldata.txt', mode='wt')
    write_rectification_data(
        rect_data_file, image_path.name, im_size, rect_markers, overwrite=overwrite
    )
    click.echo(f"Output files written to: '{out_dir.path}'.")


if __name__ == '__main__':
    cli()
