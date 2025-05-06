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

import string
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio as rio
from click.testing import CliRunner
from orthority.camera import FrameCamera, create_camera
from orthority.enums import CameraType, Interp
from orthority.param_io import (
    read_oty_gcps,
    read_oty_int_param,
    write_gcps,
    write_int_param,
)

from plan_rect.cli import cli
from plan_rect.param_io import write_rectification_data
from plan_rect.rectify import rectify


def get_gcps(camera: FrameCamera, ji: np.ndarray) -> list[dict[str, Any]]:
    """Return a list of GCPs generated with the given camera and ji pixel
    coordinates.
    """
    xyz = camera.pixel_to_world_z(ji, z=0)
    ids = string.ascii_uppercase[: ji.shape[1]]
    return [
        dict(id=gcp_id, ji=tuple(gcp_ji), xyz=tuple(gcp_xyz), info=None)
        for gcp_id, gcp_ji, gcp_xyz in zip(ids, ji.T.tolist(), xyz.T.tolist())
    ]


def read_rectification_ji(file: Path) -> np.ndarray:
    """Read a NumPy array of marker pixel coordinates from a NedCAD rectification data
    file.
    """
    with open(file) as f:
        lines = f.readlines()
    mkr_start = [len(line.split(',')) == 3 for line in lines].index(True)
    ji = [
        tuple(map(float, line.split(';')[-1].split(',')[:2]))
        for line in lines[mkr_start:]
    ]
    return np.array(ji).T


@pytest.fixture()
def runner():
    """Click runner for command line execution."""
    return CliRunner()


@pytest.fixture()
def gradient_array() -> np.ndarray:
    """An asymmetrical gradient array."""
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 100)
    xgrid, ygrid = np.meshgrid(x, y, indexing='xy')
    return (xgrid * ygrid * 250).astype('uint8')


@pytest.fixture()
def int_param(gradient_array: np.ndarray) -> dict[str, Any]:
    """Pinhole camera interior parameters."""
    im_size = gradient_array.shape[::-1]
    return dict(
        cam_type=CameraType.pinhole,
        im_size=im_size,
        focal_len=1.0,
        sensor_size=(1.0, im_size[1] / im_size[0]),
    )


@pytest.fixture()
def straight_camera(int_param: dict[str, Any]) -> FrameCamera:
    """A pinhole camera aligned with world coordinate axes and positioned above the
    world coordinate origin.
    """
    return create_camera(**int_param, opk=(0.0, 0.0, 0.0), xyz=(0.0, 0.0, 1.0))


@pytest.fixture()
def oblique_camera(int_param: dict[str, Any]) -> FrameCamera:
    """A pinhole camera with an oblique world view."""
    return create_camera(
        **int_param, opk=np.radians((15.0, -5.0, 10.0)).tolist(), xyz=(1.0, 2.0, 3.0)
    )


@pytest.fixture()
def gradient_image_file(gradient_array: np.ndarray, tmp_path: Path) -> Path:
    """A single band gradient image file."""
    filename = tmp_path.joinpath('src.png')
    profile = dict(
        driver='png',
        width=gradient_array.shape[1],
        height=gradient_array.shape[0],
        count=1,
        dtype=gradient_array.dtype,
    )
    with rio.open(filename, 'w', **profile) as im:
        im.write(gradient_array, indexes=1)
    return filename


@pytest.fixture()
def int_param_file(int_param: dict[str, Any], tmp_path: Path) -> Path:
    """An Orthority interior parameter file."""
    file = tmp_path.joinpath('int_param.yaml')
    int_param_dict = {'PR Camera': int_param}
    write_int_param(file, int_param_dict)
    return file


@pytest.fixture()
def gcp_ji(int_param: dict[str, Any]) -> np.ndarray:
    """A Numpy array of pixel coordinates for four GCPs."""
    buf = 10
    w, h = (int_param['im_size'][0] - 1, int_param['im_size'][1] - 1)
    ji = [[buf, h - buf], [w - buf, h - buf], [w - buf, buf], [buf, buf]]
    return np.array(ji).T


@pytest.fixture()
def marker_ji(int_param: dict[str, Any], gcp_ji: np.ndarray) -> np.ndarray:
    """A Numpy array of pixel coordinates for four 'markers' (coordinate origin in
    bottom left image corner).
    """
    return np.array([gcp_ji[0], int_param['im_size'][1] - 1 - gcp_ji[1]])


@pytest.fixture()
def straight_gcps(
    straight_camera: FrameCamera, gcp_ji: np.ndarray
) -> list[dict[str, Any]]:
    """A list of GCPs generated with 'straight_camera'."""
    return get_gcps(straight_camera, gcp_ji)


@pytest.fixture()
def straight_gcp_file(
    gradient_image_file: Path, straight_gcps: list[dict[str, Any]], tmp_path: Path
) -> Path:
    """An Orthority format GCP file containing GCPs generated with 'straight_camera'."""
    gcp_file = tmp_path.joinpath('gcps.geojson')
    gcp_dict = {gradient_image_file.name: straight_gcps}
    write_gcps(gcp_file, gcp_dict)
    return gcp_file


@pytest.fixture()
def cli_focal_len_sensor_size_marker_str(
    gradient_image_file: Path,
    int_param: dict[str, Any],
    straight_gcps: list[dict[str, Any]],
    marker_ji: np.ndarray,
    gcp_ji: np.ndarray,
) -> str:
    """A CLI string using the --focal-len, --sensor-size and --marker options."""
    # create marker option strings
    marker_strs = [
        f' -m {m_id} {gcp["xyz"][0]} {gcp["xyz"][1]} {m_ji[0]} {m_ji[1]} '
        for m_id, m_ji, gcp in zip('ABCD', marker_ji.T, straight_gcps)
    ]

    # create cli string
    cli_str = (
        f'-im {gradient_image_file} -fl {int_param["focal_len"]} -ss '
        f'{int_param["sensor_size"][0]} {int_param["sensor_size"][1]}'
    )
    cli_str += ''.join(marker_strs)
    return cli_str


@pytest.fixture()
def cli_int_param_gcp_str(
    gradient_image_file: Path, int_param_file: dict[str, Any], straight_gcp_file: Path
) -> str:
    """A CLI string using the --int-param and --gcp options."""
    cli_str = f'-im {gradient_image_file} -ip {int_param_file} -g {straight_gcp_file}'
    return cli_str


def test_rectify(
    straight_camera: FrameCamera, gradient_image_file: Path, gradient_array: np.ndarray
):
    """Test rectify.rectify() with auto resolution."""
    # the camera looks straight down on world coordinates so that the rectified image
    # should match the source image
    rect_array, transform = rectify(
        gradient_image_file, straight_camera, interp=Interp.nearest
    )

    assert (rect_array[0] == gradient_array).all()
    assert transform == (0.005, 0, -0.5, 0, -0.005, 0.25)


def test_rectify_resolution(straight_camera: FrameCamera, gradient_image_file: Path):
    """Test the rectify.rectify() resolution parameter."""
    res = (0.02, 0.01)
    rect_array, transform = rectify(
        gradient_image_file, straight_camera, resolution=res, interp='average'
    )
    assert rect_array.shape[1:] == (50, 50)
    assert (transform[0], abs(transform[4])) == res


def test_rectify_interp(straight_camera: FrameCamera, gradient_image_file: Path):
    """Test the rectify.rectify() interp parameter."""
    rect_arrays = []
    # use a resolution that gives non-integer remap maps to force interpolation
    res = (0.011, 0.011)
    for interp in [Interp.nearest, Interp.cubic]:
        rect_array, _ = rectify(
            gradient_image_file, straight_camera, interp=interp, resolution=res
        )
        rect_arrays.append(rect_array)

    # test images are similar but different
    assert rect_arrays[0] == pytest.approx(rect_arrays[1], abs=5)
    assert (rect_arrays[0] != rect_arrays[1]).any()


def test_rectify_nodata(oblique_camera: FrameCamera, gradient_image_file: Path):
    """Test the rectify.rectify() nodata parameter."""
    rect_masks = []
    for nodata in [254, 255]:
        # use the oblique camera so the rectified image contains invalid areas
        rect_array, _ = rectify(gradient_image_file, oblique_camera, nodata=nodata)
        rect_mask = rect_array == nodata
        assert rect_mask.sum() > 0
        rect_masks.append(rect_mask)

    assert (rect_masks[0] == rect_masks[1]).all()


def test_write_rectification_data(
    int_param: dict, marker_ji: np.ndarray, tmp_path: Path
):
    """Test param_io.write_rectification_data()."""
    src_name = 'source.jpg'
    ids = 'ABCD'
    markers = [dict(id=mkr_id, ji=mkr_ji) for mkr_id, mkr_ji in zip(ids, marker_ji.T)]
    out_file = tmp_path.joinpath('pixeldata.txt')
    write_rectification_data(out_file, src_name, int_param, markers)
    assert out_file.exists()

    # rough check of contents
    with open(out_file) as f:
        lines = f.readlines()
    prefixes = ['Photo', 'Size', 'Lens', 'Sensor', *ids]
    assert len(lines) == len(prefixes)
    assert all([line.startswith(start + ';') for start, line in zip(prefixes, lines)])
    for line, mkr_ji in zip(lines[-len(markers) :], marker_ji.T):
        assert f'{mkr_ji[0]:.4f},{mkr_ji[1]:.4f}' in line


def test_write_rectification_data_overwrite(int_param: dict, tmp_path: Path):
    """Test the param_io.write_rectification_data() overwrite parameter."""
    out_file = tmp_path.joinpath('pixeldata.txt')
    out_file.touch()
    with pytest.raises(FileExistsError):
        write_rectification_data(out_file, 'source.jpg', int_param, [])
    write_rectification_data(out_file, 'source.jpg', int_param, [], overwrite=True)
    assert out_file.exists()


@pytest.mark.parametrize(
    'cli_str', ['cli_focal_len_sensor_size_marker_str', 'cli_int_param_gcp_str']
)
def test_cli_outputs(
    cli_str: str,
    marker_ji: np.ndarray,
    gradient_array: np.ndarray,
    runner: CliRunner,
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    """Tes the accuracy of CLI output files with different interior parameter and
    marker location options.
    """
    cli_str: str = request.getfixturevalue(cli_str)
    # run the command
    cli_str += f' -od {tmp_path}'
    res = runner.invoke(cli, cli_str.split())
    assert res.exit_code == 0

    # test output files exist
    rect_image_file = tmp_path.joinpath('rect.png')
    rect_data_file = tmp_path.joinpath('pixeldata.txt')
    assert rect_image_file.exists()
    assert rect_data_file.exists()

    # test accuracy of output data
    with rio.open(rect_image_file, 'r') as rect_im:
        transform = rect_im.transform
        rect_array = rect_im.read()
    rect_ji = read_rectification_ji(rect_data_file)

    # the camera looks straight down on world coordinates so that the rectified image
    # should ~match the source image, and rectified marker pixel coordinates ~match
    # the input marker pixel coordinates
    assert transform[:6] == pytest.approx((0.005, 0, -0.5, 0, -0.005, 0.25), abs=1e-6)
    assert rect_array[0] == pytest.approx(gradient_array, abs=1)
    assert rect_ji == pytest.approx(marker_ji, abs=1)


def test_cli_resolution(cli_int_param_gcp_str: str, runner: CliRunner, tmp_path: Path):
    """Tes the CLI --res option."""
    res = (0.02, 0.01)
    cli_str = cli_int_param_gcp_str + f' -r {res[0]} -r {res[1]} -od {tmp_path}'
    res_ = runner.invoke(cli, cli_str.split())
    assert res_.exit_code == 0

    rect_image_file = tmp_path.joinpath('rect.png')
    assert rect_image_file.exists()
    with rio.open(rect_image_file, 'r') as rect_im:
        assert rect_im.res == res


def test_cli_interp(
    cli_int_param_gcp_str: str,
    runner: CliRunner,
    tmp_path: Path,
):
    """Tes the CLI --interp option."""
    # create rectified images with different interpolation types
    out_dirs = []
    for interp in ['nearest', 'cubic']:
        out_dir = tmp_path.joinpath(interp)
        out_dir.mkdir()
        out_dirs.append(out_dir)
        # use a resolution that gives non-integer remap maps to force interpolation
        cli_str = cli_int_param_gcp_str + f' -i {interp} -r 0.011 -od {out_dir}'
        res_ = runner.invoke(cli, cli_str.split())
        assert res_.exit_code == 0

    # test rectified images are similar but different
    rect_arrays = []
    for out_dir in out_dirs:
        rect_image_file = out_dir.joinpath('rect.png')
        assert rect_image_file.exists()
        with rio.open(rect_image_file, 'r') as rect_im:
            rect_arrays.append(rect_im.read())

    assert rect_arrays[0] == pytest.approx(rect_arrays[1], abs=5)
    assert (rect_arrays[0] != rect_arrays[1]).any()


def test_cli_nodata(
    oblique_camera: FrameCamera,
    gcp_ji: np.ndarray,
    gradient_image_file: Path,
    int_param_file: Path,
    runner: CliRunner,
    tmp_path: Path,
):
    """Test the CLI --nodata option."""
    # create a GCP file (using the oblique camera so the rectified image contains
    # invalid areas)
    gcps = get_gcps(oblique_camera, gcp_ji)
    gcp_file = tmp_path.joinpath('gcps.geojson')
    write_gcps(gcp_file, {gradient_image_file.name: gcps})

    # rectify with different --nodata vals
    out_dirs = []
    nodatas = [254, 255]  # image vals are <= 250
    cli_common_str = f'-im {gradient_image_file} -ip {int_param_file} -g {gcp_file}'
    for nodata in nodatas:
        out_dir = tmp_path.joinpath(str(nodata))
        out_dir.mkdir()
        out_dirs.append(out_dir)
        cli_str = cli_common_str + f' -n {nodata} -od {out_dir}'
        res = runner.invoke(cli, cli_str.split())
        assert res.exit_code == 0

    # test nodata masks
    rect_masks = []
    for nodata, out_dir in zip(nodatas, out_dirs):
        rect_image_file = out_dir.joinpath('rect.png')
        assert rect_image_file.exists()
        with rio.open(rect_image_file, 'r') as rect_im:
            rect_array = rect_im.read()
            rect_mask = rect_array == nodata
            assert rect_mask.sum() > 0
            rect_masks.append(rect_mask)

    assert (rect_masks[1] == rect_masks[0]).all()


def test_cli_overwrite(cli_int_param_gcp_str: str, runner: CliRunner, tmp_path: Path):
    """Tes the CLI --overwrite option."""
    rect_image_file = tmp_path.joinpath('rect.png')
    rect_image_file.touch()
    cli_str = cli_int_param_gcp_str + f' -od {tmp_path}'
    res = runner.invoke(cli, cli_str.split())
    assert res.exit_code != 0
    cli_str = cli_int_param_gcp_str + f' -o -od {tmp_path} '
    res = runner.invoke(cli, cli_str.split())
    assert res.exit_code == 0


def test_cli_export_params(
    cli_focal_len_sensor_size_marker_str: str,
    gradient_image_file: Path,
    int_param: dict[str, Any],
    straight_gcps: list[dict[str, Any]],
    runner: CliRunner,
    tmp_path: Path,
):
    """Tes the CLI --export-params option."""
    cli_str = cli_focal_len_sensor_size_marker_str + f' -ep -od {tmp_path}'
    res_ = runner.invoke(cli, cli_str.split())
    assert res_.exit_code == 0

    int_param_file = tmp_path.joinpath('int_param.yaml')
    gcp_file = tmp_path.joinpath('gcps.geojson')
    assert int_param_file.exists()
    assert gcp_file.exists()

    test_int_param_dict = read_oty_int_param(int_param_file)
    assert next(iter(test_int_param_dict.values())) == int_param

    ref_gcp_dict = {gradient_image_file.name: straight_gcps}
    test_gcp_dict = read_oty_gcps(gcp_file)
    assert test_gcp_dict == ref_gcp_dict
