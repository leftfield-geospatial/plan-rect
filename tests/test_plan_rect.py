from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio as rio
from orthority.camera import FrameCamera, create_camera
from orthority.enums import CameraType, Interp

from plan_rect.param_io import write_rectification_data
from plan_rect.rectify import rectify


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
def straight_camera(int_param: dict) -> FrameCamera:
    """A pinhole camera aligned with world coordinate axes and positioned above the
    world coordinate origin.
    """
    return create_camera(**int_param, opk=(0.0, 0.0, 0.0), xyz=(0.0, 0.0, 1.0))


@pytest.fixture()
def oblique_camera(int_param: dict) -> FrameCamera:
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
    # use a resolution that gives non-integer remap maps to force interpolation
    res = (0.011, 0.011)
    ref_array, transform = rectify(
        gradient_image_file, straight_camera, interp=Interp.nearest, resolution=res
    )
    test_array, transform = rectify(
        gradient_image_file, straight_camera, interp=Interp.cubic, resolution=res
    )

    # test images are similar but different
    assert test_array == pytest.approx(ref_array, abs=5)
    assert (test_array != ref_array).any()


def test_rectify_nodata(oblique_camera: FrameCamera, gradient_image_file: Path):
    """Test the rectify.rectify() nodata parameter."""
    ref_array, transform = rectify(
        gradient_image_file, oblique_camera, interp=Interp.nearest, nodata=255
    )
    test_array, transform = rectify(
        gradient_image_file, oblique_camera, interp=Interp.cubic, nodata=0
    )

    assert (test_array == 0).sum() > (ref_array == 0).sum()


def test_write_rectification_data(int_param: dict, tmp_path: Path):
    """Test param_io.write_rectification_data()."""
    src_name = 'source.jpg'
    mkr_ids = 'ABCD'
    ji = np.random.rand(4, 2) * 1000
    markers = [dict(id=mkr_id, ji=mkr_ji) for mkr_id, mkr_ji in zip(mkr_ids, ji)]
    out_file = tmp_path.joinpath('pixeldata.txt')

    write_rectification_data(out_file, src_name, int_param, markers)
    assert out_file.exists()

    # rough check of contents
    with open(out_file) as f:
        lines = f.readlines()
    prefixes = ['Photo', 'Size', 'Lens', 'Sensor', *mkr_ids]
    assert len(lines) == len(prefixes)
    assert all([line.startswith(start + ';') for start, line in zip(prefixes, lines)])
    for line, mkr_ji in zip(lines[-len(markers) :], ji):
        assert f'{mkr_ji[0]:.4f},{mkr_ji[1]:.4f}' in line


def test_write_rectification_data_overwrite(int_param: dict, tmp_path: Path):
    """Test the param_io.write_rectification_data() overwrite parameter."""
    out_file = tmp_path.joinpath('pixeldata.txt')
    out_file.touch()
    with pytest.raises(FileExistsError):
        write_rectification_data(out_file, 'source.jpg', int_param, [])
    write_rectification_data(out_file, 'source.jpg', int_param, [], overwrite=True)
    assert out_file.exists()
