[![Tests](https://github.com/leftfield-geospatial/plan-rect/actions/workflows/run-unit-tests.yml/badge.svg)](https://github.com/leftfield-geospatial/plan-rect/actions/workflows/run-unit-tests.yml)
![PyPI - Version](https://img.shields.io/pypi/v/plan-rect?color=blue)


# Plan-Rect

Plan-Rect is command line tool for rectifying oblique images to a plane.

## Installation

Plan-Rect is a python 3 package that can be installed with [pip](<https://pip.pypa.io/>): 

```commandline
pip install plan-rect
```

## Usage

Rectification is performed with the ``plan-rect`` command.  It requires an image and marker locations as inputs, and creates a rectified image and rectification data file as outputs.  Its options are described below:

| Option                        | Value                                                          | Description                                                                                                                                        |
|-------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| ``-im`` / ``--image``         | FILE                                                           | Path / URI of the source image (required).                                                                                                         |
| ``-m`` / ``--marker``         | ID X Y COL ROW                                                 | Marker ID and location in world and pixel coordinates, with pixel coordinate origin at the bottom left image corner.                               | 
| ``-g`` / ``--gcp``            | FILE                                                           | Path / URI of an Orthority GCP file defining marker locations.                                                                                     |
| ``-r`` / ``--res``            | FLOAT                                                          | Rectified pixel size in meters.  Can be used twice for non-square pixels: ``--res WIDTH --res HEIGHT``.  Defaults to the ground sampling distance. |
| ``-i`` / ``--interp``         | ``nearest``, ``average``, ``bilinear``, ``cubic``, ``lanczos`` | Interpolation method for remapping source to rectified image.  Defaults to ``cubic``.                                                              |
| ``-n`` /  ``--nodata``        | FLOAT                                                          | Nodata value for the rectified image.  Defaults to the maximum value of the image data type if it is integer, and ``nan`` if it is floating point. |
| ``-ep`` / ``--export-params`` |                                                                | Export interior parameters and markers to Orthority format files and exit.                                                                         |
| ``-od`` / ``--out-dir``       | DIRECTORY                                                      | Path / URI of the output file directory.  Defaults to the current working directory.                                                               | 
| ``-o`` / ``--overwrite``      |                                                                | Overwrite existing output(s).                                                                                                                      |
| ``--version``                 |                                                                | Show the version and exit.                                                                                                                         |
| ``--help``                    |                                                                | Show the help and exit.                                                                                                                            |

Marker locations are required with either ``-m`` / ``--marker`` or ``-g`` / ``--gcp``.  The ``-m`` / ``--marker`` option can be provided multiple times. At least four markers are required. 


### Examples

Supply marker locations with ``-m`` / ``--marker``:

```commandline
plan-rect --image source.jpg --marker A 0 0 1002 1221 --marker B 2.659 0 4261 1067 --marker C 2.321 5.198 3440 3706 --marker D -0.313 4.729 1410 3663
```

Supply marker locations with ``-g`` / ``--gcp``:

```commandline
plan-rect --image source.jpg --gcp gcps.geojson
```

Set the rectified image pixel size with ``-r`` / ``--res``:

```commandline
plan-rect --image source.jpg --res 0.01 --gcp gcps.geojson
```

Export marker locations to an Orthority GCP file in the ``data`` directory, overwriting any existing file:

```commandline
plan-rect --image source.jpg --export-params --out-dir data --overwrite --marker A 0 0 1002 1221 --marker B 2.659 0 4261 1067 --marker C 2.321 5.198 3440 3706 --marker D -0.313 4.729 1410 3663
```

## Licence

Plan-Rect is licenced under the [GNU Affero General Public License v3.0 (AGPLv3)](LICENSE).

## Acknowledgments

This project was funded by [NedCAD](https://nedcad.nl/).