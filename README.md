# Plan-Rect

Plan-Rect is command line tool for rectifying oblique images to a plane.

## Installation

Plan-Rect is a python 3 package that depends on [Orthority](https://github.com/leftfield-geospatial/orthority).  It has not yet been published to PyPI or conda-forge.  To install the package, first clone the repository: 

```commandline
git clone https://github.com/leftfield-geospatial/plan-rect.git
cd plan-rect
```

Then use [``pip``](https://pip.pypa.io/) to install in editable mode:

```commandline
pip install -e .
```

If installing into a [``conda``](https://docs.anaconda.com/free/miniconda) environment, it is best to install Orthority with ``conda`` first, before running the command above:

```commandline
conda install -c conda-forge "orthority>=0.6.0"
```

## Usage

Rectification is performed with the ``plan-rect`` command.  It requires an image, camera interior parameters and marker locations as inputs, and creates a rectified image and rectification data file as outputs.  Its options are described below:

| Option                        | Value                                                          | Description                                                                                                                                        |
|-------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| ``-im`` / ``--image``         | FILE                                                           | Path / URI of the source image (required).                                                                                                         |
| ``-fl`` / ``--focal-len``     | FLOAT                                                          | Camera focal length (any units).                                                                                                                   |
| ``-ss`` / ``--sensor-size``   | WIDTH HEIGHT                                                   | Camera sensor size in the same units as ``-fl`` / ``--focal-len``.                                                                                 |
| ``-ip`` / ``--int-param``     | FILE                                                           | Path / URI of an Orthority format interior parameter file.                                                                                         |
| ``-m`` / ``--marker``         | ID X Y COL ROW                                                 | Marker ID and location in world and pixel coordinates, with pixel coordinate origin at the bottom left image corner.                               | 
| ``-g`` / ``--gcp``            | FILE                                                           | Path / URI of an Orthority GCP file defining marker locations.                                                                                     |
| ``-r`` / ``--res``            | FLOAT                                                          | Rectified pixel size in meters.  Can be used twice for non-square pixels: ``--res WIDTH --res HEIGHT``.  Defaults to the ground sampling distance. |
| ``-i`` / ``--interp``         | ``nearest``, ``average``, ``bilinear``, ``cubic``, ``lanczos`` | Interpolation method for remapping source to rectified image.  Defaults to ``cubic``.                                                              |
| ``-n`` /  ``--nodata``        | FLOAT                                                          | Nodata value for the rectified image.  Defaults to the maximum value the image data type if it is integer, and ``nan`` if it is floating point.    |
| ``-ep`` / ``--export-params`` |                                                                | Export interior parameters and markers to Orthority format files and exit.                                                                         |
| ``-od`` / ``--out-dir``       | DIRECTORY                                                      | Path / URI of the output file directory.  Defaults to the current working directory.                                                               | 
| ``-o`` / ``--overwrite``      |                                                                | Overwrite existing output(s).                                                                                                                      |
| ``--version``                 |                                                                | Show the version and exit.                                                                                                                         |
| ``--help``                    |                                                                | Show the help and exit.                                                                                                                            |

Camera interior parameters are required with either ``-fl`` / ``--focal-len`` and ``-ss`` / ``--sensor-size``, or ``-ip`` / ``--int-param``.  

Marker locations are required with either ``-m`` / ``--marker`` or ``-g`` / ``--gcp``.  The ``-m`` / ``--marker`` option can be provided multiple times. At least three markers are required. 


### Examples

Supply interior parameters with ``-fl`` / ``--focal-len`` and ``-ss`` / ``--sensor-size``, and marker locations with ``-m`` / ``--marker``:

```commandline
plan-rect --image source.jpg --focal-len 50 --sensor-size 31.290 23.491 --marker A 0 0 1002 1221 --marker B 2.659 0 4261 1067 --marker C 2.321 5.198 3440 3706 --marker D -0.313 4.729 1410 3663
```

Supply interior parameters with ``-ip`` / ``--int-param``  and marker locations with ``-g`` / ``--gcp``:

```commandline
plan-rect --image source.jpg --int-param int_param.yaml --gcp gcps.geojson
```

Set the rectified image pixel size with ``-r`` / ``--res``:

```commandline
plan-rect --image source.jpg --res 0.01 --int-param int_param.yaml --gcp gcps.geojson
```

Export interior parameters and marker locations to Orthority format files in the ``data`` directory, overwriting existing outputs:

```commandline
plan-rect --image source.jpg --export-params --out-dir data --overwrite --focal-len 50 --sensor-size 31.290 23.491 --marker A 0 0 1002 1221 --marker B 2.659 0 4261 1067 --marker C 2.321 5.198 3440 3706 --marker D -0.313 4.729 1410 3663
```

## Licence

Plan-Rect is licenced under the [GNU Affero General Public License v3.0 (AGPLv3)](LICENSE).

## Acknowledgments

This project was funded by [NedCAD](https://nedcad.nl/).