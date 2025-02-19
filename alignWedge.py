"""
alignWedge.py

Author: Michael Frasunkiewicz
Date: 19.02.2025

This script rotates a tomogram to symmetrize the missing wedge in the Fourier space. The missing wedge typically arises in cryo-ET due to the limited tilt-range during acquisition. By rotating the tomogram in the XZ-plane, the missing wedge is aligned symmetrically to the Z-axis.

Modules:
    os, argparse, mrcfile, scipy

Functions:
    rotateTomogram(tomogram_path, angles, save_dir=None):
        Rotates a tomogram in the XZ-plane based on the given tilt angles to symmetrize the missing wedge.

    main():
        Command-line interface for processing tomograms.

Command-line Usage:
    $ python alignWedge.py <tomogram_path> <negative_angle> <positive_angle> [--save_dir <save_dir>]

Command-line arguments:
    tomogram_path (str): Path to the input tomogram (MRC file).
    angles (tuple of float): The start and end angles (in degrees) of the missing wedge, measured from the Z-axis.
                             The first value should be negative and the second positive.
    --save_dir (str, optional): Directory to save the rotated tomogram. Defaults to the input tomogram's directory.

Example Usage:
    $ python alignWedge.py path/to/tomogram.mrc -60 30 --save_dir path/to/output
"""



import os
import argparse
import mrcfile
import scipy


def rotateTomogram(tomogram_path, angles, save_dir=None):
    """
    Rotates a tomogram with non-symmetrical missing wedge in the XZ-plane to make the wedge symmetrical to the Z-axis.

    Args::
        tomogram_path (str): Path leading to the tomogram to be rotated.
        angles (tuple of float): The start and end angle of the tilt series acquisition.

    Example:
        rotateTomogram(tomogram_path='path/to/tomogram', angles=(-60, 30))
    """
    # calculate rotation-angle from tilt series acquisition angles
    wedge_angle_start, wedge_angle_end = angles
    wedge_angle_sum = abs(wedge_angle_start) + abs(wedge_angle_end)
    rotation_angle = (abs(wedge_angle_start) - abs(wedge_angle_end)) / 2

    print(f"\nwedge-sum: {wedge_angle_sum} deg")
    print(f"rotation: {-rotation_angle} deg")

    print('\nloading tomogram...')

    # Open the MRC file
    with mrcfile.open(f'{tomogram_path}', permissive=True) as mrc:
        tomo = mrc.data

    print('\nfinished loading')

    rotated_tomogram = scipy.ndimage.rotate(tomo, -rotation_angle, axes=(0, 2), reshape=False, order=3, mode='constant', cval=0)

    # extract tomograms directory path as default save_dir
    base_path = os.path.splitext(tomogram_path)[0]
    base_name = os.path.basename(base_path)

    if save_dir:
        save_path = os.path.join(save_dir, base_name) + '_rotated.mrc'
    else:
        save_path = f'{base_path}_rotated.mrc'

    with mrcfile.new(f'{save_path}', overwrite=True) as mrc:
        mrc.set_data(rotated_tomogram)

    print(f"\nTomogram successfully rotated.\n")


def main():
    """
    Command-line tool for rotating tomograms for missing wedge symmetry.

    Usage:
        python script.py <tomogram_path> --mode <mode> [--save_dir <save_dir>] [--plane <plane>]

    Command-line arguments:
        tomogram_path (str):
            Path to the tomogram file.

    Options:
        --angle (tuple of float, required):
            Specifies the start and end angles of the tilt series (start, end). start: beginning of tilt series acquisition (usually negative) end: end of tilt series acquisition (usually positive).

        --save_dir (str, optional):
            Directory where the output will be saved. Defaults to the directory of the input tomogram.

    Example Usage:
        python alignWedge.py path/to/tomogram --angles (-60, 60) --save_dir path/to/output
    """
    parser = argparse.ArgumentParser(description="Rotate tomograms for missing wedge symmetry.")
    parser.add_argument("tomogram_path", type=str, help="Path to the tomogram file.")
    parser.add_argument("angles", type=float, nargs=2, help="Start and end angles of the missing wedge (degrees).")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the output. Defaults to the tomogram's location.")

    args = parser.parse_args()
    rotateTomogram(args.tomogram_path, args.angles, args.save_dir)


if __name__ == "__main__":
    main()
