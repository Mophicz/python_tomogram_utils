

import os
import argparse
import mrcfile
import scipy
from sympy.codegen.ast import float32


def rotateTomogram(tomogram_path, angles, save_dir=None):
    """
    Rotates a tomogram with asymmetrical missing wedge to make the wedge symmetrical to the XZ-plane.

    Args::
        tomogram_path (str): Path leading to the tomogram to be rotated.
        angles (tuple of float): The start and end angle of the tilt series (the first should be negative).

    Returns:
        None

    Raises:
        None

    Example:
        rotateTomogram(tomogram_path='path/to/tomogram', angles=(-60, 60))
    """
    # calculate rotation-angle from tilt series acquisition angles
    wedge_angle_start = 90 + angles[0]
    wedge_angle_end = 90 - angles[1]
    wedge_angle_sum = wedge_angle_start + wedge_angle_end
    rotation_angle = wedge_angle_end - wedge_angle_sum / 2

    print(f"\nwedge-sum: {wedge_angle_sum} deg")
    print(f"rotation: {-rotation_angle} deg")

    print('\nloading tomogram..')

    # Open the MRC file
    with mrcfile.open(f'{tomogram_path}', permissive=True) as mrc:
        tomo = mrc.data

    print('\nfinished loading')

    rotated_tomogram = scipy.ndimage.rotate(tomo, -rotation_angle, axes=(0, 2), reshape=False)

    # extract tomograms directory path as default save_dir
    base_path = os.path.splitext(tomogram_path)[0]
    base_name = os.path.basename(base_path)

    if save_dir:
        save_path = os.path.join(save_dir, base_name) + '_rotated.mrc'
    else:
        save_path = f'{base_path}_rotated.mrc'

    with mrcfile.new(f'{save_path}', overwrite=True) as mrc:
        mrc.set_data(rotated_tomogram)

    print(f"\nTomogram successfully rotated.")


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
        python rotate.py path/to/tomogram --angles (-60, 60) --save_dir path/to/output
    """
    parser = argparse.ArgumentParser(description="Rotate tomograms for missing wedge symmetry.")
    parser.add_argument("tomogram_path", type=str, help="Path to the tomogram file.")
    parser.add_argument("angles", type=float, nargs=2, help="Start and end angles of the missing wedge (degrees).")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the output. Defaults to the tomogram's location.")

    args = parser.parse_args()
    rotateTomogram(args.tomogram_path, args.angle, args.save_dir)


if __name__ == "__main__":
    main()
