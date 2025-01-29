import os
import argparse
import numpy as np
import mrcfile


def add_missing_wedge(tomogram_path, angle):
    """
    Add a missing wedge to a tomogram in the xz-plane of the Fourier space.

    Args:
        tomogram_path (str): Path leading to the tomogram.
        angle (float): Angle of the missing wedge in degrees.

    Returns:
        None

    Example:
        add_missing_wedge(tomogram_path='path/to/tomogram', angles=60)
    """
    print('\nLoading tomogram...')

    # Open the MRC file
    with mrcfile.open(tomogram_path, permissive=True) as mrc:
        tomo = mrc.data

    print('\nFinished loading')

    # Perform FFT on the tomogram to get the Fourier transform
    tomogram_fft = np.fft.fftn(tomo)

    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Get the size of the tomogram
    x, y, z = tomo.shape

    # Create 1D arrays for the kx, ky, and kz frequencies
    kx = np.fft.fftfreq(x) * x
    ky = np.fft.fftfreq(y) * y
    kz = np.fft.fftfreq(z) * z

    # Create a 3D grid of frequencies
    kX, kY, kZ = np.meshgrid(kx, ky, kz, indexing='ij')

    # Calculate angles in the xy-plane
    theta = np.arctan2(np.abs(kZ), np.abs(kX))

    # Create the missing wedge mask
    missing_wedge_mask = (theta < angle_rad)

    # Apply the missing wedge mask to the Fourier transform
    tomogram_fft[missing_wedge_mask] = 0

    # Perform inverse FFT to get the tomogram with the missing wedge
    tomogram_with_missing_wedge = np.fft.ifftn(tomogram_fft).real

    base_path = os.path.splitext(tomogram_path)[0]
    base_name = os.path.basename(base_path)
    new_path = f'{base_path}_mw_{angle}.mrc'

    with mrcfile.new(new_path, overwrite=True) as mrc:
        mrc.set_data(tomogram_with_missing_wedge.astype(np.float32))

    print(f"\nFile '{base_name}_mw_{angle}.mrc' successfully created.")


def main():
    """
    Parse command-line arguments and apply a missing wedge to the given tomogram.

    This function allows the script to be executed from the command line, where it:
    - Reads a tomogram from an MRC file.
    - Applies a missing wedge in the xz-plane of its Fourier space based on user-provided angles.
    - Saves the modified tomogram as a new MRC file.

    Command-line arguments:
        tomogram_path (str): Path to the input tomogram (MRC file).
        angles (float, float): The start and end angles (in degrees) defining the missing wedge.

    Example usage:
        $ python addWedge.py path/to/tomogram.mrc -60 30
    """
    parser = argparse.ArgumentParser(description="Add a missing wedge to a tomogram in Fourier space.")
    parser.add_argument("tomogram_path", type=str, help="Path to the input tomogram.")
    parser.add_argument("angles", type=float, nargs=1, help="Start and end angles of the missing wedge (degrees).")

    args = parser.parse_args()
    add_missing_wedge(args.tomogram_path, args.angles)


if __name__ == "__main__":
    main()
