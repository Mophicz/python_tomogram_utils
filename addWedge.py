import numpy as np
import mrcfile


def add_missing_wedge(filepath, missing_angle):
    """
    Add a missing wedge to a tomogram in the xz-plane of the Fourier space.

    Parameters:
    tomogram (numpy.ndarray): The input 3D tomogram.
    missing_angle (float): The missing angle in degrees.

    Returns:
    numpy.ndarray: The tomogram with the missing wedge applied.
    """
    # Open the MRC file
    with mrcfile.open(filepath, permissive=True) as mrc:
        tomo = mrc.data

    # Perform FFT on the tomogram to get the Fourier transform
    tomogram_fft = np.fft.fftn(tomo)

    # Calculate the missing wedge in radians
    missing_angle_rad = np.deg2rad(missing_angle)

    # Get the size of the tomogram
    x = tomo.shape[0]
    y = tomo.shape[1]
    z = tomo.shape[2]

    # Create 1D arrays for the kx, ky, and kz frequencies
    kx = np.fft.fftfreq(x) * x
    ky = np.fft.fftfreq(y) * y
    kz = np.fft.fftfreq(z) * z

    # Create a 3D grid of frequencies
    kX, kY, kZ = np.meshgrid(kx, ky, kz, indexing='ij')

    # Calculate angles in the xz-plane
    theta = np.arctan2(np.abs(kZ), np.abs(kX))

    # Create the missing wedge mask
    missing_wedge_mask = (theta < missing_angle_rad)

    # Apply the missing wedge mask to the Fourier transform
    tomogram_fft[missing_wedge_mask] = 0

    # Perform inverse FFT to get the tomogram with the missing wedge
    tomogram_with_missing_wedge = np.fft.ifftn(tomogram_fft).real

    split_filename = filepath.split('.')
    with mrcfile.new(f'{split_filename[0]}_mw_{missing_angle}.mrc', overwrite=True) as mrc:
        mrc.set_data(tomogram_with_missing_wedge)

    print(f"File '{split_filename[0]}_mw_{missing_angle}.mrc' successfully created.")


if __name__ == "__main__":
    add_missing_wedge('/Volumes/homes/frasunkiewicz/Projects/isonet/artificial_data_tests/repetitive_sphere/tomos/repetitive_sphere.mrc', 30)
