import numpy as np


def add_missing_wedge(tomogram, missing_angle):
    """
    Add a missing wedge to a tomogram in the xz-plane of the Fourier space.

    Parameters:
    tomogram (numpy.ndarray): The input 3D tomogram.
    missing_angle (float): The missing angle in degrees.

    Returns:
    numpy.ndarray: The tomogram with the missing wedge applied.
    """
    # Perform FFT on the tomogram to get the Fourier transform
    tomogram_fft = np.fft.fftn(tomogram)

    # Calculate the missing wedge in radians
    missing_angle_rad = np.deg2rad(missing_angle)

    # Get the size of the tomogram
    size = tomogram.shape[0]

    # Create 1D arrays for the kx, ky, and kz frequencies
    kx = np.fft.fftfreq(size) * size
    ky = np.fft.fftfreq(size) * size
    kz = np.fft.fftfreq(size) * size

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

    return tomogram_with_missing_wedge
