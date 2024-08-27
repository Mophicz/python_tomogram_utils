import numpy as np
import mrcfile


def add_missing_wedge(filename, missing_angle):
    """
    Add a missing wedge to a tomogram in the xz-plane of the Fourier space.

    Parameters:
    tomogram (numpy.ndarray): The input 3D tomogram.
    missing_angle (float): The missing angle in degrees.

    Returns:
    numpy.ndarray: The tomogram with the missing wedge applied.
    """
    # Open the MRC file
    with mrcfile.open(f'{filename}.mrc', permissive=True) as mrc:
        tomo = mrc.data

    # Perform FFT on the tomogram to get the Fourier transform
    tomogram_fft = np.fft.fftn(tomo)

    # Calculate the missing wedge in radians
    missing_angle_rad = np.deg2rad(missing_angle)

    # Get the size of the tomogram
    size = tomo.shape[0]

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

    with mrcfile.new(f'{filename}_with_MW.mrc', overwrite=True) as mrc:
        mrc.set_data(tomogram_with_missing_wedge)

    print(f"File '{filename}_with_MW.mrc' successfully created.")


if __name__ == "__main__":
    add_missing_wedge('sphere', 40)
    add_missing_wedge('randomSpheres', 40)
