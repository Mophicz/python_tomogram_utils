import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, fftshift
import mrcfile


def plotTomo(filename):
    # Open the MRC file
    with mrcfile.open(f'{filename}.mrc', permissive=True) as mrc:
        tomo = mrc.data

    # Perform the Fourier transform
    fourier_transformed = fftn(tomo)

    # Shift the zero-frequency component to the center of the spectrum
    fourier_transformed_shifted = fftshift(fourier_transformed)

    # Compute the magnitude spectrum for visualization
    magnitude_spectrum = np.abs(fourier_transformed_shifted)

    # Log scale the magnitude spectrum for better visibility
    magnitude_spectrum_log = np.log(magnitude_spectrum + 1)

    # Extract central slices of the tomogram and its Fourier transform in XY, XZ, and YZ planes
    central_slice_xy = tomo[tomo.shape[0] // 2]
    fourier_slice_xy = magnitude_spectrum_log[magnitude_spectrum_log.shape[0] // 2]

    central_slice_xz = tomo[:, tomo.shape[1] // 2, :]
    fourier_slice_xz = magnitude_spectrum_log[:, magnitude_spectrum_log.shape[1] // 2, :]

    central_slice_yz = tomo[:, :, tomo.shape[2] // 2]
    fourier_slice_yz = magnitude_spectrum_log[:, :, magnitude_spectrum_log.shape[2] // 2]

    # Plot the original tomogram slices and their corresponding Fourier transform slices

    fig, axes = plt.subplots(3, 2, figsize=(16, 9))

    # XY plane
    axes[0, 0].imshow(central_slice_xy, cmap='gray')
    axes[0, 0].set_title('Tomogram Central Slice (XY Plane)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(fourier_slice_xy, cmap='gray')
    axes[0, 1].set_title('Fourier Transform Central Slice (XY Plane)')
    axes[0, 1].axis('off')

    # XZ plane
    axes[1, 0].imshow(central_slice_xz, cmap='gray')
    axes[1, 0].set_title('Tomogram Central Slice (XZ Plane)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(fourier_slice_xz, cmap='gray')
    axes[1, 1].set_title('Fourier Transform Central Slice (XZ Plane)')
    axes[1, 1].axis('off')

    # YZ plane
    axes[2, 0].imshow(central_slice_yz, cmap='gray')
    axes[2, 0].set_title('Tomogram Central Slice (YZ Plane)')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(fourier_slice_yz, cmap='gray')
    axes[2, 1].set_title('Fourier Transform Central Slice (YZ Plane)')
    axes[2, 1].axis('off')

    plt.savefig(f'{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Plot '{filename}.png' successfully created.")


if __name__ == "__main__":
    plotTomo('sphere')
    plotTomo('sphere_with_MW')
    plotTomo('randomSpheres')
    plotTomo('randomSpheres_with_MW')
