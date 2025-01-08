import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, fftshift
import mrcfile
import sys


def plotCentralSlices(filename):
    # Open the MRC file
    with mrcfile.open(f'{filename}', permissive=True) as mrc:
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

    png_filename = os.path.splitext(filename)[0]

    plt.savefig(f'{png_filename}.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Plot '{png_filename}.png' successfully created.")


def plotTomogram(filename, plane='XY', output_dir='output_frames'):
    # Open the MRC file
    with mrcfile.open(f'{filename}', permissive=True) as mrc:
        tomo = mrc.data

    # Determine the 1st and 99th percentiles for normalization
    p1, p99 = np.percentile(tomo, (1, 99))

    # Select slices based on the plane parameter
    if plane == 'XY':
        slices = tomo
    elif plane == 'XZ':
        slices = np.transpose(tomo, (1, 0, 2))
    elif plane == 'YZ':
        slices = np.transpose(tomo, (2, 0, 1))
    else:
        raise ValueError("Invalid plane. Choose from 'XY', 'XZ', or 'YZ'.")

    # Normalize the slices based on the percentiles
    normalized_slices = np.clip((slices - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each slice as an image using matplotlib
    for i, slice_img in enumerate(normalized_slices):
        sys.stdout.write(f"\rSaving slice {i + 1}/{len(normalized_slices)}")
        sys.stdout.flush()
        plt.imshow(slice_img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        output_filename = os.path.join(output_dir, f'frame_{i:04d}.png')
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    print()
    print(f"Exported {len(normalized_slices)} frames to '{output_dir}'.")


def plotPowerSpectrum(filename, plane='XY', output_dir='output_frames'):
    # Open the MRC file
    with mrcfile.open(f'{filename}', permissive=True) as mrc:
        tomo = mrc.data

    # Perform the Fourier transform
    fourier_transformed = fftn(tomo)

    # Shift the zero-frequency component to the center of the spectrum
    fourier_transformed_shifted = fftshift(fourier_transformed)

    # Compute the magnitude spectrum for visualization
    magnitude_spectrum = np.abs(fourier_transformed_shifted)

    # Log scale the magnitude spectrum for better visibility
    magnitude_spectrum_log = np.log(magnitude_spectrum + 1)

    # Determine the 1st and 99th percentiles for normalization
    p1, p99 = np.percentile(magnitude_spectrum_log, (1, 99))

    # Normalize the entire log-transformed data to the range [0, 255]
    normalized_spectrum = np.clip((magnitude_spectrum_log - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)

    # Select slices based on the plane parameter
    if plane == 'XY':
        slices = normalized_spectrum  # Already in the form of XY slices
    elif plane == 'XZ':
        slices = np.transpose(normalized_spectrum, (1, 0, 2))  # Transpose to XZ slices
    elif plane == 'YZ':
        slices = np.transpose(normalized_spectrum, (2, 0, 1))  # Transpose to YZ slices
    else:
        raise ValueError("Invalid plane. Choose from 'XY', 'XZ', or 'YZ'.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each slice as an image using matplotlib
    for i, slice_img in enumerate(slices):
        sys.stdout.write(f"\rSaving slice {i + 1}/{len(slices)}")
        sys.stdout.flush()
        plt.imshow(slice_img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        output_filename = os.path.join(output_dir, f'frame_{i:04d}.png')
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    print()
    print(f"Exported {len(slices)} frames to '{output_dir}'.")


def deleteAllFrames(output_dir='output_frames'):
    # Get a list of all files in the directory
    file_pattern = os.path.join(output_dir, '*.png')
    files = glob.glob(file_pattern)

    # Delete each file
    for file in files:
        os.remove(file)


if __name__ == "__main__":

    #deleteAllFrames()

    #plotPowerSpectrum(filename='/Volumes/homes/frasunkiewicz/Projects/deepdewedge/Drosophila/rec_tomo_5_half1_binned_rotated.mrc', plane='XZ')

    #plotTomogram(filename='/Volumes/homes/frasunkiewicz/Documents/isonet/tomo_28_binned/corrected_tomos/2xbinned_tomo_28_rec_rotated_corrected.mrc', plane='XY')

    #plotTomogram(filename='sphere_2x_binned.mrc', plane='XY')

    plotCentralSlices(filename='/Volumes/homes/frasunkiewicz/Projects/isonet/artificial_data_tests/repetitive_sphere/tomos/repetitive_sphere_mw_30.mrc')