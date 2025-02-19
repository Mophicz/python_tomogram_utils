"""
plot.py

Author: Michael Frasunkiewicz
Date: 29.01.2025

This script provides tools for generating and saving visualizations of tomograms. It allows the user to:

- Plot central slices of a tomogram alongside their corresponding Fourier transform slices.
- Generate and save image stacks of tomogram slices along specified planes (XY, XZ, YZ).
- Generate and save image stacks of the power spectrum of the tomogram along specified planes.

Modules:
    os, sys, argparse, numpy, matplotlib.pyplot, scipy.fft, mrcfile

Functions:
    plotCentralSlices(tomogram_path, save_dir=None):
        Generates a plot of the central slices of the specified tomogram along with its Fourier transform. The plot is saved as a PNG image.

    mkImageStack(tomogram_path, plane='XY', save_dir=None):
        Generates a stack of image files from the slices of the tomogram along the specified plane (XY, XZ, YZ). The images are saved in the specified directory.

    mkPowerSpectrum(tomogram_path, plane='XY', save_dir=None):
        Generates a stack of image files from the power spectrum of the tomogram along the specified plane (XY, XZ, YZ). The images are saved in the specified directory.

    main():
        Command-line tool for generating tomogram slice plots or image stacks. This function handles user input, validates parameters, and calls the appropriate function based on the mode selected.

Command-line Usage:
    python plot.py <tomogram_path> --mode <mode> [--save_dir <save_dir>] [--plane <plane>]

Command-line arguments:
    tomogram_path (str): Path to the tomogram file to process.

Options:
    --mode (str, required): Specifies the operation mode:
        - "csp": Generate central slice plots of the tomogram.
        - "stack": Generate a stack of images from the tomogram slices along a specified plane (XY, XZ, YZ).
        - "stack_ps": Generate a stack of images from the power spectrum of the tomogram along a specified plane.

    --save_dir (str, optional): Directory to save the output images. Defaults to the location of the tomogram file.

    --plane (str, optional): The plane to extract slices from for image stack generation. Options are:
        - "XY": Default plane.
        - "XZ"
        - "YZ"

Example Usage:
    1) Generate central slice plot:
        $ python plot.py path/to/tomogram --mode csp

    2) Generate image stack along the XZ plane:
        $ python plot.py path/to/tomogram --mode stack --plane XZ --save_dir path/to/output

    3) Generate power spectrum image stack in the XY plane:
        $ python plot.py path/to/tomogram --mode stack_ps --save_dir path/to/output
"""


import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, fftshift
import mrcfile


def plotCentralSlices(tomogram_path, save_dir=None):
    """
    Generates a plot of the specified tomograms central slices + power spectrum and saves it as a png file.

    Args:
        tomogram_path (str): Path leading to the tomogram to be plotted.
        save_dir (str): Path leading to the directory where the plot will be saved. Default is the location of the tomogram itself.

    Returns:
        None

    Raises:
        None

    Example:
        >>>plotCentralSlices(tomogram_path='path/to/tomogram')
    """
    print('\nloading tomogram...')

    # Open the MRC file
    with mrcfile.open(f'{tomogram_path}', permissive=True) as mrc:
        tomo = mrc.data

    print('\nfinished loading')

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
    fig, axes = plt.subplots(3, 2, figsize=(16, 9), layout="constrained")

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

    # extract tomograms directory path as default save_dir
    base_path = os.path.splitext(tomogram_path)[0]
    base_name = os.path.basename(base_path)

    # Save figure
    save_path = f'{save_dir}/{base_name}.png' if save_dir else f'{base_path}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nPlot '{base_name}.png' successfully created.\n")


def mkImageStack(tomogram_path, plane='XY', save_dir=None):
    """
    Generates a stack of image files containing the slices of a tomogram through the specified plane. These can subsequently be processed to a video using tools like QuickTime Player.

    Args:
        tomogram_path (str): Path leading to the tomogram to generate an image stack for.
        plane (str): Plane to extract the slices from. Options: XY, XZ or YZ. Default: XY.
        save_dir (str): Path leading to the directory where the image stack will be saved. Default is the location of the tomogram itself.

    Returns:
        None

    Raises:
        ValueError: If plane is not one of the three options: 'XY', 'XZ or 'YZ'.

    Example:
        mkImageStack(tomogram_path='path/to/tomogram')
    """
    print('\nloading tomogram..')

    # Open the MRC file
    with mrcfile.open(f'{tomogram_path}', permissive=True) as mrc:
        tomo = mrc.data

    print('\nfinished loading\n')

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

    base_path = os.path.splitext(tomogram_path)[0]
    base_name = os.path.basename(base_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.join(os.path.dirname(tomogram_path), base_name) + "_image_stack"
        os.makedirs(save_dir, exist_ok=True)

    # Save each slice as an image using matplotlib
    for i, slice_img in enumerate(normalized_slices):
        sys.stdout.write(f"\rSaving slice {i + 1}/{len(normalized_slices)}")
        sys.stdout.flush()
        plt.imshow(slice_img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        output_filename = os.path.join(save_dir, f'frame_{i:04d}.png')
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    print(f"\nExported {len(normalized_slices)} frames to '{save_dir}'.\n")


def mkPowerSpectrum(tomogram_path, plane='XY', save_dir=None):
    """
    Generates a stack of image files containing the slices of a tomogram`s power spectrum through the specified plane. These can subsequently be processed to a video using tools like QuickTime Player.

    Args:
        tomogram_path (str): Path leading to the tomogram to generate an image stack for.
        plane (str): Plane to extract the slices from. Options: XY, XZ or YZ. Default: XY.
        save_dir (str): Path leading to the directory where the image stack will be saved. Default is the location of the tomogram itself.

    Returns:
        None

    Raises:
        ValueError: If plane is not one of the three options: 'XY', 'XZ or 'YZ'.

    Example:
        mkPowerSpectrum(tomogram_path='path/to/tomogram')
        """
    print('\nloading tomogram..')

    # Open the MRC file
    with mrcfile.open(f'{tomogram_path}', permissive=True) as mrc:
        tomo = mrc.data

    print('\nfinished loading')

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

    base_path = os.path.splitext(tomogram_path)[0]
    base_name = os.path.basename(base_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.join(os.path.dirname(tomogram_path), base_name) + "_image_stack_ps"
        os.makedirs(save_dir, exist_ok=True)

    # Save each slice as an image using matplotlib
    for i, slice_img in enumerate(slices):
        sys.stdout.write(f"\rSaving slice {i + 1}/{len(slices)}")
        sys.stdout.flush()
        plt.imshow(slice_img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        output_filename = os.path.join(save_dir, f'frame_{i:04d}.png')
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    print(f"\nExported {len(slices)} frames to '{save_dir}'.\n")


def main():
    """
    Command-line tool for generating and saving tomogram slice plots or image stacks.

    This script allows users to process tomogram files and generate:
    - Central slice plots (CSP)
    - Image stacks along a specified plane
    - Power spectrum image stacks

    Usage:
        python plot.py <tomogram_path> --mode <mode> [--save_dir <save_dir>] [--plane <plane>]

    Command-Line arguments:
        tomogram_path (str):
            Path to the tomogram file.

    Options:
        --mode (str, required):
            Specifies the operation mode. Available options:
            - "csp"      : Generate and save central slice plots.
            - "stack"    : Generate an image stack from tomogram slices along a specified plane.
            - "stack_ps" : Generate a power spectrum image stack.

        --save_dir (str, optional):
            Directory where the output will be saved. Defaults to the directory of the input tomogram.

        --plane (str, optional):
            Plane for image stack generation (only used in "stack" and "stack_ps" modes). Options:
            - "XY" (default)
            - "XZ"
            - "YZ"

    Example Usage:
        1) Generate a central slice plot:
            $ python plot.py path/to/tomogram --mode csp

        2) Generate an image stack along the XZ plane:
            $ python plot.py path/to/tomogram --mode stack --plane XZ --save_dir path/to/output

        3) Generate a power spectrum image stack in the default XY plane:
            $ python plot.py path/to/tomogram --mode stack_ps --save_dir path/to/output
    """
    parser = argparse.ArgumentParser(description="Generate and save tomogram slice plots or image stacks.")
    parser.add_argument("tomogram_path", type=str, help="Path to the tomogram file.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the output. Defaults to the tomogram's location.")
    parser.add_argument("--mode", type=str, choices=["csp", "stack", "stack_ps"], required=True,
                        help="Operation mode: 'cs' for central slice plots, 'stack' for image stack, 'stack_ps' for power spectrum image stack.")
    parser.add_argument("--plane", type=str, choices=["XY", "XZ", "YZ"], default="XY",
                        help="Plane for image stack generation. Options: XY, XZ, YZ. Default: XY (only used in 'stack' mode).")

    args = parser.parse_args()

    if args.mode == "csp":
        plotCentralSlices(args.tomogram_path, args.save_dir)
    elif args.mode == "stack":
        mkImageStack(args.tomogram_path, args.plane, args.save_dir)
    elif args.mode == "stack_ps":
        mkPowerSpectrum(args.tomogram_path, args.plane, args.save_dir)


if __name__ == "__main__":
    main()
