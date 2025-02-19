"""
mkTomogram.py

Author: Michael Frasunkiewicz
Date: 19.02.2025

This script has two examples of generating simple synthetic tomograms using spheres. The functions can be called by running the script after entering the function with a valid filepath in the if-statement at the bottom of the script.

Modules:
    numpy, mrcfile

Functions:
    getSphere(x, y, z, center, radius, thickness):
        Generate a sphere at given coordinates with given radius and thickness in a 3D volume.

    mkSphere(filepath):
        Save a tomogram (with .mrc file extension) of a volume with defined dimensions and a single sphere of specified dimensions in the center.

    mkRandomSpheres(filepath):
        Save a tomogram (with .mrc file extension) of a volume with defined dimensions and a defined number of randomly positioned spheres with specified dimensions.

Example Usage:
    if __name__ == "__main__":
        mkSphere("/path/to/tomogram.mrc")
"""


import numpy as np
import mrcfile


def getSphere(x, y, z, center, radius, thickness):
    """
    Generate a sphere at given coordinates with given radius and thickness in a 3D volume.
    """
    center_distance = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2

    # Compute inner and outer radius
    outer_radius_squared = radius ** 2
    inner_radius_squared = (radius - thickness) ** 2

    sphere = (center_distance <= outer_radius_squared) & (center_distance >= inner_radius_squared)
    return sphere


def mkSphere(filepath):
    """
    Save a tomogram (with .mrc file extension) of a volume with defined dimensions and a single sphere of specified dimensions in the center.
    """
    dim = (512, 720, 650)
    inner_radius = 150
    outer_radius = 170
    thickness = 5

    # Create a 3D grid of coordinates
    x, y, z = np.indices(dim)

    # Define center
    center = (dim[0] // 2, dim[1] // 2, dim[2] // 2)

    # Create a spherical mask
    inner_sphere = getSphere(x,y,z,center, inner_radius, thickness)
    outer_sphere = getSphere(x, y, z, center, outer_radius, thickness)

    sphere = inner_sphere | outer_sphere

    # Convert the mask to float32 for MRC file compatibility
    sphere = sphere.astype(np.float32)

    # Create and save the MRC file
    with mrcfile.new(filepath, overwrite=True) as mrc:
        mrc.set_data(sphere)

    print(f"File {filepath} successfully created.")


def mkRandomSpheres(filepath):
    """
    Save a tomogram (with .mrc file extension) of a volume with defined dimensions and a defined number of randomly positioned spheres with specified dimensions.
    """
    # Volume dimensions (same as in mkSphere)
    dim = (512, 720, 650)
    num_spheres = 50
    inner_radius = 30
    outer_radius = 50
    thickness = 5

    # Initialize the 3D volume
    volume = np.zeros(dim, dtype=np.float32)

    # Grid size for one sphere bounding box
    grid_size = 2 * outer_radius + 1
    x, y, z = np.indices((grid_size, grid_size, grid_size))

    # Generate masks for the inner and outer shells
    inner_sphere_mask = getSphere(x, y, z, center=(outer_radius, outer_radius, outer_radius),
                                  radius=inner_radius, thickness=thickness)
    outer_sphere_mask = getSphere(x, y, z, center=(outer_radius, outer_radius, outer_radius),
                                  radius=outer_radius, thickness=thickness)

    # Combined double-shell mask
    sphere_mask = inner_sphere_mask | outer_sphere_mask

    max_attempts = 1000  # Max attempts to place each sphere

    for i in range(num_spheres):
        placed = False
        attempts = 0

        while not placed and attempts < max_attempts:
            # Randomly choose the center of the sphere
            cx = np.random.randint(outer_radius, dim[0] - outer_radius)
            cy = np.random.randint(outer_radius, dim[1] - outer_radius)
            cz = np.random.randint(outer_radius, dim[2] - outer_radius)

            # Define the bounding box in the volume
            x_start, x_end = cx - outer_radius, cx + outer_radius + 1
            y_start, y_end = cy - outer_radius, cy + outer_radius + 1
            z_start, z_end = cz - outer_radius, cz + outer_radius + 1

            # Extract the subvolume
            subvolume = volume[x_start:x_end, y_start:y_end, z_start:z_end]

            # Check for overlap
            if np.all(subvolume + sphere_mask <= 1.0):  # Ensure no overlap
                # Place the sphere in the volume
                volume[x_start:x_end, y_start:y_end, z_start:z_end] += sphere_mask
                placed = True
            attempts += 1

        if not placed:
            print(f"Warning: Could not place sphere {i + 1} after {max_attempts} attempts.")

    # Save the volume as an MRC file
    with mrcfile.new(filepath, overwrite=True) as mrc:
        mrc.set_data(volume)

    print(f"File '{filepath}' successfully created.")


if __name__ == "__main__":
    mkSphere('')
