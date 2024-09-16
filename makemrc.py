import numpy as np
import mrcfile


def mkSphere(filename):
    # Define the size of the volume
    size = 512
    radius = 150

    # Create a 3D grid of coordinates
    x, y, z = np.indices((size, size, size)) - size//2

    # Create a spherical mask
    sphere = (x**2 + y**2 + z**2) <= radius**2

    # Convert the mask to float32 for MRC file compatibility
    sphere = sphere.astype(np.float32)

    # Create and save the MRC file
    with mrcfile.new(f'{filename}.mrc', overwrite=True) as mrc:
        mrc.set_data(sphere)

    print(f"File '{filename}.mrc' successfully created.")


def mkRandomSpheres(filename, num_spheres, sphere_radius):
    # Define the size of the volume
    size = 100

    # Create an empty volume
    volume = np.zeros((size, size, size), dtype=np.float32)

    # Create a small sphere mask
    x, y, z = np.indices((2 * sphere_radius, 2 * sphere_radius, 2 * sphere_radius)) - sphere_radius
    sphere = (x ** 2 + y ** 2 + z ** 2) <= sphere_radius ** 2

    max_attempts = 1000  # Max attempts to place each sphere

    # Attempt to place spheres randomly without overlap
    for _ in range(num_spheres):
        placed = False
        attempts = 0
        while not placed and attempts < max_attempts:
            # Randomly choose the center of the new sphere
            cx, cy, cz = np.random.randint(sphere_radius, size - sphere_radius, size=3)

            # Extract the subvolume where the new sphere will be placed
            subvolume = volume[cx - sphere_radius:cx + sphere_radius,
                        cy - sphere_radius:cy + sphere_radius,
                        cz - sphere_radius:cz + sphere_radius]

            # Check if there's any overlap
            if np.all(subvolume + sphere <= 1.0):  # Ensure no overlap
                # Place the sphere by adding the mask
                volume[cx - sphere_radius:cx + sphere_radius,
                cy - sphere_radius:cy + sphere_radius,
                cz - sphere_radius:cz + sphere_radius] += sphere
                placed = True
            attempts += 1

        if not placed:
            print(f"Warning: Could not place sphere {_ + 1} after {max_attempts} attempts.")

    # Create and save the MRC file
    with mrcfile.new(f'{filename}.mrc', overwrite=True) as mrc:
        mrc.set_data(volume)

    print(f"File '{filename}.mrc' successfully created.")


if __name__ == "__main__":
    mkSphere('sphere')
    #mkRandomSpheres('randomSpheres', 50, 5)
