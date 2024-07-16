import numpy as np
import mrcfile

# Define the size of the volume
size = 100
radius = size // 2

# Create a 3D grid of coordinates
x, y, z = np.indices((size, size, size)) - radius

# Create a spherical mask
sphere = (x**2 + y**2 + z**2) <= radius**2

# Convert the mask to float32 for MRC file compatibility
sphere = sphere.astype(np.float32)

# Create and save the MRC file
with mrcfile.new('sphere.mrc', overwrite=True) as mrc:
    mrc.set_data(sphere)

print("MRC file 'sphere.mrc' created successfully.")
