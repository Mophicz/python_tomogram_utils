import numpy as np
import mrcfile
import scipy
from numpy.ma.core import angle


def rotateTomogram(filename, angles=(-60, 60)):
    """
    Takes a tomogram and the (start, end) angles of the tilt series and rotates it to make the angles symmetrical to the XZ-plane

    Parameters:
    filename (string): tomogram .mrc filepath.
    angles (float, float): The start and end angle of the tilt series (the first should be negative).

    Returns:
    Nothing. Just saves a new .mrc at the same location as the input file, differentiated by a '_rotated' in the filename
    """
    wedge_angle_start = 90+angles[0]
    wedge_angle_end = 90-angles[1]
    wedge_angle_sum = wedge_angle_start + wedge_angle_end
    rotation_angle = wedge_angle_end - wedge_angle_sum/2

    print(f"rotation: {-rotation_angle}")

    with mrcfile.open(f'{filename}', permissive=True) as mrc:
        tomogram = mrc.data

    rotated_tomogram = scipy.ndimage.rotate(tomogram, -rotation_angle, axes=(0, 2), reshape=False)

    split_filename = filename.split('.')
    new_filename = split_filename[0] + '_rotated.mrc'
    with mrcfile.new(f'{new_filename}', overwrite=True) as mrc:
        mrc.set_data(rotated_tomogram)

    print(f"File successfully created.")


if __name__ == "__main__":
    rotateTomogram(filename='/Volumes/homes/frasunkiewicz/Documents/isonet/Nephrocytes_NP5_S2/tomo34_clicker_rec_ctf_binned.mrc', angles=(-59.99, 34.00))
