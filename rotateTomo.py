import numpy as np
import mrcfile
import scipy

def rotateTomogram(filename, angle):
    with mrcfile.open(f'{filename}', permissive=True) as mrc:
        tomogram = mrc.data

    rotated_tomogram = scipy.ndimage.rotate(tomogram, angle, axes=(0, 2), reshape=False)

    split_filename = filename.split('.')
    new_filename = split_filename[0] + '_rotated.mrc'
    with mrcfile.new(f'{new_filename}', overwrite=True) as mrc:
        mrc.set_data(rotated_tomogram)

    print(f"File successfully created.")


if __name__ == "__main__":
    rotateTomogram(filename='/Volumes/homes/frasunkiewicz/Documents/isonet/tomo_28_binned/2xbinned_tomo_28_rec.mrc', angle=-33)
