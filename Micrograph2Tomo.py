import os
import mrcfile
import numpy as np

# Source and destination directories
source_dir = '/Volumes/homes/microscope/Pelin/Krios/241113_EKO/View_images/'
dest_dir = '/Volumes/homes/zengin/Projects/Internship_Pos_lab/BW25II3_images_raw/'

def mrc2raw(source_dir, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate through all files in the source directory
    for file_name in os.listdir(source_dir):
        if file_name.endswith('.mrc'):  # Process only .mrc files
            # Full path to the current file
            source_path = os.path.join(source_dir, file_name)

            # Open the .mrc file and flatten the data
            with mrcfile.open(source_path, permissive=True) as mrc:
                data = mrc.data
            data = data.flatten()

            # Define the output .raw file name
            raw_file_name = os.path.splitext(file_name)[0] + '.raw'
            dest_path = os.path.join(dest_dir, raw_file_name)

            # Save the flattened data as a .raw file
            data.tofile(dest_path)
            print(f"Saved: {dest_path}")

    print("All files have been processed.")


def micrograph2tomo(filepath):

    with mrcfile.open(filepath, permissive=True) as mrc:
        data = mrc.data

    new_data = np.expand_dims(data, axis=0)
    output_filepath = os.path.splitext(filepath)[0] + '_3d.mrc'

    with mrcfile.new(output_filepath, overwrite=True) as new_mrc:
        new_mrc.set_data(new_data.astype(np.float32))

if __name__ == "__main__":
    micrograph2tomo('/Volumes/homes/microscope/Pelin/Krios/241113_EKO/View_images/images12_binned.mrc')