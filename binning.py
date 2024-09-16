import numpy as np
import mrcfile


def process_chunk(tomogram_chunk, bin_factor):
    """
    Bin (downsample) a tomogram chunk by a given bin factor.
    """
    # Calculate new shape after binning
    new_shape = (tomogram_chunk.shape[0] // bin_factor,
                 tomogram_chunk.shape[1] // bin_factor,
                 tomogram_chunk.shape[2] // bin_factor)

    # Truncate to make sure the dimensions are divisible by bin_factor
    truncated_chunk = tomogram_chunk[:new_shape[0] * bin_factor,
                                     :new_shape[1] * bin_factor,
                                     :new_shape[2] * bin_factor]

    # Reshape to group voxels by bin_factor and then take the mean
    reshaped_chunk = truncated_chunk.reshape(new_shape[0], bin_factor,
                                             new_shape[1], bin_factor,
                                             new_shape[2], bin_factor)

    # Average over the grouped voxels
    binned_chunk = reshaped_chunk.mean(axis=(1, 3, 5))

    return binned_chunk


def process_in_chunks(filename, chunk_size, overlap, bin_factor):
    """
    Process the tomogram in chunks with overlap and bin the data.
    """
    with mrcfile.open(filename, permissive=True) as mrc:
        tomogram = mrc.data

    z_chunks = list(range(0, tomogram.shape[0], chunk_size[0] - overlap[0]))
    y_chunks = list(range(0, tomogram.shape[1], chunk_size[1] - overlap[1]))
    x_chunks = list(range(0, tomogram.shape[2], chunk_size[2] - overlap[2]))

    binned_chunks = []

    for z in z_chunks:
        z_end = min(z + chunk_size[0], tomogram.shape[0])
        for y in y_chunks:
            y_end = min(y + chunk_size[1], tomogram.shape[1])
            for x in x_chunks:
                x_end = min(x + chunk_size[2], tomogram.shape[2])

                # Extract the chunk including overlap
                chunk = tomogram[z:z_end, y:y_end, x:x_end]

                # Pad the chunk to the expected size if it's smaller along any dimension (edge chunks)
                padded_chunk = np.pad(chunk,
                                      ((0, chunk_size[0] - chunk.shape[0]),
                                       (0, chunk_size[1] - chunk.shape[1]),
                                       (0, chunk_size[2] - chunk.shape[2])),
                                      mode='constant', constant_values=0)

                # Process the chunk (e.g., binning)
                binned_chunk = process_chunk(padded_chunk, bin_factor)

                # Store the binned chunk
                binned_chunks.append(binned_chunk)

    # Concatenate the chunks along the Z axis
    final_binned_volume = np.concatenate(binned_chunks, axis=0)

    return final_binned_volume


def bin_tomogram(filename, chunk_size=(64, 64, 64), overlap=(32, 32, 32), bin_factor=2):
    """
    Bin the entire tomogram by processing it in chunks.
    """
    binned_tomogram = process_in_chunks(filename, chunk_size, overlap, bin_factor)

    split_filename = filename.split('.')
    output_filename = f'{split_filename[0]}_{bin_factor}x_binned.mrc'

    # Save the binned tomogram to a new MRC file
    with mrcfile.new(output_filename, overwrite=True) as mrc:
        mrc.set_data(binned_tomogram.astype(np.float32))

    print(f'Binned tomogram saved to {output_filename}')


if __name__ == "__main__":
    bin_tomogram('sphere.mrc')
