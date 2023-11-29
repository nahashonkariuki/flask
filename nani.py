import h5py
import gzip

# Open the original .h5 file
with h5py.File('magnetic_tile_defect_model.h5', 'r') as original_file:
    # Create a compressed file with a .gz extension
    with gzip.open('compressed_file.h5.gz', 'wb') as compressed_file:
        # Compress each dataset in the original file
        for name, dataset in original_file.items():
            # Convert the dataset to a NumPy array
            data = dataset.value
            # Compress the NumPy array using gzip
            compressed_data = gzip.compress(data.tobytes())
            # Create a new dataset in the compressed file
            compressed_file.create_dataset(name, data=compressed_data, compression='gzip')