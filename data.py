import numpy as np
import tensorflow as tf
from tensorflow.io.gfile import glob


# Computed elsewhere
WINDOWS_PER_SHARD = 50000

def read_tfrecord(example_batch):
    features = {
        'signal': tf.io.FixedLenFeature([1024], tf.float32), # TODO: Window size - Remove hardcoding
        'label': tf.io.VarLenFeature(tf.float32),
        'signal_length': tf.io.FixedLenFeature([], tf.int64), # shape [] means scalar
        'label_length': tf.io.FixedLenFeature([], tf.int64)
    }

    tf_record = tf.io.parse_example(example_batch, features)
    
    signal = tf_record['signal']
    label = tf.sparse.to_dense(tf_record['label']) # VarLenFeature decoding required
    signal_length = tf_record['signal_length']
    label_length = tf_record['label_length']

    inputs = {
        'inputs': signal,
        'labels': label,
        'input_length': signal_length,
        'label_length': label_length
    }
    outputs = {'ctc': np.zeros([32])} # TODO: Batch size - Remove hardcoding

    return inputs, outputs

def get_dataset(shard_files, batch_size, val = False):
    # Dynamically tune the level of parallelism.
    AUTO = tf.data.experimental.AUTOTUNE

    # Construct the dataset from the shard files.
    dataset = tf.data.Dataset.from_tensor_slices(shard_files)

    # Training datasets need to be shuffled.
    if val == False:
        option_no_order = tf.data.Options()
        option_no_order.experimental_deterministic = False
        dataset = dataset.with_options(option_no_order)
    else:
        option_order = tf.data.Options()
        option_order.experimental_deterministic = True
        dataset = dataset.with_options(option_order)

    # Process multiple input files concurrently.
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                cycle_length=32, 
                                num_parallel_calls=AUTO)

    # Cache the dataset to save loading time.
    dataset = dataset.cache()

    # Training datasets need to be shuffled and infinitely repeated.
    if val == False:
        dataset = dataset.shuffle(buffer_size=WINDOWS_PER_SHARD+1)
        dataset = dataset.repeat()
    
    # Produce batches of data.
    dataset = dataset.batch(batch_size)

    # Generate the dataset using read_tfrecord function.
    dataset = dataset.map(map_func = read_tfrecord,
                          num_parallel_calls = AUTO)
    
    # Overlap data processing and model training to improve performance
    dataset = dataset.prefetch(AUTO)
    
    return dataset

if __name__ == "__main__":
    shard_dir = "/Users/alexsneddon/Downloads"
    batch_size = 32

    data_files = glob("{}/*.tfrecords".format(shard_dir))
    dataset = get_dataset(data_files, batch_size, val=True)

    for sample in dataset:
        print(sample)
