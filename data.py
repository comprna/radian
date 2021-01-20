import numpy as np
import tensorflow as tf

# Computed elsewhere
WINDOWS_PER_SHARD = 50000

def read_tfrecord(example_batch):
    features = {
        'signal': tf.io.FixedLenFeature([512], tf.float32), # TODO: Remove hardcoding
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
    outputs = {'ctc': np.zeros([256])} # TODO: Remove hardcoding

    return inputs, outputs

def get_dataset(shard_files, batch_size, val = False):
    AUTO = tf.data.experimental.AUTOTUNE

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    dataset = tf.data.Dataset.from_tensor_slices(shard_files)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                cycle_length=32, 
                                num_parallel_calls=AUTO)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=WINDOWS_PER_SHARD+1)
    if val == False:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(map_func = read_tfrecord,
                          num_parallel_calls = AUTO)
    dataset = dataset.prefetch(AUTO)
    return dataset