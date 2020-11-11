import ast
import csv
import tensorflow as tf
import time

def benchmark(dataset, num_epochs=1):
    start_time = time.perf_counter()
    # for epoch_num in range(num_epochs):
    for s in dataset:
        pass
    tf.print("execution time: {0}".format(time.perf_counter() - start_time))

def count_training_size(dataset):
    n = 0
    for sample in dataset:
        n += 1
    print(n)

def get_data_info():
    with open('hek293-fold1.csv', "r") as f:
          i = 0
          max_len = 0
          while True:
              line = f.readline()
              i += 1
              # Move back to beginning of file if at end
              if line == "":
                  print("End of file at line {0}".format(i))
                  break

              signal, sequence = line.split('\t')
              sequence = ast.literal_eval(sequence)
              
              len_sequence = len(sequence)
              if len_sequence > max_len:
                  max_len = len_sequence
          
          print("Maximum sequence length: {0}".format(max_len))

def setup_local():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
