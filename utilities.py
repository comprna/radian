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