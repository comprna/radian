import ast
import numpy as np

class DataGenerator:
    def __init__(self, csv_file, batch_size, window_size, eval_mode, max_label_len):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.window_size = window_size
        self.eval_mode = eval_mode
        self.max_label_len = max_label_len

    def next_batch(self):
        with open(self.csv_file, "r") as f:
            while True: # Generator must always return something
                # Initialise batch
                signals = np.zeros((self.batch_size, self.window_size, 1))
                labels = np.zeros((self.batch_size, self.max_label_len))
                signal_length = np.ones((self.batch_size, 1)) * self.window_size
                label_length = np.zeros((self.batch_size, 1))

                # Get the next batch
                for i in range(self.batch_size):
                    line = f.readline()
                    # Move back to beginning of file if at end
                    if line == "":
                        f.seek(0)
                        line = f.readline()

                        # But if we are evaluating, then we only need to go through test set once
                        if self.eval_mode == True:
                            break
                    
                    # Extract signal and label
                    signal, label = line.split('\t')
                    signal = ast.literal_eval(signal)
                    label = ast.literal_eval(label)

                    signals[i] = signal
                    labels[i] = label
                    label_length[i] = len(label)

                inputs = {
                    'input': signals,
                    'labels': labels,
                    'input_length': signal_length,
                    'label_length': label_length
                }
                outputs = {'ctc': np.zeros((self.batch_size))}
                yield(inputs, outputs)
