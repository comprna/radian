import ast
import numpy as np
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    # def __init__(self, csv_file, batch_size, window_size, eval_mode, max_label_len):
    #     self.csv_file = csv_file
    #     self.batch_size = batch_size
    #     self.window_size = window_size
    #     self.eval_mode = eval_mode
    #     self.max_label_len = max_label_len

    def __init__(self, list_ids, labels, batch_size, window_size, 
        max_label_len, shuffle=True):
        self.list_ids = list_ids
        self.labels = labels
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_label_len = max_label_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs in the batch
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data for the batch
        inputs, outputs = self.__data_generation(list_ids_temp)

        return (inputs, outputs)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        signals = np.zeros((self.batch_size, self.window_size, 1))
        labels = np.zeros((self.batch_size, self.max_label_len))
        signal_length = np.ones((self.batch_size, 1)) * self.window_size
        label_length = np.zeros((self.batch_size, 1))

        # Generate data
        for i, data_id in enumerate(list_ids_temp):
            signal = np.load('data2/data/' + data_id)
            signal = np.expand_dims(signal, -1)
            # signals[i,] = signal
            signals[i] = signal

            sequence = self.labels[data_id]
            label = self._sequence_to_label(sequence)
            labels[i][:len(label)] = label
            label_length[i] = len(label)
        
        inputs = {
            'inputs': signals,
            'labels': labels,
            'input_length': signal_length,
            'label_length': label_length
        }
        out = np.zeros([self.batch_size])
        outputs = {'ctc': out}

        return inputs, outputs


    ######### OLD IMPLEMENTATION

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
                    signal, label = self._extract_signal_label(line)
                    signals[i] = signal
                    labels[i][:len(label)] = label
                    label_length[i] = len(label)

                inputs = {
                    'inputs': signals,
                    'labels': labels,
                    'input_length': signal_length,
                    'label_length': label_length
                }
                out = np.zeros([self.batch_size])
                outputs = {'ctc': out}

                yield(inputs, outputs)
    
    def _extract_signal_label(self, line):
        signal, sequence = line.split('\t')
        signal = ast.literal_eval(signal)
        signal = np.expand_dims(signal, -1)

        sequence = ast.literal_eval(sequence)
        label = self._sequence_to_label(sequence)

        return signal, label

    def _sequence_to_label(self, sequence):
        bases = ['A', 'C', 'G', 'T']
        return list(map(lambda b: bases.index(b), sequence))
