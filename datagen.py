import ast
import h5py
import numpy as np
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
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
        with h5py.File('/home/alex/Documents/rnabasecaller/data3/data.h5', 'r') as h5:
            for i, data_id in enumerate(list_ids_temp):
                signal = h5[data_id]['signal'][()]
                signal = np.expand_dims(signal, -1)
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
    
    def _sequence_to_label(self, sequence):
        bases = ['A', 'C', 'G', 'T']
        return list(map(lambda b: bases.index(b), sequence))
