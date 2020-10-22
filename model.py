import tensorflow as tf
import tcn
import functools

def ctc_loss_lambda(args):
    """
    This function is required because Keras currently doesn't support
    loss functions with additional parameters so it needs to be
    implemented in a lambda layer.

    y_true:  Tensor (samples, max_string_length) containing the truth 
        labels.
        --> samples = batch_size
        --> These labels should be padded to the length of the longest
            label in the batch.  In the "padded slots" any value can 
            be used, because ctc_batch_cost internally combines
            y_true with label_length to produce a sparce tensor, so any
            values in slots beyond label_length will be ignored.
    y_pred:	 Tensor (samples, time_steps, num_categories) containing 
        the prediction, or output of the softmax.
    input_length:  Tensor (samples, 1) containing the sequence length 
        for each batch item in y_pred.
    label_length:  Tensor (samples, 1) containing the sequence length 
        for each batch item in y_true.
    """
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_model(batch_size, timesteps, max_label_len, num_classes):
    input_shape = (timesteps, 1)
    
    inputs = tf.keras.Input(shape=input_shape, name="inputs") # (None, 512, 1)

    inner = tcn.TCN(return_sequences=True)(inputs) # (None, 512, 64)
    inner = tf.keras.layers.Dense(num_classes)(inner) # (None, 512, 5)
    y_pred = tf.keras.layers.Activation('softmax')(inner) # (None, 512, 5)

    labels = tf.keras.Input(shape=(max_label_len,), name="labels") # (None, 39)
    input_length = tf.keras.Input(shape=[1],name="input_length") # (None, 1)
    label_length = tf.keras.Input(shape=[1],name="label_length") # (None, 1)

    loss_out = tf.keras.layers.Lambda(ctc_loss_lambda, output_shape=(1,), name='ctc')((y_pred, labels, input_length, label_length))
    
    return  tf.keras.Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
