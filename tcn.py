import tensorflow as tf
from tf.keras.layers import Dense, Activation, Lambda
from tf.keras import Input, Model
from tf.keras.backend import ctc_batch_cost
from tcn import TCN, tcn_full_summary
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
    labels, y_pred, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)

def main():
    # Load data
    train_x = [] # TBD
    train_y = [] # TBD
    
    # Get info about data
    max_label_length = 1 # TODO
    num_classes = 5 # TODO

    # Prepare TCN inputs
    batch_size, timesteps, input_dim = None, 512, 1 # TBD
    inputs = Input(batch_shape=(batch_size, timesteps, input_dim), name="signal_input")

    labels = tf.keras.layers.Input((max_label_length,), name="labels")
    label_length = tf.keras.layers.Input((1,),name="label_length")
    input_length = tf.keras.layers.Input((1,),name="input_length")

    # Prepare model
    outputs = TCN(return_sequences=False)(inputs)
    outputs = Dense(num_classes)(outputs)
    y_pred = Activation('softmax')(outputs)
    loss_out = Lambda(ctc_loss_lambda, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])

    # Compile model
    model.compile(optimizer='adam', loss={'ctc': lambda labels, y_pred: y_pred})

    # Training
    tcn_full_summary(model, expand_residual_blocks=False)
    model.fit(train_x, train_y, epochs=10, validation_split=0.2)

if __name__ == "__main__":
    main()
