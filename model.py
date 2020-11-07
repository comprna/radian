from tcn import TCN
from tensorflow.keras.layers import Dense, Activation, Lambda
from tensorflow.keras import Input, Model, backend
from tensorflow import Variable
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers import Adam

def ctc_loss_lambda(args):
    """
    This function is required because Keras currently doesn't support
    loss functions with additional parameters so it needs to be
    implemented in a lambda layer.
    """
    y_pred, labels, input_length, label_length = args
    return backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_optimizer(config):
    step = Variable(0, trainable = False)
    boundaries = [int(config.max_steps * bound) for bound in config.boundaries]
    values = [config.init_rate * decay for decay in config.decays]
    learning_rate_fn = PiecewiseConstantDecay(boundaries, values)
    return Adam(learning_rate=learning_rate_fn(step))

def initialise_model(model, opt, max_label_len):
    model = build_model(model, max_label_len)
    # optimizer = get_optimizer(opt)
    optimizer = Adam(learning_rate=0.004)
    model.compile(optimizer = optimizer,
                  loss = {'ctc': lambda labels, y_pred: y_pred})
    return model

def build_model(config, max_label_len):
    c = config
    input_shape = (c.timesteps, 1)

    inputs = Input(shape=input_shape, name="inputs") # (None, 512, 1)

    params = {'nb_filters': c.tcn.nb_filters,
              'kernel_size': c.tcn.kernel_size,
              'nb_stacks': c.tcn.nb_stacks,
              'dilations': c.tcn.dilations,
              'padding': c.tcn.padding,
              'use_skip_connections': c.tcn.use_skip_connections,
              'dropout_rate': c.tcn.dropout_rate,
              'return_sequences': c.tcn.return_sequences,
              'activation': c.tcn.activation,
              'kernel_initializer': c.tcn.kernel_initializer,
              'use_batch_norm': c.tcn.use_batch_norm}

    inner = TCN(**params)(inputs)   # (None, 512, 64)
    inner = Dense(c.relu_units)(inner) # (None, 512, 5)
    inner = Activation('relu')(inner)
    inner = Dense(c.softmax_units)(inner) # (None, 512, 5)
    y_pred = Activation('softmax')(inner) # (None, 512, 5)

    labels = Input(shape=(max_label_len,), name="labels") # (None, 39)
    input_length = Input(shape=[1],name="input_length") # (None, 1)
    label_length = Input(shape=[1],name="label_length") # (None, 1)

    loss_out = Lambda(ctc_loss_lambda, output_shape=(1,), name='ctc')((y_pred, labels, input_length, label_length))

    return  Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
