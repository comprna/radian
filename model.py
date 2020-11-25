from tcn import TCN

import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras import Input, Model, backend
from tensorflow.keras.backend import ctc_batch_cost, get_value, set_value
from tensorflow.keras.layers import Dense, Activation, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# Computed elsewhere
MAX_LABEL_LEN = 46

def create_sparse(ten):
    # n = len(ten)
    # ind = [[xi, 0, yi] for xi,x in enumerate(ten) for yi,y in enumerate(x)]
    # chars = list(''.join(ten))
    # return tf.SparseTensorValue(ind, chars, [n,1,1])
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(ten, zero)
    indices = tf.where(where)
    values = tf.gather_nd(ten, indices)
    sparse = tf.compat.v1.SparseTensorValue(indices, values, ten.shape)
    print(sparse)
    return sparse

def ed(y_true,y_pred):
    print("Y_pred:")
    print(y_pred)
    print(type(y_pred))
    print(len(y_pred))

    print("\n\ny_true:")
    print(y_true)
    print(type(y_true))
    print(len(y_true))
    return tf.edit_distance(create_sparse(y_pred), create_sparse(y_true), normalize=True)

def get_training_model(checkpoint, epoch_to_resume, config):
    if checkpoint is not None:
        model = restore_checkpoint(checkpoint, config)
        # update_learning_rate(model, config.train.opt.lr)
        initial_epoch = epoch_to_resume
    else:
        model = initialise_model(config)
        initial_epoch = 0
    return model, initial_epoch

def initialise_model(config):
    model = build_model(config.model, train=True)
    optimizer = get_optimizer(config.train.opt)
    model.compile(optimizer = optimizer,
                  loss = {'ctc': lambda labels, y_pred: y_pred},
                  metrics = [ed])
    return model

def restore_checkpoint(checkpoint, config):
    model = build_model(config.model, train=True)
    model.load_weights(checkpoint)
    print("Loaded checkpoint {0}".format(checkpoint))
    optimizer = get_optimizer(config.train.opt)
    model.compile(optimizer = optimizer,
                  loss = {'ctc': lambda labels, y_pred: y_pred})
    return model

def get_prediction_model(checkpoint, config):
    model = build_model(config.model, train=False)
    model.load_weights(checkpoint)
    return model

def build_model(config, train=True):
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
              'use_batch_norm': c.tcn.use_batch_norm,
              }

    inner = TCN(**params)(inputs)   # (None, 512, 64)
    inner = Dense(c.relu_units)(inner) # (None, 512, 5)
    inner = Activation('relu')(inner)
    inner = Dense(c.softmax_units)(inner) # (None, 512, 5)
    y_pred = Activation('softmax')(inner) # (None, 512, 5)

    labels = Input(shape=(MAX_LABEL_LEN,), name="labels") # (None, 39)
    input_length = Input(shape=[1],name="input_length") # (None, 1)
    label_length = Input(shape=[1],name="label_length") # (None, 1)

    loss_out = Lambda(
        ctc_loss_lambda, output_shape=(1,), name='ctc')((
            y_pred, labels, input_length, label_length))

    if train == True:
        return Model(inputs=[inputs, labels, input_length, label_length],
                     outputs=[loss_out])
    else:
        return Model(inputs=[inputs], outputs=y_pred)  

def ctc_loss_lambda(args):
    """
    This function is required because Keras currently doesn't support
    loss functions with additional parameters so it needs to be
    implemented in a lambda layer.
    """
    y_pred, labels, input_length, label_length = args
    return ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_optimizer(config):
    if config.use_cc_opt == True:
        return get_causalcall_optimizer(config.cc_opt)
    else:
        return Adam(learning_rate=config.adam.lr,
                    beta_1=config.adam.beta_1,
                    beta_2=config.adam.beta_2,
                    epsilon=config.adam.epsilon,
                    amsgrad=config.adam.amsgrad
                    )

def get_causalcall_optimizer(config):
    c = config
    step = Variable(0, trainable = False)
    boundaries = [int(c.max_steps * bound) for bound in c.boundaries]
    values = [c.init_rate * decay for decay in c.decays]
    learning_rate_fn = PiecewiseConstantDecay(boundaries, values)
    return Adam(learning_rate=learning_rate_fn(step))

def update_learning_rate(model, new_rate):
    print("Old learning rate: {}".format(get_value(model.optimizer.lr)))
    set_value(model.optimizer.lr, new_rate)
    print("New learning rate: {}".format(get_value(model.optimizer.lr)))
