from tcn import TCN

import tensorflow as tf
from tensorflow import Variable
from tensorflow.keras import Input, Model, backend
from tensorflow.keras.backend import ctc_batch_cost, get_value, set_value
from tensorflow.keras.layers import Dense, Activation, Lambda, LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, Adagrad
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

MAX_LABEL_LEN = 25 # 3 / 256 / 32
# MAX_LABEL_LEN = 24 # 3 / 256 / 64
# MAX_LABEL_LEN = 38 # 3 / 512 / 64,128
# MAX_LABEL_LEN = 63 # 3 / 1024 / 64,128


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
                  loss = {'ctc': lambda labels, y_pred: y_pred})
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

def get_evaluation_model(config, weights):
    model = build_model(config.model, train=False)
    model.set_weights(weights)
    return model

def build_model(config, train=True):
    c = config
    input_shape = (c.timesteps, 1)

    inputs = Input(shape=input_shape, name="inputs")

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

    inner = TCN(**params)(inputs)
    inner = Dense(c.relu_units)(inner)
    inner = Activation('relu')(inner)
    inner = Dense(c.softmax_units)(inner)
    y_pred = Activation('softmax')(inner)

    labels = Input(shape=(MAX_LABEL_LEN,), name="labels")
    input_length = Input(shape=[1],name="input_length")
    label_length = Input(shape=[1],name="label_length")

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
    opt = config.type
    if opt == 'cc_opt':
        return get_causalcall_optimizer(config.cc_opt)
    elif opt == 'adam':
        if config.adam.clipnorm != False:
            return Adam(learning_rate=config.adam.lr,
                        beta_1=config.adam.beta_1,
                        beta_2=config.adam.beta_2,
                        epsilon=config.adam.epsilon,
                        amsgrad=config.adam.amsgrad,
                        clipnorm=config.adam.clipnorm
                        )
        if config.adam.clipvalue != False:
            return Adam(learning_rate=config.adam.lr,
                        beta_1=config.adam.beta_1,
                        beta_2=config.adam.beta_2,
                        epsilon=config.adam.epsilon,
                        amsgrad=config.adam.amsgrad,
                        clipvalue=config.adam.clipvalue
                        )
        else:
            return Adam(learning_rate=config.adam.lr,
                        beta_1=config.adam.beta_1,
                        beta_2=config.adam.beta_2,
                        epsilon=config.adam.epsilon,
                        amsgrad=config.adam.amsgrad
                        )
    elif opt == 'sgd':
        if config.sgd.clipnorm != False:
            return SGD(learning_rate = config.sgd.lr,
                       momentum = config.sgd.momentum,
                       nesterov = config.sgd.nesterov,
                       clipnorm = config.sgd.clipnorm)
        elif config.sgd.clipvalue != False:
            return SGD(learning_rate = config.sgd.lr,
                       momentum = config.sgd.momentum,
                       nesterov = config.sgd.nesterov,
                       clipvalue = config.sgd.clipvalue)
        else:
            return SGD(learning_rate = config.sgd.lr,
                       momentum = config.sgd.momentum,
                       nesterov = config.sgd.nesterov)
    elif opt == 'adagrad':
        return Adagrad(learning_rate = config.adagrad.lr)

# TODO: Fix up passing config object around
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
