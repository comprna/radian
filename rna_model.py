from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.models import Sequential

N_BASES = 4
WINDOW_LENGTH = 8 ##### TODO: Remove hardcoding

def get_rna_prediction_model(checkpoint, config):
    model = build_model(config)
    model.load_weights(checkpoint)
    return model

def build_model(config):
    model = Sequential()

    if config.model.type == "lstm":
        params = {'activation': config.model.lstm.activation,
                  'recurrent_activation': config.model.lstm.recurrent_activation,
                  'use_bias': config.model.lstm.use_bias,
                  'kernel_initializer': config.model.lstm.kernel_initializer,
                  'recurrent_initializer': config.model.lstm.recurrent_initializer,
                  'bias_initializer': config.model.lstm.bias_initializer,
                  'unit_forget_bias': config.model.lstm.unit_forget_bias,
                  'dropout': config.model.lstm.dropout,
                  'recurrent_dropout': config.model.lstm.recurrent_dropout,
                  'return_state': config.model.lstm.return_state,
                  'go_backwards': config.model.lstm.go_backwards,
                  'stateful': config.model.lstm.stateful,
                  'time_major': config.model.lstm.time_major,
                  'unroll': config.model.lstm.unroll
                  }

        # Initial layers in stack should return sequences
        for i in range(config.model.lstm.layers-1):
            model.add(LSTM(config.model.lstm.units,
                        batch_input_shape=(1, WINDOW_LENGTH, N_BASES),
                        return_sequences=True,
                        **params))

        # Final layer should only return last value
        model.add(LSTM(config.model.lstm.units,
                    batch_input_shape=(1, WINDOW_LENGTH, N_BASES),
                    return_sequences=False,
                    **params))
        model.add(Dense(N_BASES, activation="softmax"))

    elif config.model.type == "gru":
        # Initial layers in stack should return sequences
        for i in range(config.model.gru.layers-1):
            model.add(GRU(config.model.gru.units,
                        batch_input_shape=(1, WINDOW_LENGTH, N_BASES),
                        return_sequences=True))

        # Final layer should only return last value
        model.add(GRU(config.model.gru.units,
                    batch_input_shape=(1, WINDOW_LENGTH, N_BASES),
                    return_sequences=False))
        model.add(Dense(N_BASES, activation="softmax"))

    else:
        print(f"Model type {config.model.type} in config file is invalid!")
    
    print(model.summary())
    return model

