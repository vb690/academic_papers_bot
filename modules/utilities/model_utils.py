from keras.layers import LSTM, CuDNNLSTM, Dense
from keras.layers import Activation, Dropout, BatchNormalization

from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras_contrib.callbacks.cyclical_learning_rate import CyclicLR


class _AbstractModel:
    '''
    Protected class that will be inherited by the brain creator, it contains
    a series of utility functions.
    '''
    def create_fc_block(self, input_tensor, **kwargs):
        '''
        Function for creating a block of fully connected layers

        Arguments:
            - input_tensor:
            - **kwargs:

        Returns:
            - fc:
        '''
        if len(kwargs['layers']) == 0:
            return input_tensor
        for layer, units in enumerate(kwargs['layers']):

            if layer == 0:
                fc = Dense(units)(input_tensor)
                fc = Activation(kwargs['activation'])(fc)
                fc = BatchNormalization()(fc)
                fc = Dropout(kwargs['dropout_rate'])(fc)
            else:
                fc = Dense(units)(input_tensor)
                fc = Activation(kwargs['activation'])(fc)
                fc = BatchNormalization()(fc)
                fc = Dropout(kwargs['dropout_rate'])(fc)

        return fc

    def create_recurrent_block(self, input_tensor, CUDA=True, **kwargs):
        '''
        Function for creating a block of fully connected layers

        Arguments:
            - input_tensor:
            - CUDA:
            - **kwargs:

        Returns:
            - recurrent:
        '''
        if len(kwargs['layers']) == 0:
            return input_tensor
        if CUDA:
            lstm = CuDNNLSTM
        else:
            lstm = LSTM
        return_sequences = len(kwargs['layers']) > 1
        for layer, units in enumerate(kwargs['layers']):

            if layer == 0:
                recurrent = lstm(
                    units,
                    return_sequences=return_sequences
                )(input_tensor)
            elif layer == len(kwargs['layers']) - 1:
                recurrent = lstm(
                    units,
                    return_sequences=kwargs['temporal']
                )(recurrent)
            else:
                recurrent = lstm(
                    units,
                    return_sequences=return_sequences
                )(recurrent)

        return recurrent

    def train_model(self, X, y, batch_size, epochs, weights_name, verbose=1,
                    **kwargs):
        '''
        Method for training the model

        Arguments:
            - weights_name: is a string specifying the name of the hdf5 file
            - epochs: is an integer specifying for how many iterations the
              model will train
            - batch_size: is an integer specifying the size of the mini-batch
            - verbose: is an integer specifying the verbosity of the printed
              callback

        Returns:
            - None
        '''
        # define the checkpoint
        save_location = 'loadings\\models_weights\\{}.hdf5'.format(
            weights_name
        )
        checkpoint = ModelCheckpoint(
            save_location,
            monitor='loss',
            verbose=verbose,
            save_best_only=True,
            mode='min',
        )
        learning_rate_sched = CyclicLR(
            mode='triangular2',
            step_size=(X.shape[0] // batch_size) * 2
        )
        self.model.fit(
            X,
            y,
            verbose=verbose,
            callbacks=[checkpoint, learning_rate_sched],
            batch_size=batch_size,
            epochs=epochs,
            **kwargs
        )

    def save_model(self, model_name):
        '''
        Method for saving the model as hdf5

        Arguments:
            - model_name: is a string specifying the name of the h5 file

        Returns:
            - None
        '''
        self.model.save(
            'loadings\\models\\{}.h5'.format(model_name)
        )

    def save_model_weights(self, model_name):
        '''
        Method for saving the model weights as hdf5

        Arguments:
            - model_name: is a string specifying the name of the h5 file

        Returns:
            - None
        '''
        self.model.save_weights(
            'loadings\\models_weights\\{}.h5'.format(model_name)
        )

    def save_architecture(self, model_name):
        '''
        Method for saving the model architecture

        Arguments:
            - model_name: is a string specifying the name of the json file

        Returns:
            - None
        '''
        in_json = self.model.to_json()
        with open(
            'loadings\\models_architectures\\{}.json'.format(model_name), 'w'
        ) as out_json:
            out_json.write(in_json)

    def load_architecture(self, model_name):
        '''
        Method for loading the model architecture

        Arguments:
            - model_name: is a string specifying the name of the json file

        Returns:
            - None
        '''
        in_json = open(
            'loadings\\models_architectures\\{}.json'.format(model_name), 'r'
        )
        self.model = model_from_json(in_json.read())
        in_json.close()

    def load_weights(self, weights_name):
        '''
        Method for loading the model weights

        Arguments:
            - weights_name: is a string specifying the name of the hdf5 file

        Returns:
            - None
        '''
        if self.model is None:
            print('Error: no architecture specified')
        else:
            self.model.load_weights(
                'loadings\\models_weights\\{}.h5'.format(weights_name)
            )
