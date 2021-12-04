from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Activation, Dropout

from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class _AbstractModel:
    """Protected class that will be inherited by the brain creator, it contains
    a series of utility functions for creating, saving and loading the
    computational grpah.
    """
    def create_fc_block(self, input_tensor, **kwargs):
        """Function for creating a block of fully connected layers

        Arguments:
            - input_tensor:
            - **kwargs:

        Returns:
            - fc:
        """
        if len(kwargs['layers']) == 0:
            return input_tensor
        for layer, units in enumerate(kwargs['layers']):

            if layer == 0:
                fc = Dense(units)(input_tensor)
                fc = Activation(kwargs['activation'])(fc)
                fc = Dropout(kwargs['dropout_rate'])(fc)
            else:
                fc = Dense(units)(input_tensor)
                fc = Activation(kwargs['activation'])(fc)
                fc = Dropout(kwargs['dropout_rate'])(fc)

        return fc

    def create_recurrent_block(self, input_tensor, **kwargs):
        """Function for creating a block of fully connected layers

        Arguments:
            - input_tensor:
            - **kwargs:

        Returns:
            - recurrent:
        """
        if len(kwargs['layers']) == 0:
            return input_tensor
        return_sequences = len(kwargs['layers']) > 1
        for layer, units in enumerate(kwargs['layers']):

            if layer == 0:
                recurrent = LSTM(
                    units,
                    return_sequences=return_sequences
                )(input_tensor)
            elif layer == len(kwargs['layers']) - 1:
                recurrent = LSTM(
                    units,
                    return_sequences=kwargs['temporal']
                )(recurrent)
            else:
                recurrent = LSTM(
                    units,
                    return_sequences=return_sequences
                )(recurrent)

        return recurrent

    def train_model(self, X, y, batch_size, epochs, weights_name, verbose=1,
                    **kwargs):
        """Method for training the model

        Arguments:
            - weights_name: is a string specifying the name of the hdf5 file
            - epochs: is an integer specifying for how many iterations the
              model will train
            - batch_size: is an integer specifying the size of the mini-batch
            - verbose: is an integer specifying the verbosity of the printed
              callback

        Returns:
            - None
        """
        # define the checkpoint
        save_location = 'loadings\\models_weights\\{}.hdf5'.format(
            weights_name
        )
        checkpoint = ModelCheckpoint(
            save_location,
            monitor='val_loss',
            verbose=verbose,
            save_best_only=True,
            mode='min',
        )
        stopper = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=10,
            restore_best_weights=True
        )
        self.model.fit(
            X,
            y,
            verbose=verbose,
            callbacks=[checkpoint, stopper],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            **kwargs
        )
        return None

    def save_model(self, model_name):
        """Method for saving the model as hdf5

        Arguments:
            - model_name: is a string specifying the name of the h5 file

        Returns:
            - None
        """
        self.model.save(
            'loadings\\models\\{}.h5'.format(model_name)
        )
        return None

    def save_model_weights(self, model_name):
        """Method for saving the model weights as hdf5

        Arguments:
            - model_name: is a string specifying the name of the h5 file

        Returns:
            - None
        """
        self.model.save_weights(
            'loadings\\models_weights\\{}.h5'.format(model_name)
        )
        return None

    def save_architecture(self, model_name):
        """Method for saving the model architecture

        Arguments:
            - model_name: is a string specifying the name of the json file

        Returns:
            - None
        """
        in_json = self.model.to_json()
        with open(
            'loadings\\models_architectures\\{}.json'.format(model_name), 'w'
        ) as out_json:
            out_json.write(in_json)
        return None

    def load_architecture(self, model_name):
        """Method for loading the model architecture

        Arguments:
            - model_name: is a string specifying the name of the json file

        Returns:
            - None
        """
        in_json = open(
            'loadings\\models_architectures\\{}.json'.format(model_name), 'r'
        )
        self.model = model_from_json(in_json.read())
        in_json.close()
        return None

    def load_weights(self, weights_name):
        """Method for loading the model weights

        Arguments:
            - weights_name: is a string specifying the name of the hdf5 file

        Returns:
            - None
        """
        if self.model is None:
            print('Error: no architecture specified')
        else:
            self.model.load_weights(
                'loadings\\models_weights\\{}.h5'.format(weights_name)
            )
        return None
