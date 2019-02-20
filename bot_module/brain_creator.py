from keras.models import Model, model_from_json
from keras.layers import Input, CuDNNLSTM, Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras_contrib.callbacks.cyclical_learning_rate import *

class BrainCreator():
    def __init__(self, text_object):
        self.text_object = text_object
        self.X, self.y = text_object.X, text_object.y
        self.model = None

    def create_recurrent_model (self, hidden_units, dropout, optimizer):
        '''
        Method for creating the recurrent model (LSTM cells)

        Arguments:
            - hidden_units: is an iterable of integers storing the number of 'neurons' per layer
            - dropout_value: is a float reporting the percentage of neurons randomly masked to 0

        Returns:
            - None
        '''
        return_sequences = len(hidden_units) > 1
        model_input = Input(shape=(self.X.shape[1], self.X.shape[2]))
        for layer, units in enumerate(hidden_units):

            if layer == 0:
                recurrent = CuDNNLSTM(units
                                      , return_sequences=return_sequences
                                      )(model_input)
            elif layer == len(hidden_units) - 1:
                recurrent = CuDNNLSTM(units)(recurrent)
            else:
                recurrent = CuDNNLSTM(units
                                      , return_sequences=return_sequences
                                     )(recurrent)
        model_output = Dropout(dropout)
        model_output = Dense(self.y.shape[1])(recurrent)
        model_output = Activation('softmax')(model_output)
        model = Model(inputs=[model_input], outputs=[model_output])
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        self.model = model

    def train_model (self, weights_name, epochs, batch_size, verbose=1):
        '''
        Method for training the model

        Arguments:
            - weights_name: is a string specifying the name of the hdf5 file
            - epochs: is an integer specifying for how many iterations the model will train
            - batch_size: is an integer specifying the size of the mini-batch
            - verbose: is an integer specifying the verbosity of the printed callback

        Returns:
            - None
        '''
        # define the checkpoint
        save_location = 'loadings\\models_weights\\{}.hdf5'.format(weights_name)
        checkpoint = ModelCheckpoint(save_location, monitor='loss', verbose=verbose, save_best_only=True, mode='min')
        learning_rate_sched = CyclicLR(mode='triangular2')
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[checkpoint, learning_rate_sched])

    def save_model (self, model_name):
        '''
        Method for saving the model

        Arguments:
            - model_name: is a string specifying the name of the json file

        Returns:
            - None
        '''
        in_json = self.model.to_json()
        with open('loadings\\models_architectures\\{}.json'.format(model_name), 'w') as out_json:
            out_json.write(in_json)

    def load_model (self, model_name):
        '''
        Method for loading the model architecture

        Arguments:
            - model_name: is a string specifying the name of the json file

        Returns:
            - None
        '''
        in_json = open('loadings\\models_architectures\\{}.json'.format(model_name), 'r')
        self.model = model_from_json(in_json.read())
        in_json.close()

    def load_weights (self, weights_name):
        '''
        Method for loading the model weights

        Arguments:
            - weights_name: is a string specifying the name of the hdf5 file

        Returns:
            - None
        '''
        if self.model == None:
            print('Error: no architecture specified')
        else:
            self.model.load_weights('loadings\\models_weights\\{}.hdf5'.format(weights_name))
