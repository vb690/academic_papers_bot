from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint

class ModelCreator():
    def __init__(self, text_object, model_path, weights_path):
        self.text_object = text_object
        self.X, self.y = text_object.get_normalized()
        self.weights_path = weights_path
        self.model_path = model_path
        self.model = None

    def create_recurrent_model (self,  hidden_units, dropout_value):
        '''
        Method for creating the recurrent model (LSTM cells)

        Arguments:
            - hidden_units: is an iterable of integers storing the number of 'neurons' per layer
            - dropout_value: is a float reporting the percentage of neurons randomly masked to 0

        Returns:
            - None
        '''
        model = Sequential()
        for layer, neurons in enumerate(hidden_units, 1):

            if layer == 1:
                model.add(LSTM(neurons, input_shape=(self.X.shape[1], self.X.shape[2]), return_sequences=True))
                model.add(Dropout(dropout_value))
            elif layer == len(hidden_units):
                model.add(LSTM(neurons))
                model.add(Dense(self.y.shape[1], activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
            else:
                model.add(LSTM(neurons, return_sequences=True))
                model.add(Dropout(dropout_value))

        self.model = model

    def train_model (self, weights_name, epochs, batch_size):
        '''
        '''
        # define the checkpoint
        save_location = '{}{}.hdf5'.format(self.weights_path, weights_name)
        checkpoint = ModelCheckpoint(save_location, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self. model.fit(self.X, self.y, epochs = epochs, batch_size=batch_size, verbose = 1, callbacks=[checkpoint])

    def save_model (self, model_name):
        '''
        '''
        in_json = self.model.to_json()
        with open('{}{}.json'.format(self.model_path, model_name), 'w') as out_json:
            out_json.write(in_json)

    def load_model (self, model_name):
        '''
        '''
        in_json = open('{}{}.json'.format(self.model_path, model_name), 'r')
        self.model = model_from_json(in_json.read())
        in_json.close()

    def load_weights (self, weights_name):
        '''
        '''
        if self.model == None:
            print('Error: no model specified')
        else:
            self.model.load_weights('{}{}.hdf5'.format(self.weights_path, weights_name))
