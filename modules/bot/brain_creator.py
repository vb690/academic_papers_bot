import pickle

from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation

from ..utilities.model_utils import _AbstractModel


class AssembledBrain:
    '''
    Class assemblying a brain object from saved model and TextObject
    '''
    def __init__(self, theme):
        with open(
            'loadings\\text_objects\\{}.pkl'.format(theme), 'rb'
        ) as input_file:
            text_object = pickle.load(input_file)
        self.model = load_model('loadings\\models\\{}.h5'.format(theme))
        self.text_object = text_object


class BrainCreator(_AbstractModel):
    def __init__(self, X, y):
        self.X_shape, self.y_shape = X.shape, y.shape
        self.model = None

    def create_recurrent_model(self, recurrent_layers, fc_layers,
                               dropout_rate, optimizer, batch_size=256):
        '''
        Method for creating the recurrent model (LSTM cells)

        Arguments:
            - hidden_units: is an iterable of integers storing the number of
                            'neurons' per layer
            - dropout_value: is a float reporting the probability of a neuron
                             to be randomly masked to 0

        Returns:
            - None
        '''
        model_input = Input(shape=(None, self.X_shape[2]))
        recurrent_block = self.create_recurrent_block(
            input_tensor=model_input,
            CUDA=True,
            layers=recurrent_layers,
            temporal=False
        )
        fc_block = self.create_fc_block(
            input_tensor=recurrent_block,
            layers=fc_layers,
            activation='relu',
            dropout_rate=dropout_rate
        )
        model_output = Dense(self.y_shape[1])(fc_block)
        model_output = Activation('softmax')(model_output)
        model = Model(
            inputs=[model_input],
            outputs=[model_output]
        )
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer
        )
        self.model = model
