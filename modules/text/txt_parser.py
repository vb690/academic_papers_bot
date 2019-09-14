import pickle

import numpy as np

from keras.utils import to_categorical


class _TextObject():
    def __init__(self):
        '''
        Initialize a TextObject for storing from all the parsed txt files:

            - characters: an iterable of all the unique characters in the
              parsed files
            - X: an iterable containing sequences of characters parsed from
              the txt files that will be used for modelling the probability
              of the subsequent character
            - y: an iterable containing characters to be predicted based
            - starting seeds: an iterable containing all the starting sequences
              for each parsed sentence, they will be used as seeds for
              generating new text
            - char_int: a dictionary for converting the character to integer
            - int_char: a dictionary for converting the integer to character
        '''
        self.characters = []
        self.X = []
        self.y = []
        self.starting_seeds = []
        self.unit_inte = {}
        self.inte_unit = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['characters']
        del state['X']
        del state['y']
        return state

    def _save_text_object(self, txt_obj_name):
        with open('loadings\\text_objects\\{}.pkl'.format(txt_obj_name),
                  'wb'
                  ) as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


class TextParser():
    def __init__(self, names, sequence_len):
        '''
        Initialize a TextParser that will store the parsed text
        in a TextObject:

        Args:
            - text_path: is a string specifying where the txt files are located
            - names: is an iterable specifying the names of txt files to parse
            - sequence_len: is an integer specifying the length of the seed
              used for training the RNN

              Example sequence_len=5:
              'Lorem ipsum suas propriae ne name'
              [l, o, r, e, m] ---> [' ']
              [o, r, e, m,  ] ---> ['i']
              [r, e, m,  , i] ---> ['p']
                        ...
        '''
        self.names = names
        self.sequence_len = sequence_len
        self.text_object = _TextObject()

    @staticmethod
    def get_units(sentences, char=True):
        '''
        Method for extracting  a list of unique units (characters or words)
        from a list of sentences

        Args:
            - sentences: an iterable containing a series of sentences
              (stored as strings)

        Returns:
            - a sorted list of unique characters present in sentences
        '''
        units = []
        for sentence in sentences:

            if char:
                units.extend(list(sentence))
            else:
                units.extend(sentence.split())

        return sorted(list(set(units)))

    @staticmethod
    def create_unit_inte_codes(units):
        '''
        Method for creating a char_int and int_char dictionaries

        Args:
            - characters: an iterable containing all the unique characters
              from the parsed txt file

        Returns:
            - char_int: is a dictionary having as keys the characters and as
              value their relative code
            - int_char: is the reverse of char_int
        '''
        unit_int = {unit: inte for inte, unit in enumerate(units, 0)}
        int_char = {inte: unit for unit, inte in unit_int.items()}
        return unit_int, int_char

    @staticmethod
    def get_sequences(sentence, sequence_len, unit_inte):
        '''
        Method for dividing a sentence in training characters (X) of length
        sequence_len and target character (y)

        Args:
            - sentence: is a string
            - sequence_len: is an ineger
            - char_int: is a dictionary

        Returns:
            - X: is an iterable
            - y: is an iterable
        '''
        X = []
        y = []
        for index in range(len(sentence) - sequence_len):

            sequence = [unit_inte[unit] for unit
                        in sentence[index:index + sequence_len]
                        ]
            target = unit_inte[sentence[index + sequence_len]]
            X.append(sequence)
            y.append(target)

        return X, y

    def parse_text(self, text_path, txt_obj_name, normalize=True, save=True):
        '''
        Method for parsing the text and updating the TextObject

        Arguments:
            - save:
            - txt_obj_name:

        Returns:
            - text_object:
        '''
        sentences = []
        for topic in self.names:

            in_text = open('{}{}.txt'.format(text_path, topic),
                           encoding='utf-8',
                           mode='r'
                           )

            for sentence in in_text:

                sentences.append(sentence)

        self.text_object.characters = self.get_units(sentences)
        self.text_object.unit_inte, self.text_object.inte_unit = \
            self.create_unit_inte_codes(self.text_object.characters)
        for sentence in sentences:

            X, y = self.get_sequences(
                        list(sentence),
                        self.sequence_len,
                        self.text_object.unit_inte
                    )
            if len(X) == 0:
                continue
            self.text_object.starting_seeds.append(X[0])
            self.text_object.X.extend(X)
            self.text_object.y.extend(y)

        if normalize:
            self.text_object.X = \
                np.array(self.text_object.X) / len(self.text_object.unit_inte)
        self.text_object.X = np.reshape(self.text_object.X,
                                        (len(self.text_object.X),
                                         self.sequence_len, 1)
                                        )
        self.text_object.y = to_categorical(np.array(self.text_object.y))

        setattr(self.text_object, 'X_shape', self.text_object.X.shape)
        setattr(self.text_object, 'y_shape', self.text_object.y.shape)
        if save:
            self.text_object._save_text_object(txt_obj_name)
        return self.text_object
