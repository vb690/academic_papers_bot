import numpy as np

from keras.utils import to_categorical

class TextObject():
    def __init__(self):
        '''
        Initialize a TextObject for storing from all the parsed txt files:

            - characters: an iterable of all the unique characters in the parsed files
            - X: an iterable containing sequences of characters parsed from the txt files that will be used for modelling the probability of the subsequent character
            - y: an iterable containing characters to be predicted based
            - starting seeds: an iterable containing all the starting sequences for each parsed sentence, they will be used as seeds for generating new text
            - char_int: a dictionary for converting the character to integer
            - int_char: a dictionary for converting the integer to character
        '''
        self.characters = []
        self.X = []
        self.y = []
        self.starting_seeds = []
        self.char_int = {}
        self.int_char = {}

    def get_normalized(self):
        X = self.X / len(self.characters)
        y = to_categorical(self.y)
        return X, y

class TextParser():
    def __init__(self, text_path, names, sequence_len):
        '''
        Initialize a TextParser that will store in a TextObject:

        Args:
            - text_path: is a string specifying where the txt files are located
            - names: is an iterable specifying the names of txt files to parse
            - sequence_len: is an integer specifying the length
        '''
        self.text_path = text_path
        self.names = names
        self.sequence_len = sequence_len
        self.text_object = TextObject()

    @staticmethod
    def get_characters(sentences):
        '''
        Method for extracting  a list of unique characters from a list of sentences

        Args:
            - sentences: an iterable containing a series of sentences (stored as strings)

        Returns:
            - a sorted list of unique characters present in sentences
        '''
        characters = []
        for sentence in sentences:

            characters.extend(list(sentence))

        return sorted(list(set(characters)))

    @staticmethod
    def create_char_int_codes(characters):
        '''
        Method for creating a char_int and int_char dictionaries

        Args:
            - characters: an iterable containing all the unique characters from the parsed txt file

        Returns:
            - char_int: is a dictionary having as keys the characters and as value their relative code
            - int_char: is the reverse of char_int
        '''
        normalize_factor = len(characters)
        char_int = {character : integer for integer, character in enumerate(characters, 0)}
        int_char = {integer : character for character, integer in char_int.items()}
        return char_int, int_char

    @staticmethod
    def get_sequences(sentence, sequence_len, char_int):
        '''
        Method for dividing a sentence in training characters (X) of length sequence_len and target character (y)

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

            sequence = [char_int[character] for character in sentence[index:index + sequence_len]]
            target = char_int[sentence[index + sequence_len]]
            X.append(sequence)
            y.append(target)

        return X, y

    def parse_text(self):
        sentences = []
        for topic in self.names:

            in_text =  open('{}{}.txt'.format(self.text_path, topic), encoding = 'utf-8', mode = 'r')

            for sentence in in_text:

                sentences.append(sentence)

        self.text_object.characters = self.get_characters(sentences)
        self.text_object.char_int, self.text_object.int_char = self.create_char_int_codes(self.text_object.characters)
        for sentence in sentences:

                X, y = self.get_sequences(list(sentence), self.sequence_len, self.text_object.char_int)
                self.text_object.starting_seeds.append(X[0])
                self.text_object.X.extend(X)
                self.text_object.y.extend(y)

        self.text_object.X = np.array(self.text_object.X)
        self.text_object.X = np.reshape(self.text_object.X, (len(self.text_object.X), self.sequence_len, 1))
        self.text_object.y = np.array(self.text_object.y)
        return self.text_object
