from copy import deepcopy

import random
import numpy as np

from txt_parser import *
from model_creator import *

class GeneralBot():
    def __init__(self, brain):
        '''
        Instatiate a GeneralBot object which will use the trained RNN stored in the ModelCreator for producing new paper titles

        Arguments:
            - titles: in an iterable storing the generated titles
            - brain: is a ModelCreator object storing a trained RNN and a TextObject
        '''
        self.titles = []
        self.model = brain.model
        self.text_object = brain.text_object

    @staticmethod
    def temperated_prediction(predictions, temperature=1.0):
        '''
        Static method for sampling a prediction from temperated output of the softmax function
        Higher temperature = more variability, more randomness, more errors
        Lower temperature  = less variability, more repetition, less errors

        Arguments:
            - predictions: is an iterable of probability over the available characters provided by the softmax function
            - temperature: is a float controlling the probability of each available characters

        Returns:
            - The argmax of the temperated softmax output
        '''
        predictions = np.asarray(predictions[0]).astype('float64')
        # take the log of the probabilities
        predictions = np.log(predictions) / temperature
        # take the exponentiated result
        exp_preds = np.exp(predictions)
        p_preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, p_preds)
        return np.argmax(probas)

    @staticmethod
    def random_seed_sampling(starting_seeds):
        '''
        Static method for sampling at random starting seed from all the possible ones

        Arguments:
            - starting_seed: is an iterable of starings containing all the starting seeds

        Returns:
            - A string sampled at random from starting_seed
        '''

        index = np.random.randint(len(starting_seeds))
        return starting_seeds[index]

    def bot_thinking(self, total_titles, temperature):
        '''
        Method for generating titles with a specific temperature and storing them in titles attribute

        Args:
            - total_titles: is an integer specifying the total number of titles that has to be generated
            - temperature: is a float specifying the temperature parameter for the temperated_prediction function

        Returns:
            - None
        '''
        counter_titles = 0
        starting_seeds = self.text_object.starting_seeds
        # we keep generating titles untill we have a sufficient number of 'good ones'
        while counter_titles != total_titles:

            predicted_character = None
            seed = deepcopy(self.random_seed_sampling(starting_seeds))
            title = [self.text_object.int_char[character] for character in seed]
            # we keep generating titles untill an EOS is generated
            while predicted_character != '\n':

                X = np.reshape(seed, (1, len(seed), 1))
                predicted_character = self.model.predict(X, verbose=0)
                predicted_character = self.temperated_prediction(predicted_character, temperature)
                seed.append(predicted_character)
                title.append(self.text_object.int_char[predicted_character])
                seed = seed[1:len(seed)]
                # stop generating if the title is longer than 100 characters (it needs to fit in a tweet)
                if len(title) > 100:
                    break

            # if the generated title is too short, discard it and generate a new one without updating the count
            if len(title) < 15:
                continue

            self.titles.append(''.join(title))
            counter_titles +=1

    def bot_proposing(self):
        '''
        '''
        for index, title in enumerate(self.titles):

            print('[{}] {} \n'.format(index, title))

        selected = input('Select title: ')
        return int(selected)

    def bot_speaking(self):
        '''
        Method for for printing the genrated paper titles
        '''
        for paper in self.titles:

            print('Hey! This seems cool, check it out!')
            print(paper)
            print('')
