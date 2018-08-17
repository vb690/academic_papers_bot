import random
import numpy as np

from txt_parser import *
from model_creator import *

class GeneralBot():
    def __init__(self, brain):
        self.titles = []
        self.model = brain.model
        self.text_object = brain.text_object

    @staticmethod
    def temperated_prediction(predictions, temperature=1.0):
        '''
        Function for sampling a prediction from temperated output of the softmax function
        Higher temperature = more variability, more randomness, more errors
        Lower temperature  = less variability, more repetition, less errors

        Arguments:

        predictions: is an iterable of probability over the available characters provided by the softmax function
        temperature: is a float controlling the

        Returns:

        The argmax of the temperated softmax output
        '''
        predictions = np.asarray(predictions[0]).astype('float64')
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        p_preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, p_preds)
        return np.argmax(probas)

    @staticmethod
    def random_seed_sampling(starting_seeds):
        '''
        '''
        index = np.random.randint(len(starting_seeds))
        return starting_seeds[index]

    def bot_thinking(self, total_titles, temperature):
        '''
        '''
        counter_titles = 0
        starting_seeds = self.text_object.starting_seeds
        while counter_titles != total_titles:

            predicted_character = None
            seed = self.random_seed_sampling(starting_seeds)
            title = [self.text_object.int_char[character] for character in seed]
            while predicted_character != '\n':

                X = np.reshape(seed, (1, len(seed), 1))
                predicted_character = self.model.predict(X, verbose=0)
                predicted_character = self.temperated_prediction(predicted_character, temperature)
                seed.append(predicted_character)
                title.append(self.text_object.int_char[predicted_character])
                seed = seed[1:len(seed)]
                if len(title) > 140:
                    break

            if len(title) > 140 or len(title) < 15:
                continue

            self.titles.append(''.join(title))
            counter_titles +=1

    def bot_speaking(self):
        '''
        '''
        for paper in self.titles:

            print('Hey! This seems cool, check it out!')
            print(paper)
            print('')
