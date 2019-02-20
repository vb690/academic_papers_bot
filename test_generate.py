import pickle

from bot_module.bot_creator import *
from bot_module.brain_creator import *


with open('loadings\\text_objects\\test.pkl', 'rb') as input_file:
    text_object = pickle.load(input_file)

brain = BrainCreator(text_object=text_object)
brain.create_recurrent_model(hidden_units=[256, 256, 256]
                             , dropout=0.0
                             , optimizer='rmsprop'
                            )
brain.load_weights('test_model')
general_bot = GeneralBot(brain=brain)
#general_bot = GeneralBot(brain=brain)
general_bot.bot_thinking(total_titles=20, temperature=0.6, min_len=50)
general_bot.bot_speaking()
