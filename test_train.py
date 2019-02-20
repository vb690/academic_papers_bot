from text_module.txt_parser import *
from bot_module.brain_creator import *
from bot_module.bot_creator import *

from keras.optimizers import Adam

topics = ['dermot_b_h', 'albert_bandura', 'b_f_skinner', 'michael_gazzaniga', 'jaak_panksepp', 'albert_einstein']

text_object = TextParser( names=topics
                         , sequence_len = 6
                         ).parse_text('loadings\\authors_papers\\', 'test')

brain = BrainCreator( text_object=text_object
                    )

#brain.load_model('test_model')


brain.create_recurrent_model(hidden_units=[256, 256, 256]
                             , dropout=0.0
                             , optimizer=Adam()
                            )

#brain.load_weights('test_model')

brain.train_model(weights_name='test_model'
                  , epochs=2500
                  , batch_size=250
                 )
