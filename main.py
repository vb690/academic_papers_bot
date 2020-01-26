from modules.bot.brain_creator import BrainCreator, AssembledBrain
from modules.bot.bot_creator import GeneralBot
from modules.text.txt_parser import TextParser

authors = ['albert_bandura',
           'jaak_panksepp',
           'michael_gazzaniga'
           ]

topics = ['anthropology',
          'artificial_intelligence',
          'theology'
          ]

# PARSE
parser = TextParser(
    topics=[authors, topics],
    sequence_len=12
)
text_object = parser.parse_text(
    ['loadings\\authors_papers\\', 'loadings\\topic_papers\\'],
    'weird_mix'
)

# CREATE AND TRAIN
brain = BrainCreator(
    X=text_object.X,
    y=text_object.y
    )

brain.create_recurrent_model(
    recurrent_layers=[250, 250, 250, 250],
    fc_layers=[150, 75, 25],
    dropout_rate=0.0,
    optimizer='adam',
    batch_size=256
)
brain.train_model(
    text_object.X,
    text_object.y,
    weights_name='weird_mix',
    epochs=500,
    batch_size=256
)
brain.save_model('weird_mix')

# GENERATE TITLES
finished = False
brain = AssembledBrain('weird_mix')
bot = GeneralBot(brain=brain)
while not finished:

    bot.thinking(
        total_titles=10,
        temperature=0.6,
        min_len=0
    )
    finished, selected_title = bot.proposing()
