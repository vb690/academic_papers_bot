from modules.bot.brain_creator import BrainCreator, AssembledBrain
from modules.bot.bot_creator import GeneralBot
from modules.text.txt_parser import TextParser

topics = ['albert_bandura', 'jaak_panksepp', 'michael_gazzaniga']

# PARSE
parser = TextParser(
    names=topics,
    sequence_len=12
)
text_object = parser.parse_text(
    'loadings\\authors_papers\\',
    'bandura_panksepp_gazzaniga'
)

# CREATE AND TRAIN
brain = BrainCreator(
    X=text_object.X,
    y=text_object.y
    )
brain.create_recurrent_model(
    recurrent_layers=[250, 250, 250],
    fc_layers=[150, 75, 25],
    dropout_rate=0.0,
    optimizer='adam',
    batch_size=256
)
brain.train_model(
    text_object.X,
    text_object.y,
    weights_name='bandura_panksepp_gazzaniga',
    epochs=150,
    batch_size=250
)
brain.save_model('bandura_panksepp_gazzaniga')

# GENERATE TITLES
brain = AssembledBrain('bandura_panksepp_gazzaniga')
bot = GeneralBot(brain=brain)
bot.thinking(
    total_titles=10,
    temperature=0.6,
    min_len=0
)
bot.proposing()
