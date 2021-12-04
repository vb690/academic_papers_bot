from modules.bot.brain_creator import BrainCreator
from modules.text.txt_parser import TextParser

brains = {
    'Einstein Panksepp Bandura Gazzaniga Sagan Berridge Skinner': [
        'Albert Einstein',
        'Jaak Panksepp',
        'Albert Bandura',
        'Michael Gazzaniga',
        'Carl Sagan',
        'Kent Berridge',
        'Burrhus Frederic Skinner'
    ]
}

for brain_name, brain_papers in brains.items():

    parser = TextParser(
        topics=[brain_papers],
        sequence_len=10
    )
    text_object = parser.parse_text(
        ['loadings\\authors_papers\\'],
        brain_name,
        char=True
    )

    # CREATE AND TRAIN
    brain = BrainCreator(
        X=text_object.X,
        y=text_object.y
        )

    brain.create_recurrent_model(
        recurrent_layers=[150],
        fc_layers=[75, 25],
        dropout_rate=0.1,
        optimizer='adam',
        batch_size=256
    )
    brain.train_model(
        text_object.X,
        text_object.y,
        weights_name=brain_name,
        epochs=500,
        batch_size=256
    )
    brain.save_model(brain_name)
