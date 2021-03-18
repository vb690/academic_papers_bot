from modules.bot.brain_creator import BrainCreator
from modules.text.txt_parser import TextParser

brains = {
    'Albert Einstein': ['Albert Einstein'],
    'Jaak Panksepp': ['Jaak Panksepp'],
}

for brain_name, brain_papers in brains.items():

    parser = TextParser(
        topics=[brain_papers],
        sequence_len=12
    )
    text_object = parser.parse_text(
        ['loadings\\authors_papers\\'],
        brain_name
    )

    # CREATE AND TRAIN
    brain = BrainCreator(
        X=text_object.X,
        y=text_object.y
        )

    brain.create_recurrent_model(
        recurrent_layers=[250],
        fc_layers=[150, 75, 25],
        dropout_rate=0.0,
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
