from modules.bot.brain_creator import AssembledBrain
from modules.bot.bot_creator import GeneralBot

# GENERATE TITLES
finished = False
brain = AssembledBrain(
    'Einstein Panksepp Bandura Gazzaniga Sagan Berridge Skinner'
)
bot = GeneralBot(brain=brain)
while not finished:

    bot.thinking(
        total_titles=10,
        temperature=1,
        min_len=20
    )
    finished, selected_title = bot.proposing()
    print(finished)
