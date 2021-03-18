from modules.bot.brain_creator import AssembledBrain
from modules.bot.bot_creator import GeneralBot

# GENERATE TITLES
finished = False
brain = AssembledBrain('Jaak Panksepp')
bot = GeneralBot(brain=brain)
while not finished:

    bot.thinking(
        total_titles=10,
        temperature=0.6,
        min_len=0
    )
    finished, selected_title = bot.proposing()
