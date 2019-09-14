from .bot.brain_creator import AssembledBrain
from .bot.bot_creator import GeneralBot
from .utilities.twitter_utils import _AbstractPoster


class TwitterPoster(_AbstractPoster):
    '''
    '''
    def __init__(self, theme):
        '''
        '''
        brain = AssembledBrain(theme=theme)
        self.bot = GeneralBot(brain=brain)
        password = input('Password: ')
        self.connection = self.__connect(
            self.__authenticates(
                self.__read_credentials(
                    password=password
                )
            )
        )

    def generate(self, total_titles=10, temperature=0.6, min_len=10):
        '''
        '''
        self.bot.thinking(
            total_titles=total_titles,
            temperature=temperature,
            min_len=min_len
        )
        selected, title = self.bot.proposing()

    def post(self, paper_title):
        '''
        '''
        selected = False
        while selected:

            selected, title = self.generate()

        print('The chosen title is ')
        print(title)
        agree = input('Do you want to post it? y n')
        if agree == 'y':
            self.connection.update(paper_title)
