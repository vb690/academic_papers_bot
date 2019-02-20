import tweepy as tpy

class __PosterUtilities():
    '''
    '''
    def _read_twitter_credentials(self, password):
        '''
        '''
        credentials = {}
        with open('\\credentails\\OAuth_credentials.txt'):
            for line in lines:

                field, value = line.split()[0], line.split()[1]
                credentails[field] = value

        return credentials

class TwitterPoster(__PosterUtilities):
    '''
    '''
    def __init__(self):
        '''
        '''
        password = input('Password: ')
        credentials = self._read_twitter_credentials(password)
        auth = tpy.OAuthHandler(credentials['consumer_key']
                                , credentials['consumer_secret']
                                )
        self.auth = auth.set_access_token(credentials['access_key']
                                         , credentials['access_secret']
                                         )

    def __connection(self):
        api_connection = tpy.API(self.auth)
        return api_connection

    def post(self, paper_title):
        api_connection = self.__connection()
        api_connection.update(paper_title)
