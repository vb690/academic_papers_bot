import tweepy as tpy


class _AbstractPoster:
    '''
    '''
    def _read_credentials(self, password):
        '''
        '''
        credentials = {}
        with open('\\credentails\\OAuth_credentials.txt') as file:
            for line in file:

                field, value = line.split()[0], line.split()[1]
                credentials[field] = value

        return credentials

    def _authenticate(self, credentials):
        auth = tpy.OAuthHandler(
            credentials['consumer_key'],
            credentials['consumer_secret']
        )
        auth.set_access_token(
            credentials['access_key'],
            credentials['access_secret']
        )
        return auth

    def _connect(self, auth):
        '''
        '''
        api_connection = tpy.API(auth)
        try:
            api_connection.verify_credentials()
            print('Authentication OK')
            return api_connection
        except:
            print('Error during authentication')
