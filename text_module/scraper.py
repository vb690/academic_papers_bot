import scholarly as sc

class ScholarScraper():
    def __init__(self, results_path, hard_limit = 2000):
        '''
        Initialize a scraper object.

        Arguments:
            - results_path: a string specifying the path where to save file containing the papers titles
            - hard_limit: an integer specifying the maximum ammount of titles to scrape, Google Scholar however poses a limit of 900 hits every =~ 24h
        '''
        self.results_path = results_path
        self.hard_limit = hard_limit

    def scrape_topic (self, topic, min_len):
        '''
        Method for scraping scientific papers' titles from Google Scholar on a specific topic

        Arguments:
            - topic: a string specifying the topic to scrape, it defines also the file name

        Returns:
            - None
        '''
        searches = sc.search_pubs_query(topic)
        counter = 0
        with open('{}{}.txt'.format(self.results_path, topic), 'a', encoding='utf-8') as file:
            for search in searches:
                
                if len(search.bib['title']) < min_len:
                    continue
                file.write(search.bib['title'])
                file.write('\n')
                counter += 1
                if counter > self.hard_limit:
                    break

    def scrape_author (self, author_name, min_len):
        '''
        Method for scraping scientific papers' titles from Google Scholar from a specific author

        Arguments:
            - author_name: a string specifying the name of the author

        Returns:
            - None
        '''
        search = sc.search_author(author_name)
        author = next(search).fill()
        counter = 0
        with open('{}{}.txt'.format(self.results_path, author_name), 'a', encoding='utf-8') as file:
            for pubblication in author.publications:

                if len(pubblication.bib['title']) < min_len:
                    continue
                file.write(pubblication.bib['title'])
                file.write('\n')
                counter += 1
                if counter > self.hard_limit:
                    break
