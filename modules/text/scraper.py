from scholarly import scholarly as sc


class ScholarScraper:
    def __init__(self, hard_limit=2000):
        """Initialize a scraper object.

        Arguments:
            - results_path: a string specifying the path where to save file
              containing the papers titles
            - hard_limit: an integer specifying the maximum ammount of titles
              to scrape, Google Scholar however poses a limit of 900 hits
              every =~ 24h
        """
        self.hard_limit = hard_limit

    def scrape_topic(self, topic, min_len=0, max_len=9999):
        """Method for scraping scientific papers' titles from Google Scholar on
        a specific topic

        Arguments:
            - topic: a string specifying the topic to scrape, it defines also
              the file name
            - min_len: is an integer specifying the minimum length of the title
            - max_len: is an integer specifying the maximum length of the title

        Returns:
            - None
        """
        search = sc.search_keyword(topic)
        keyword = next(search).fill()
        with open(
            'loadings\\topic_papers\\{}.txt'.format(topic),
            'a',
            encoding='utf-8'
        ) as file:
            for counter, pubblication in enumerate(keyword.publications):

                if len(pubblication.bib['title']) < min_len \
                 or len(pubblication.bib['title']) > max_len:
                    continue
                file.write(pubblication.bib['title'])
                file.write('\n')
                counter += 1
                if counter > self.hard_limit:
                    break

    def scrape_author(self, author_name, min_len=0, max_len=9999):
        """Method for scraping scientific papers' titles from Google Scholar
        from a specific author

        Arguments:
            - author_name: a string specifying the name of the author
            - min_len: is an integer specifying the minimum length of the title
            - max_len: is an integer specifying the maximum length of the title

        Returns:
            - None
        """
        search = sc.search_author(author_name)
        author = next(search)
        sc.fill(author, sections=['publications'])
        print(author.keys())
        with open(
            'loadings\\authors_papers\\{}.txt'.format(author_name),
            'w',
            encoding='utf-8'
        ) as file:
            for counter, pubblication in enumerate(author['publications']):

                if len(pubblication['bib']['title']) < min_len \
                 or len(pubblication['bib']['title']) > max_len:
                    continue
                file.write(pubblication['bib']['title'])
                file.write('\n')
                counter += 1
                if counter > self.hard_limit:
                    break


if __name__ == '__main__':
    scraper = ScholarScraper()
    # test scraping authors
    for author in ['Stephen Hawking', 'Carl Sagan',  'Charles Robert Darwin']:

        scraper.scrape_author(author_name=author)
