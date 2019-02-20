
from text_module.scraper import *

scraper = ScholarScraper()

for topic in ['phisics']:

    scraper.scrape_topic(topic=topic)

for author in ['Albert Bandura']:

    scraper.scrape_author(author_name=author)
