<p align="center">
  <img width="400" height="100" src="https://github.com/vb690/academic_papers_bot/blob/master/images/font.png">
<p align="center">
  <img width="200" height="200" src="https://github.com/vb690/academic_papers_bot/blob/master/images/hal.png">
<p align="center">
  <i>"Good afternoon, gentlemen.   <br /> 
   I am a ScholHAL 9000 computer" </i>
</p>

## Generating Academic-Sound Papers Titles 

A simple bot for generating and posting paper titles based on original ones obtained from Google Scholar.

### Scrape and Parse
Scrape some topics 
```python
from modules.text.scraper import ScholarScraper
from modules.text.txt_parser import ScholarScraper


scraper = ScholarScraper()

# scraping a topic
for topic in ['phisics']:

    scraper.scrape_topic(
      topic=topic
    )

# scraping authors
for author in ['Albert Bandura', 'Albert Einstein']:

    scraper.scrape_author(
      author_name=author
    )
```
And parse them in a TextObject
```python
from modules.text.txt_parser import ScholarScraper

authors = ['Albert Bandura', 'Albert Einstein']

parser = TextParser(
    topics=[authors],
    sequence_len=12
)
text_object = parser.parse_text(
    text_paths=['loadings\\authors_papers\\'],
    txt_obj_name='bandura_einstein'
)
```
### Create and Train Bot Brain
```python
from modules.bot.brain_creator import BrainCreator

brain = BrainCreator(
    X=text_object.X,
    y=text_object.y
    )

brain.create_recurrent_model(
    recurrent_layers=[150, 150],
    fc_layers=[150, 75],
    dropout_rate=0.0,
    optimizer='adam',
    batch_size=256,
    embedded=50
)
brain.train_model(
    text_object.X,
    text_object.y,
    weights_name='bandura_einstein',
    epochs=250,
    batch_size=256
)
brain.save_model('bandura_einstein')
```
### Assemble and Query Bot
```python
from modules.bot.brain_creator import AssembledBrain
from modules.bot.bot_creator import GeneralBot

brain = AssembledBrain(
  theme='bandura_einstein'
)
bot = GeneralBot(
  brain=brain
)
while not finished:

    bot.thinking(
        total_titles=10,
        temperature=0.6,
        min_len=0
    )
    finished, selected_title = bot.proposing()
```

### Streamlit App
...

### Available Data
Authors
```
* Albert Bandura
* Albert Einstein
* Burrhus Frederic Skinner
* Jaak Panksepp
* Kent Berridge
* Michael Gazzaniga
* Richard Davidson
* Stephen Hawking
* Charles Robert Darwin
```
Topics
```
* Affective Neuroscience
* Anthropology
* Artificial Intelligence
* Behavioural Neuroscience
* Biology
* Chemistry
* Cognitive Neuroscience
* Cognitive Psychology
* Genetics
* Greek Literature
* Physics
* Political Science
* Social Neuroscience
* Theology
* Zoology
```

### Examples
...

### Roadmap
1. Implement a streamlit app.
3. Expand the scraper functionalities integrating [arXiv API](https://arxiv.org/help/api/user-manual).
