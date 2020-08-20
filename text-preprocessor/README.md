# text-preprocessor
An application to preprocess a simple document corpus. This script can tokenize, remove stopwords, lemmatize words and remove part-of-speech (POS) categories from the data.

### Setup
Activate/create the Pipenv environment with: ```pipenv shell```

Then run: ```pipenv install```


### Running
You just need to execute ```python3 src/main.py``` script. The script needs some arguments to be passed on. These are the following:

* ```--datasetFile``` - path of the dataset to preprocess. The dataset must be a Python dictionary array/JSON array
* ```--field``` - field to be preprocessed inside each dataset object.
* ```--lang``` - dataset language.
* ```--lemmatize``` - boolean indicating wheter or not lemmatization should be carried on the dataset.
* ```--desiredPos``` - list os POS (part-of-speech) categories to maintain on preprocessing. These must be [spaCy](https://spacy.io) POS categories.
* ```--stopwordsFile``` (*optional*) - file containing additional stopwords to remove. Completely optional

Below, a command example:

```python3 preprocess.py --datasetFile datasets/reddit_pt_2005_2020_desabafos_brasil.json --field body --lang pt --lemmatize True --desiredPos NOUN VERB```
