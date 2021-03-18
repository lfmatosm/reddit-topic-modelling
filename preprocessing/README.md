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


python text_preprocessor/preprocess.py --datasetFile datasets/original/brasil_desabafos_2008_2021/reddit-posts-gatherer-pt.submissions.json --datasetName TEST_lemmatized_nouns_only --datasetFolder . --field body --lang pt --lemmatize True --removeStopwords True --removePos True

python text_preprocessor/preprocess.py --datasetFile datasets/original/depression_2009_2015/reddit-posts-gatherer-en.submissions_[until_2015-01-01_dataset].json --datasetName lemmatized_nouns_only_en --datasetFolder TEST_lemmatized_nouns_only --field body --lang en --lemmatize True --removeStopwords True --removePos True