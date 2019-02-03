# NBADD (Neural Bivalent Arbaic Dialect Detection)

## Installation
Requires python >= 3.6.1. Install the requirements:
`pip install -r requirements.txt`

## Data and Embeddings

The data is automatically downloaded when you run the code. The AOC data comes from the following [paper](https://aclanthology.coli.uni-saarland.de/papers/W18-3930/w18-3930) of which the data can be found [here](https://github.com/UBC-NLP/aoc_id/)

The Twitter Embeddings have to be donwloaded from [here](https://drive.google.com/file/d/1hEuNHn2PA7kIf1IK0FUGUskA77YZJ3vO/view) and then processed by running the following:

`python process_twitter_vectors.py LOCATION_OF_THE_DOWNLOADED_TWITTER_VECTORS --verbose`

This will create a new file in the current directory called `TwitterVectors.txt` which contains a WORD and it's vector representation on each new line, like the [Glove Vectors](https://nlp.stanford.edu/projects/glove/). This is done as it fits into the AllenNLP library better, which is the main library used in this project. The Twitter embeddings come form the following [paper](https://aclanthology.coli.uni-saarland.de/papers/L18-1577/l18-1577).

## Run the code
There is so far 3 different models:
1. Word level Bi-directional LSTM model
2. Character level Bi-directional LSTM model
3. Word and Character level Bi-directional LSTM model

These three models can be found through the three different configuration files: [word](./word_bilstm.json), [character](./char_bilstm.json), and [word and character](./word_char_bilstm.json) respectively.

To run these models use the following command where `word_bilstm.json` represents which of the three models you want to run in this case it is the Word level Bi-directional LSTM model:

`allennlp train word_bilstm.json -s /tmp/anything --include-package nbadd`

To get the AOC data that came from the following [paper](https://aclanthology.coli.uni-saarland.de/papers/W18-3930/w18-3930) that has been cleaned so that samples that contain no text are removed run the following command. This will create the `clean_aoc_data` directory within this directory. Within the `clean_aoc_data` directory it will save the train, validation and test data at the following paths `./clean_aoc_data/train.json`, `./clean_aoc_data/val.json` and `./clean_aoc_data/test.json` respectively. These files as they contain a sample per line in json format can be used with the predictor once a model has been trained.

`python get_aoc_data.py ./clean_aoc_data`

The following will run each of the saved word and character attention based models accross the test data:
1. Standard Word Character attention (Standard)
2. Word Character with code switching regulisation (Code Switching)
3. Word Character with bivalency regulisation (Bivalency)
4. Word Character with code switching and bivalency regulisation (Bivalency and Code Switching)

And save the results in the `confusion_matrix_results` folder under the names in the brackets. The results are saved one per line in json format where the predicted label is in the `prediction` field and the true label is in the `label` field. NOTE: To run this script it assumes you have the pre-trained models saved in the models directory within this directory.

`python create_confusion_matrix_results.py`

## Initial Scores
The models have been run on a GPU machine and the scores for each of the three models can be found [here](./model_run_scores.txt).

## [Notebooks](./notebooks)
Here is a list of COLAB notebooks that have been created and what they show (NOTE: These are best viewed with COLAB as COLAB should show the altair graphs):
1. [AOC Dataset Explore](./notebooks/AOC_Dataset.ipynb) -- This shows the distribution of the labels in the AOC dataset that has been used and how we have further processed it to remove samples that have no text.

## Code Switching
Given that we know the language ahead of time and the MSA words that occur only in that language we want the attention to assign more weight to those words.

## Bivalent words
Given that we know there is word over lap between dialects we want to assign less weight to these words in the attention mechanism. 