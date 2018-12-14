# NBADD (Neural Bivalent Arbaic Dialect Detection)

## Installation
Requires python >= 3.6.1. Install the requirements:
`pip install -r requirements.txt`

## Data and Embeddings

Before anything can be run the data has to be downloaded from the following [Box folder](https://lancaster.app.box.com/folder/55791268702), this can be saved anywhere on your computer. Once you have unzipped `training.zip`, `validation.zip`, and `testing.zip`, the respective folders need processing so that there is only a single file representing `training`, `validation`, and `testing`. To process each of the folders run the following:
`python process_folder.py TRAINING_FOLDER_PATH ./train.json`
`python process_folder.py VALIDATION_FOLDER_PATH ./val.json`
`python process_folder.py TESTING_FOLDER_PATH ./test.json`

The Twitter Embeddings have to be donwloaded from [here](https://drive.google.com/file/d/1hEuNHn2PA7kIf1IK0FUGUskA77YZJ3vO/view) and then processed by running the following:
`python process_twitter_vectors.py LOCATION_OF_THE_DOWNLOADED_TWITTER_VECTORS --verbose`
This will create a new file in the current directory called `TwitterVectors.txt` which contains a WORD and it's vector representation on each new line, like the [Glove Vectors](https://nlp.stanford.edu/projects/glove/). This is done as it fits into the AllenNLP library better, which is the main library used in this project. Once this has run you should 

## Run the code
There is so far 3 different models:
1. Word level Bi-directional LSTM model
2. Character level Bi-directional LSTM model
3. Word and Character level Bi-directional LSTM model

These three models can be found through the three different configuration files: [word](./word_bilstm.json), [character](./char_bilstm.json), and [word and character](./word_char_bilstm.json) respectively.

To run these models use the following command where `word_bilstm.json` represents which of the three models you want to run in this case it is the Word level Bi-directional LSTM model:
`allennlp train word_bilstm.json -s /tmp/anything --include-package nbadd`