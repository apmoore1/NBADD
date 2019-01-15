import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path('..').resolve()))
import tempfile
import logging 

from allennlp.commands.train import train_model_from_file
from allennlp.common import Params
from allennlp.data import DataIterator, DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import Model, load_archive

import nbadd

logging.basicConfig(format='%(message)s',
                    level=logging.INFO)


model_fp = str(Path('.', 'training_config', 'attention.json').resolve())
params = Params.from_file(model_fp)
data_fp = str(Path('.', 'tests', 'test_data', 'aoc_test_data.csv').resolve())
reader = DatasetReader.from_params(params['dataset_reader'])
instances = list(reader.read(data_fp))
if 'vocabulary' in params:
    vocab_params = params['vocabulary']
    vocab = Vocabulary.from_params(params=vocab_params, instances=instances)
else:
    vocab = Vocabulary.from_instances(instances)
model = Model.from_params(vocab=vocab, params=params['model'])

with tempfile.TemporaryDirectory() as tmpdirname:
    model = train_model_from_file(model_fp, tmpdirname,overrides="")

print('anything')