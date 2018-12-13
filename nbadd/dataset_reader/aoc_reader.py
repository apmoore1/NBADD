import json
import logging
from pathlib import Path
from typing import Dict, Union, List

from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from overrides import overrides

logger = logging.getLogger(__name__)

@DatasetReader.register("aoc_dataset")
class AOCDatasetReader(DatasetReader):
    def __init__(self, lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        '''
        :param tokenizer: Defaults to just Whitespace if no other Tokeniser 
                          is given.
        :param token_indexers: Default to just using word tokens to represent 
                               the input.
        '''
        super().__init__(lazy)
        
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: Union[str, Path]):
        with open(file_path, "r") as data_file:
            dialect_data: List[Dict[str, str]] = json.load(data_file)
            for data in dialect_data:
                text = data['text']
                dialect = data['label']
                yield self.text_to_instance(text, dialect)
    
    def text_to_instance(self, text: str, dialect: str = None) -> Instance:
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field}
        if dialect is not None:
            fields['label'] = LabelField(dialect)
        return Instance(fields)