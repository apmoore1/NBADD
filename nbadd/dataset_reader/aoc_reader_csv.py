import csv
import logging
from pathlib import Path
from typing import Dict, Union, List, Set

from allennlp.data.fields import LabelField, TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.file_utils import cached_path
import numpy as np
from overrides import overrides

logger = logging.getLogger(__name__)

@DatasetReader.register("aoc_csv_dataset")
class AOCCSVDatasetReader(DatasetReader):
    def __init__(self, lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 code_switching: bool = False,
                 lexicon_folder: Path = None):
        '''
        :param tokenizer: Defaults to just Whitespace if no other Tokeniser 
                          is given.
        :param token_indexers: Default to just using word tokens to represent 
                               the input.
        :param lexicon_folder: Folder that contains three lexicon lists: 
                               1. MSA_DIAL_EGY.txt, 2. MSA_DIAL_GLF.txt, 
                               3. MSA_DIAL_LEV.txt. These lexicons will allow 
                               code switching regularised attention.'''
        super().__init__(lazy)
        
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}

        self.code_switching = code_switching
        if code_switching and lexicon_folder is None:
            raise ValueError('Cannot perform code switching regularised '
                             'attention without the lexicon folder.')
        if lexicon_folder:
            self.msa_egy = self._lexicon_set(Path(lexicon_folder, 'MSA_DIAL_EGY.txt'))
            self.msa_glf = self._lexicon_set(Path(lexicon_folder, 'MSA_DIAL_GLF.txt'))
            self.msa_lev = self._lexicon_set(Path(lexicon_folder, 'MSA_DIAL_LEV.txt'))

    def _lexicon_set(self, lexicon_fp: Path) -> Set[str]:
        '''
        Given a lexicon file path it will return all of the lexicon words as 
        a Set of strings.

        :param lexicon_fp: File path to the lexicon
        :returns: A set of all lexicon words from the lexicon file path.
        '''
        lexicon_words = set()
        with lexicon_fp.open('r', encoding='utf-8') as lexicon_file:
            for line in lexicon_file: 
                word = line.strip()
                if word:
                    lexicon_words.add(word)
        return lexicon_words

    def _get_code_switching_lexicon(self, dialect: str) -> Set[str]:
        dialect_lexicon_mapper = {'DIAL_EGY': self.msa_egy, 
                                  'DIAL_GLF': self.msa_glf, 
                                  'DIAL_LEV': self.msa_lev}
        return dialect_lexicon_mapper[dialect]

    def _read(self, file_path: Union[str, Path]):
        with open(cached_path(file_path), "r", 
                  newline='', encoding='utf-8') as csv_data_file:
            dialect_data = csv.reader(csv_data_file)
            # Skip the header row
            next(dialect_data)
            for data in dialect_data:
                text = data[2].strip()
                if text == '':
                    continue
                dialect = data[1]
                yield self.text_to_instance(text, dialect)
    
    def text_to_instance(self, text: str, dialect: str = None) -> Instance:
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field}
        if dialect is not None:
            fields['label'] = LabelField(dialect)
        if dialect is not None and self.code_switching:
            if dialect == 'MSA':
                code_switching_lexicon = []
            else:
                code_switching_lexicon = self._get_code_switching_lexicon(dialect)
            code_switching_array = []
            for word in tokenized_text:
                if word.text in code_switching_lexicon:
                    code_switching_array.append(1)
                else:
                    code_switching_array.append(0)
            code_switching_array = np.array(code_switching_array)
            fields['code_switching_array'] = ArrayField(code_switching_array)
        return Instance(fields)