import csv
import logging
from pathlib import Path
from typing import Dict, Union, List, Set, Callable

from allennlp.data.fields import LabelField, TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.token import Token
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
                 code_switching_lex_folder: Path = None,
                 bivalency_lex_folder: Path = None):
        '''
        :param tokenizer: Defaults to just Whitespace if no other Tokeniser 
                          is given.
        :param token_indexers: Default to just using word tokens to represent 
                               the input.
        :param code_switching_lex_folder: Folder that contains three lexicon 
                                          lists of code switching words between 
                                          different dialects and MSA: 
                                          1. MSA_DIAL_EGY.txt, 
                                          2. MSA_DIAL_GLF.txt, 
                                          3. MSA_DIAL_LEV.txt. These lexicons 
                                          will allow code switching regularised 
                                          attention.
        :param bivalency_lex_folder: Folder that contains three lexicon lists 
                                     of bivalency words between Egyptian, 
                                     Levantine and Gulf dialects of Arabic:
                                     1. EGY_GLF.txt, 2. EGY_LEV.txt, 
                                     3. GLF_LEV.txt. This will allow bivalency 
                                     regularised attention
        
        NOTE: That all code switching and bivalency words are lower cased and 
        then compared to the words within the text, where when compared the 
        words within the text are temporarly lower cased for comparison reason 
        only. The words within the text do not remain lower cased unless you 
        have specified this within the `token_indexers`.
        '''
        super().__init__(lazy)
        
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}

        self.code_switching_lex_folder = code_switching_lex_folder
        if code_switching_lex_folder is not None:
            self.msa_egy = self._lexicon_set(Path(code_switching_lex_folder, 
                                                  'MSA_DIAL_EGY.txt'))
            self.msa_glf = self._lexicon_set(Path(code_switching_lex_folder, 
                                                  'MSA_DIAL_GLF.txt'))
            self.msa_lev = self._lexicon_set(Path(code_switching_lex_folder, 
                                                  'MSA_DIAL_LEV.txt'))
        self.bivalency_lex_folder = bivalency_lex_folder
        if bivalency_lex_folder is not None:
            egy_glf = self._lexicon_set(Path(bivalency_lex_folder, 'EGY_GLF.txt'))
            egy_lev = self._lexicon_set(Path(bivalency_lex_folder, 'EGY_LEV.txt'))
            glf_lev = self._lexicon_set(Path(bivalency_lex_folder, 'GLF_LEV.txt'))
            self.bivalency_egy = egy_glf.union(egy_lev)
            self.bivalency_lev = glf_lev.union(egy_lev)
            self.bivalency_glf = egy_glf.union(glf_lev)

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
                    lexicon_words.add(word.lower())
        return lexicon_words

    def _get_code_switching_lexicon(self, dialect: str) -> Set[str]:
        '''
        :param dialect: The string of the dialect.
        :returns: The code switching MSA words for the given dialect.
        '''
        dialect_lexicon_mapper = {'DIAL_EGY': self.msa_egy, 
                                  'DIAL_GLF': self.msa_glf, 
                                  'DIAL_LEV': self.msa_lev}
        return dialect_lexicon_mapper[dialect]

    def _get_bivalency_lexicon(self, dialect: str) -> Set[str]:
        '''
        :param dialect: The string of the dialect.
        :returns: The bivalency words between the given dialect and all other 
                  dialects.
        '''
        dialect_lexicon_mapper = {'DIAL_EGY': self.bivalency_egy, 
                                  'DIAL_GLF': self.bivalency_glf, 
                                  'DIAL_LEV': self.bivalency_lev}
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

    def get_lexicon_regularized_array(self, dialect: str, 
                                      tokenized_text: List[Token],
                                      lexicon_callable: Callable[[str], Set[str]]
                                      ) -> ArrayField:
        if dialect == 'MSA':
            lexicon_regularized_array = [-1 for word in tokenized_text]
        else:
            lexicon = lexicon_callable(dialect)
            lexicon_regularized_array = []
            for word in tokenized_text:
                if word.text.lower() in lexicon:
                    lexicon_regularized_array.append(1)
                else:
                    lexicon_regularized_array.append(0)
        lexicon_regularized_array = np.array(lexicon_regularized_array)
        return ArrayField(lexicon_regularized_array)
    
    def text_to_instance(self, text: str, dialect: str = None) -> Instance:
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field}
        if dialect is not None:
            fields['label'] = LabelField(dialect)
        if dialect is not None and self.code_switching_lex_folder is not None:
            code_switching_array = self.get_lexicon_regularized_array(dialect, 
                                                                      tokenized_text, 
                                                                      self._get_code_switching_lexicon)
            fields['code_switching_array'] = code_switching_array
        if dialect is not None and self.bivalency_lex_folder is not None:
            bivalency_array = self.get_lexicon_regularized_array(dialect, 
                                                                 tokenized_text, 
                                                                 self._get_bivalency_lexicon)
            fields['bivalency_array'] = bivalency_array
        return Instance(fields)