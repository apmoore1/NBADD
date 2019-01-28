from pathlib import Path

from allennlp.common.util import ensure_list
import pytest

from nbadd.dataset_reader.aoc_reader_csv import AOCCSVDatasetReader

class TestAOCCSVDatasetReader():
    TEST_DATA_DIR = Path(__file__, '..', '..', 'test_data')
    LEXICON_FOLDER = Path(TEST_DATA_DIR, 'lexicons').resolve()

    def test_read_from_file(self):
        reader = AOCCSVDatasetReader()
        test_fp = Path(self.TEST_DATA_DIR, 'aoc_test_data.csv')
        
        instance1 = {"text": ["بالإضافة","لقيام","معلمو","الجيزة","للذهاب"],
                     "label": "MSA"}
        instance2 = {"text": ["شهادة", "البرادعي", "يا", "سنيورة", "كانت"], 
                     "label": "DIAL_EGY"}
        instance3 = {"text": ["العماله", "طلعت", "مع", "خشوم", "المواطنين", 
                              "من"],
                     "label": "DIAL_GLF"}
        instance4 = {"text": ["لمسه", "اليد", "مرتين", "واضحة", "جدا", 
                              "والحكم"], 
                     "label": "DIAL_LEV"}
        
        instances = ensure_list(reader.read(str(test_fp.resolve())))

        instance_msg = ("It should not read an empty text field instance as "
                        "an instance")
        assert len(instances) == 4, instance_msg

        fields = instances[0].fields
        assert [t.text for t in fields["text"].tokens][:5] == instance1["text"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["text"].tokens][:5] == instance2["text"]
        assert fields["label"].label == instance2["label"]
        fields = instances[2].fields
        assert [t.text for t in fields["text"].tokens] == instance3["text"]
        assert fields["label"].label == instance3["label"]
        fields = instances[3].fields
        assert [t.text for t in fields["text"].tokens] == instance4["text"]
        assert fields["label"].label == instance4["label"]

    def test_lexicon_array(self):
        reader = AOCCSVDatasetReader(code_switching=True, 
                                     lexicon_folder=self.LEXICON_FOLDER)
        test_fp = Path(self.TEST_DATA_DIR, 'aoc_test_data.csv')
        
        instance1 = {"text": ["بالإضافة","لقيام","معلمو","الجيزة","للذهاب"],
                     "label": "MSA", "code_switching_array": [0,0,0,0,0]}
        instance2 = {"text": ["شهادة", "البرادعي", "يا", "سنيورة", "كانت"], 
                     "label": "DIAL_EGY", "code_switching_array": [0,1,0,0,0]}
        instance3 = {"text": ["العماله", "طلعت", "مع", "خشوم", "المواطنين", 
                              "من"],
                     "label": "DIAL_GLF", "code_switching_array": [0,0,0,0,1,0]}
        instance4 = {"text": ["لمسه", "اليد", "مرتين", "واضحة", "جدا", 
                              "والحكم"], 
                     "label": "DIAL_LEV", "code_switching_array": [1,0,0,0,0,1]}

        instances = ensure_list(reader.read(str(test_fp.resolve())))

        instance_msg = ("It should not read an empty text field instance as "
                        "an instance")
        assert len(instances) == 4, instance_msg

        fields = instances[0].fields
        assert [t.text for t in fields["text"].tokens][:5] == instance1["text"]
        assert fields["label"].label == instance1["label"]
        assert list(fields["code_switching_array"].array)[:5] == instance1["code_switching_array"]
        fields = instances[1].fields
        assert [t.text for t in fields["text"].tokens][:5] == instance2["text"]
        assert fields["label"].label == instance2["label"]
        assert list(fields["code_switching_array"].array)[:5] == instance2["code_switching_array"]
        fields = instances[2].fields
        assert [t.text for t in fields["text"].tokens] == instance3["text"]
        assert fields["label"].label == instance3["label"]
        assert list(fields["code_switching_array"].array) == instance3["code_switching_array"]
        fields = instances[3].fields
        assert [t.text for t in fields["text"].tokens] == instance4["text"]
        assert fields["label"].label == instance4["label"]
        assert list(fields["code_switching_array"].array) == instance4["code_switching_array"]