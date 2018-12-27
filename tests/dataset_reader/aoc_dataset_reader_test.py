from pathlib import Path

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from nbadd.dataset_reader.aoc_reader import AOCDatasetReader

class TestAOCDatasetReader(AllenNlpTestCase):

    def test_read_from_file(self):
        reader = AOCDatasetReader()
        test_fp = Path(__file__, '..', '..', 'test_data', 'aoc_test_data.json')
        
        instance1 = {"text": ["الف", "مليون", "مبارك", "لفريق", "الوحدات"],
                     "label": "msa"}
        instance2 = {"text": ["ف", "بلاش", "التعميم", "وتشويه", "صوره"], 
                     "label": "egyptian"}
        instance3 = {"text": ["ارامكو", "تقول", "انوو", "7000", "مهوب", "6000"],
                     "label": "gulf"}
        instance4 = {"text": ["اكيد", "الطفل", "ورد", "في", "قلوب"], 
                     "label": "levantine"}
        
        instances = ensure_list(reader.read(str(test_fp.resolve())))

        assert len(instances) == 4

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
        assert [t.text for t in fields["text"].tokens][:5] == instance4["text"]
        assert fields["label"].label == instance4["label"]