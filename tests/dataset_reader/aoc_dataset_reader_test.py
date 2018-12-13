from pathlib import Path

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from nbadd.dataset_reader.aoc_reader import AOCDatasetReader

class TestAOCDatasetReader(AllenNlpTestCase):

    def test_read_from_file(self):
        reader = AOCDatasetReader()
        test_fp = Path(__file__, '..', '..', 'test_data', 'aoc_test_data.json')
        
        instance1 = {"text": ["\u0627\u0644\u062f\u0641\u0627\u0639",
                              "\u0627\u0644\u0647\u0644\u0627\u0644\u064a", 
                              "\u064a\u0631\u062a\u0639\u0628", 
                              "\u0639\u0646\u062f\u0645\u0627", 
                              "\u064a\u0631\u0649"],
                     "label": "msa"}
        instance2 = {"text": ["\u0648\u0645\u0646", "\u062d\u0642", 
                              "\u0627\u0644\u0643\u0648\u064a\u062a", 
                              "\u0627\u0644\u0644\u0649", 
                              "\u0639\u0645\u0644\u062a\u0647"], 
                     "label": "egyptian"}
        instance3 = {"text": ["\u0627\u0631\u0627\u0645\u0643\u0648", 
                              "\u062a\u0642\u0648\u0644", 
                              "\u0627\u0646\u0648\u0648", "7000", 
                              "\u0645\u0647\u0648\u0628", "6000"],
                     "label": "gulf"}
        instance4 = {"text": ["\u0633\u0628\u062d\u0627\u0646", 
                              "\u0627\u0644\u0644\u0647", 
                              "\u0645\u0627\u0641\u064a\u0647", 
                              "\u0627\u062d\u0644\u0649", 
                              "\u0648\u0644\u0627\u0627\u0631\u0648\u0639"], 
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