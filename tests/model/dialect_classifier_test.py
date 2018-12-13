from pathlib import Path

from allennlp.common.testing import ModelTestCase


class DialectClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        test_data_dir = Path(__file__, '..', '..', 'test_data')
        test_data_fp = str(Path(test_data_dir, 'aoc_test_data.json').resolve())
        test_model_fp = str(Path(test_data_dir, 'dialect_classifier.json').resolve())
        self.set_up_model(test_model_fp, test_data_fp)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)