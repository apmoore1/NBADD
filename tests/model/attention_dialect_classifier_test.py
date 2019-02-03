from pathlib import Path
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.data.iterators import DataIterator, BasicIterator
from allennlp.training import Trainer
import numpy


class AttentionDialectClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        test_data_dir = Path(__file__, '..', '..', 'test_data')
        test_data_fp = str(Path(test_data_dir, 'aoc_test_data.json').resolve())
        test_model_fp = str(Path(test_data_dir, 'attention_dialect_classifier.json').resolve())
        self.set_up_model(test_model_fp, test_data_fp)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_regularization(self):
        iterator = BasicIterator(batch_size=32)
        trainer = Trainer(self.model,
                          None,  # optimizer,
                          iterator,
                          self.instances)

        # You get a RuntimeError if you call `model.forward` twice on the same inputs.
        # The data and config are such that the whole dataset is one batch.
        training_batch = next(iterator(self.instances, num_epochs=1))
        validation_batch = next(iterator(self.instances, num_epochs=1))

        training_loss = trainer.batch_loss(training_batch, for_training=True).item() / 10
        validation_loss = trainer.batch_loss(validation_batch, for_training=False).item() / 10

        # Training loss should have the regularization penalty, but validation loss should not.
        numpy.testing.assert_almost_equal(training_loss, validation_loss, decimal=0)

class CodeSwitchingAttentionDialectClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        test_data_dir = Path(__file__, '..', '..', 'test_data')
        test_data_fp = str(Path(test_data_dir, 'aoc_test_data.csv').resolve())
        test_model_fp = str(Path(test_data_dir, 'code_switching_attention_dialect_classifier.json').resolve())
        self.set_up_model(test_model_fp, test_data_fp)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_code_switching_loss(self):
        iterator = BasicIterator(batch_size=32)
        trainer = Trainer(self.model,
                          None,  # optimizer,
                          iterator,
                          self.instances)

        # You get a RuntimeError if you call `model.forward` twice on the same inputs.
        # The data and config are such that the whole dataset is one batch.
        training_batch = next(iterator(self.instances, num_epochs=1))
        validation_batch = next(iterator(self.instances, num_epochs=1))

        training_loss = trainer.batch_loss(training_batch, for_training=True).item() / 10
        trainer.model.training = False
        validation_loss = trainer.batch_loss(validation_batch, for_training=False).item() / 10

        # Training loss should not equal validation loss as the training loss 
        # should include the additional attention regulisation penalty the 
        # the different will be arount 0.7
        with pytest.raises(AssertionError):
            numpy.testing.assert_almost_equal(training_loss, validation_loss, decimal=1)