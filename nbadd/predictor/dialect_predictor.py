from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides

@Predictor.register('dialect-predictor')
class DialectPredictor(Predictor):
    '''
    predict_json return class_probabilities and the label this is all 
    baseed on decode.
    '''

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"text": "..."}``.
        Returns an instance from the dataset reader
        """
        text = json_dict["text"]
        return self._dataset_reader.text_to_instance(text)