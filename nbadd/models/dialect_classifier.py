from typing import Optional, Dict

from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import numpy as np
import torch
import torch.nn.functional as F


@Model.register("dialect_classifier")
class DialectClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward
        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_text = self.text_field_embedder(text)
        if self.dropout:
            embedded_text = self.dropout(embedded_text)
        text_mask = util.get_text_field_mask(text)
        encoded_text = self.text_encoder(embedded_text, text_mask)
        if self.dropout:
            encoded_text = self.dropout(encoded_text)
        logits = self.classifier_feedforward(encoded_text)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"class_probabilities": class_probabilities}

        if label is not None:
            loss = self.loss(logits, label)
            #for metrics in [self.metrics, self.f1_metrics]:
            #    for metric in metrics.values():
            #        metric(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            #for metric in self.f1_metrics.values():
            #    metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) 
                for metric_name, metric in self.metrics.items()}
    
    def decode(self, output_dict: Dict[str, torch.Tensor]
               ) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict