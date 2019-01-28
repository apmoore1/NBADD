from typing import Optional, Dict

from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules import InputVariationalDropout, TimeDistributed
from allennlp.modules.attention import DotProductAttention
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch.nn.parameter import Parameter


@Model.register("attention_dialect_classifier")
class AttentionDialectClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 classifier_feedforward: Optional[FeedForward] = None,
                 dropout: Optional[float] = 0.0,
                 lexicon_regularizer: Optional[float] = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        '''
        :param lexicon_regularizer: The weight associated to the code switching 
                                    lexicon regulisation the lower the less 
                                    affect it has. This requires that the 
                                    dataset reader is going to supply the code 
                                    switching arrays for the forward function 
                                    of this class. If set a good values is 0.05
        '''
        super().__init__(vocab, regularizer)
        self._naive_dropout = Dropout(dropout)
        self._variational_dropout = InputVariationalDropout(dropout)
        
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        text_encoder_dim = text_encoder.get_output_dim()
        # Attention parameters
        self.project_encoded_text = TimeDistributed(Linear(text_encoder_dim, 
                                                           text_encoder_dim))
        self.attention_vector = Parameter(torch.Tensor(text_encoder_dim))
        self.reset_parameters()
        self.attention_layer = DotProductAttention(normalize=True)

        self.classifier_feedforward = classifier_feedforward
        output_dim = text_encoder_dim
        if classifier_feedforward:
            output_dim = classifier_feedforward.get_output_dim()
        self.label_projection = Linear(output_dim, self.num_classes)
        self.metrics = {
                "accuracy": CategoricalAccuracy()
        }
        self.lexicon_regularizer = lexicon_regularizer
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def reset_parameters(self):
        '''
        Intitalises the attnention vector
        '''
        torch.nn.init.uniform_(self.attention_vector, -0.01, 0.01)

    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                code_switching_array: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Embed text
        embedded_text = self.text_field_embedder(text)
        embedded_text = self._variational_dropout(embedded_text)
        text_mask = util.get_text_field_mask(text)
        # Encode text
        encoded_text = self.text_encoder(embedded_text, text_mask)
        encoded_text = self._variational_dropout(encoded_text)
        #
        # ATTENTION
        #
        # Project each hidden timestep into a new hidden space 
        # before attention
        projected_encoding = self.project_encoded_text(encoded_text)
        projected_encoding = torch.tanh(projected_encoding)
        projected_encoding = self._variational_dropout(projected_encoding)
        # Apply attention over the project encoded text
        batch_size = text_mask.shape[0]
        attention_vector = self.attention_vector.unsqueeze(0).expand(batch_size, -1)
        attention_weights = self.attention_layer(attention_vector, 
                                                 projected_encoding, text_mask)
        attention_weights = attention_weights.unsqueeze(-1)
        weighted_encoded_text = attention_weights * encoded_text
        weighted_encoded_vec = weighted_encoded_text.sum(1)
        weighted_encoded_vec = self._naive_dropout(weighted_encoded_vec)

        if self.classifier_feedforward:
            weighted_encoded_vec = self.classifier_feedforward(weighted_encoded_vec)
        logits = self.label_projection(weighted_encoded_vec)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"class_probabilities": class_probabilities}

        if label is not None:

            loss = self.loss(logits, label)

            if code_switching_array is not None:
                # Mask is required to not add loss when the label is MSA
                code_switching_mask = code_switching_array.sum(1)
                code_switching_mask = (code_switching_mask >= 0).float()
                # Calculating the lexicon loss
                code_attention_weights = attention_weights.squeeze()
                code_switching_softmax = util.masked_softmax(code_switching_array, 
                                                             text_mask)
                # Perform cross entropy like in the paper to get a loss function
                # that represents disagreement between the two vectors
                # This is to stop NANS from performing log on 0.
                code_attention_weights = code_attention_weights + 1e-8
                lexicon_loss = code_switching_softmax * torch.log(code_attention_weights)
                lexicon_loss = lexicon_loss * text_mask.float()
                lexicon_loss = lexicon_loss.abs()
                lexicon_loss = lexicon_loss.sum(1)
                # do not add the MSA label data into the loss hence the 
                # code switching mask
                lexicon_loss = lexicon_loss * code_switching_mask
                lexicon_loss = lexicon_loss.sum()
                loss = loss + (self.lexicon_regularizer * lexicon_loss)
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
