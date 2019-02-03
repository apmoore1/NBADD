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
                 code_switching_regularizer: Optional[float] = 0.0,
                 bivalency_regularizer: Optional[float] = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        '''
        :param dropout: The amount of dropout to apply. Dropout is applied 
                        after each non-linear layer and the word embeddings 
                        lookup. Two types of dropout are applied, variational 
                        dropout is applied if the input is to the dropout is 
                        a sequence of vectors (each vector in the sequence 
                        representing a word), and normal dropout if the input 
                        is a vector.
        :param code_switching_regularizer: The weight associated to the code 
                                           switching lexicon regulisation the 
                                           lower the less affect it has. This 
                                           requires that the dataset reader is 
                                           going to supply the code switching 
                                           arrays for the forward function of 
                                           this class. If set a good values is 
                                           0.001
        :param bivalency_regularizer: The weight associated to the bivalency 
                                      regulisation the lower the less affect it 
                                      has. This requires that the dataset 
                                      reader is going to supply the bivalency
                                      arrays for the forward function of this 
                                      class.
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
        self.code_switching_regularizer = code_switching_regularizer
        self.bivalency_regularizer = bivalency_regularizer
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def reset_parameters(self):
        '''
        Intitalises the attnention vector
        '''
        torch.nn.init.uniform_(self.attention_vector, -0.01, 0.01)

    def lexicon_loss(self, lexicon_array: torch.Tensor, 
                     attention_weights: torch.Tensor,
                     text_mask: torch.Tensor) -> float:
        '''
        :param lexicon_array: Tensor of shape [batch, sequence_length] that 
                              contains either 0, -1, or 1 values where -1 
                              has to be for the whole sample in the batch 
                              which would indicate not to perform attention 
                              regulisation else 0 and 1 indicate important words 
                              in the sequence that should be attended to.
        :param attention_weights: Tensor of shape [batch, sequence_length] 
                                  that contains weights associating to 
                                  the importance of the words in the sequence.
        :param text_mask: Tensor of shape [batch, sequence_length]. Mask 
                          denoting whether there is a word or not at that 
                          position.
        :returns: The difference between the lexicon_array denotating words 
                  that should important to the network and those that the 
                  network think are important via it's attention mechansim. 
                  The difference is computed via cross entropy.
        '''
        # Mask is required to not add loss when the label is MSA
        lexicon_mask = lexicon_array.sum(1)
        lexicon_mask = (lexicon_mask >= 0).float()
        # Calculating the lexicon loss
        lexicon_attention_weights = attention_weights.squeeze()
        lexicon_softmax = util.masked_softmax(lexicon_array, text_mask)
        # This is to stop NANS from performing log on 0.
        lexicon_attention_weights = lexicon_attention_weights + 1e-8
        # Perform cross entropy like in the paper to get a loss function
        # that represents disagreement between the two vectors
        lexicon_loss = lexicon_softmax * torch.log(lexicon_attention_weights)
        lexicon_loss = lexicon_loss * text_mask.float()
        lexicon_loss = lexicon_loss.abs()
        lexicon_loss = lexicon_loss.sum(1)

        lexicon_loss = lexicon_loss * lexicon_mask
        lexicon_loss = lexicon_loss.sum()
        return lexicon_loss

    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                code_switching_array: torch.Tensor = None,
                bivalency_array: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
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

            if code_switching_array is not None and self.training:
                code_switch_loss = self.lexicon_loss(code_switching_array, 
                                                     attention_weights, text_mask)
                loss = loss + (self.code_switching_regularizer * code_switch_loss)
            if bivalency_array is not None and self.training:
                bivalency_loss = self.lexicon_loss(bivalency_array, 
                                                   attention_weights, text_mask)
                loss = loss + (self.bivalency_regularizer * bivalency_loss)
            
            for metric in self.metrics.values():
                metric(logits, label)
            
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
