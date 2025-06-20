import torch
import sys

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Union, Tuple
from transformers import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from sentence_transformers import SentenceTransformer
from config import ANSWER_LENGTH


## Add classification head to SBERT model
class SbertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):

        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        self.sbert = SentenceTransformer(config.sbert_path)
        self.sbert.max_seq_length=ANSWER_LENGTH

        classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.sbert(
            {'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'position_ids':position_ids,
            'head_mask':head_mask,
            'inputs_embeds':inputs_embeds,
            'output_attentions':output_attentions,
            'output_hidden_states':output_hidden_states,
            'return_dict':return_dict}
        )

        pooled_output = self.dropout(outputs['sentence_embedding'])
        logits = self.classifier(pooled_output)

        loss = None
        
        if labels is not None:

            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)

            if self.config.problem_type is None:

                if self.num_labels == 1:

                    self.config.problem_type = "regression"

                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):

                    self.config.problem_type = "single_label_classification"

                else:

                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":

                loss_fct = MSELoss()

                if self.num_labels == 1:

                    loss = loss_fct(logits.squeeze(), labels.squeeze())

                else:

                    loss = loss_fct(logits, labels)

            elif self.config.problem_type == "single_label_classification":

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            elif self.config.problem_type == "multi_label_classification":

                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:

            output = (logits,) + outputs[2:]
            
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs['output_hidden_states'],
            attentions=outputs['output_attentions'],
        )
