import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np
import torch.nn.functional as F


class BertMulti(nn.Module):
  """span prediction and question type classification"""

  def __init__(self, base_model, n_class=4, base_model_output_size=768):
    super(BertMulti, self).__init__()
    self.base_model = base_model
    self.qa_outputs = nn.Linear(768, 2)
    self.fc = nn.Linear(base_model_output_size, n_class)

    self.num_labels = n_class
    self.dropout = nn.Dropout(0.3)
    self.classifier = nn.Linear(768, n_class)

    #RNN
    self.hidden_dim = 256
    self.n_layers = 2

    # RNN Layer
    self.rnn = nn.RNN(base_model_output_size,
                      self.hidden_dim,
                      self.n_layers,
                      bidirectional=True)
    # Fully connected layer
    self.fc1 = nn.Linear(self.n_layers * 2 * self.hidden_dim,
                         self.hidden_dim,
                         bias=True)
    self.fc2 = nn.Linear(self.hidden_dim, self.num_labels, bias=True)

    # Dropout layers
    self.dropout_train = nn.Dropout(0.3)
    self.dropout_test = nn.Dropout(0.0)

  def forward(self, inputs, do_train=False):

    total_loss = None

    ##############################
    # Pretrained transformer
    ##############################
    base_inputs = {
        'input_ids': inputs["input_ids"],
        'attention_mask': inputs["attention_mask"],
        'token_type_ids': inputs["token_type_ids"],
    }
    outputs = self.base_model(**base_inputs)

    sequence_output = outputs[0]

    ##############################
    # Span prediction
    ##############################
    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    span_loss = None
    start_positions = inputs.get("start_positions", None)
    end_positions = inputs.get("end_positions", None)

    if start_positions is not None and end_positions is not None:
      # If we are on multi-GPU, split add a dimension
      if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
      if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
      # sometimes the start/end positions are outside our model inputs, we ignore these terms
      ignored_index = start_logits.size(1)
      start_positions.clamp_(0, ignored_index)
      end_positions.clamp_(0, ignored_index)

      loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
      start_loss = loss_fct(start_logits, start_positions)
      end_loss = loss_fct(end_logits, end_positions)
      span_loss = (start_loss + end_loss) / 2

    span_output = (start_logits, end_logits)

    ##############################
    # BERT seq classification head
    ##############################
    # pooled_output = outputs[1]

    # pooled_output = self.dropout(pooled_output)
    # logits = self.classifier(pooled_output)

    ##############################
    # FCN
    ##############################
    # out = torch.max(sequence_output, dim=1)[0]
    # logits = self.classifier(out)

    ##############################
    # RNN
    ##############################
    # input shape: (sentence_length, batch_size, emb_dim)
    rnn_input = sequence_output.permute(1, 0, 2)

    # hidden layer
    batch_size = sequence_output.size(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_0 = torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(device)

    # output shape: (sentence_length, batch_size, 2 * hidden_size)
    _, h_n = self.rnn(rnn_input, h_0)

    # h_n: [batch_size, 4 * self.hidden_size]
    h_n = h_n.permute(1, 0, 2)
    h_n = h_n.contiguous().view(h_n.size(0), -1)

    # FC layers
    output = F.relu(self.fc1(h_n))
    output = self.dropout_train(output) if do_train else self.dropout_test(
        output)
    logits = F.log_softmax(self.fc2(output), dim=1)

    ##############################
    # Loss
    ##############################

    labels = inputs.get("labels", None)

    q_loss = None

    if labels is not None:
      loss_fct = CrossEntropyLoss()
      q_loss = loss_fct(logits, labels)

    # calculate loss
    if span_loss and q_loss:
      total_loss = (span_loss + q_loss) / 2

    q_output = (logits,)
    output = span_output + q_output + outputs[2:]

    return ((total_loss,) + output) if total_loss is not None else output
