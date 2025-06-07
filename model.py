import torch
import torch.nn as nn
import numpy as np
from torchcrf import CRF
from transformers import BertModel, BertConfig

class ModelOutput:
  def __init__(self, logits, labels, loss=None):
    self.logits = logits
    self.labels = labels
    self.loss = loss

class BertBiLSTMNer(nn.Module):
  def __init__(self, args):
    super(BertBiLSTMNer, self).__init__()
    self.bert = BertModel.from_pretrained(args.bert_dir)
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    hidden_size = self.bert_config.hidden_size
    self.lstm_hiden = 128
    self.max_seq_len = args.max_seq_len
    self.bilstm = nn.LSTM(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
               dropout=0.1)
    self.linear = nn.Linear(self.lstm_hiden * 2, args.num_labels)
    self.crf = CRF(args.num_labels, batch_first=True)

  def forward(self, input_ids, attention_mask, labels=None):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    seq_out = bert_output[0]  # [batchsize, max_len, 768]
    batch_size = seq_out.size(0)
    seq_out, _ = self.bilstm(seq_out)
    seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
    seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
    seq_out = self.linear(seq_out)
    logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    loss = None
    if labels is not None:
      labels=torch.tensor(labels,dtype=torch.int64)
      loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    model_output = ModelOutput(logits, labels, loss)
    return model_output


class BertBiGRUNer(nn.Module):
  def __init__(self, args):
    super(BertBiGRUNer, self).__init__()
    self.bert = BertModel.from_pretrained(args.bert_dir)
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    hidden_size = self.bert_config.hidden_size
    self.lstm_hiden = 128
    self.max_seq_len = args.max_seq_len
    self.bilstm = nn.LSTM(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
               dropout=0.1)
    self.linear = nn.Linear(self.lstm_hiden * 2, args.num_labels)
    self.crf = CRF(args.num_labels, batch_first=True)

  def forward(self, input_ids, attention_mask, labels=None):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    seq_out = bert_output[0]  # [batchsize, max_len, 768]
    batch_size = seq_out.size(0)
    seq_out, _ = self.bilstm(seq_out)
    seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
    seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
    seq_out = self.linear(seq_out)
    logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    loss = None
    if labels is not None:
      labels=torch.tensor(labels,dtype=torch.int64)
      loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    model_output = ModelOutput(logits, labels, loss)
    return model_output



class BiGRUNer(nn.Module):
  def __init__(self, args):
    super(BiGRUNer, self).__init__()
    self.bert = BertModel.from_pretrained(args.bert_dir)
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    hidden_size = self.bert_config.hidden_size # 768
    vocab_size = self.bert_config.vocab_size # 21128
    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.lstm_hiden = 128
    self.max_seq_len = args.max_seq_len
    self.bilstm = nn.GRU(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
               dropout=0.1)
    self.linear = nn.Linear(self.lstm_hiden * 2, args.num_labels)
    self.crf = CRF(args.num_labels, batch_first=True)

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.embedding(input_ids) # [batchsize, max_len, 768]
    batch_size = output.size(0)
    seq_out, _ = self.bilstm(output)
    seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
    seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1) # [12, 512, 256]
    seq_out = self.linear(seq_out)
    logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    loss = None
    if labels is not None:
      labels=torch.tensor(labels,dtype=torch.int64)
      loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    model_output = ModelOutput(logits, labels, loss)
    return model_output



class BiLSTMNer(nn.Module):
  def __init__(self, args):
    super(BiLSTMNer, self).__init__()
    self.bert = BertModel.from_pretrained(args.bert_dir)
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    hidden_size = self.bert_config.hidden_size # 768
    vocab_size = self.bert_config.vocab_size # 21128
    self.embedding = nn.Embedding(vocab_size, hidden_size)
    self.lstm_hiden = 128
    self.max_seq_len = args.max_seq_len
    self.bilstm = nn.GRU(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
               dropout=0.1)
    self.linear = nn.Linear(self.lstm_hiden * 2, args.num_labels)
    self.crf = CRF(args.num_labels, batch_first=True)

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.embedding(input_ids) # [batchsize, max_len, 768]
    batch_size = output.size(0)
    seq_out, _ = self.bilstm(output)
    seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
    seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1) # [12, 512, 256]
    seq_out = self.linear(seq_out)
    logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    loss = None
    if labels is not None:
      labels=torch.tensor(labels,dtype=torch.int64)
      loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    model_output = ModelOutput(logits, labels, loss)
    return model_output
  

class BertCrfNer(nn.Module):
  def __init__(self, args):
    super(BertCrfNer, self).__init__()
    self.bert = BertModel.from_pretrained(args.bert_dir)
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    
    self.linear = nn.Linear(768, args.num_labels)
    self.crf = CRF(args.num_labels, batch_first=True)

  def forward(self, input_ids, attention_mask, labels=None):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    seq_out = bert_output[0] # [batchsize, max_len, 768]
    # print(seq_out.shape)
    seq_out = self.linear(seq_out)
    logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    loss = None
    if labels is not None:
      labels=torch.tensor(labels,dtype=torch.int64)
      loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    model_output = ModelOutput(logits, labels, loss)
    return model_output





class GlyphEmbedding(nn.Module):
  """Glyph2Image Embedding"""

  def __init__(self, font_npy_files):
      super(GlyphEmbedding, self).__init__()
      font_arrays = [
          np.load(np_file).astype(np.float32) for np_file in font_npy_files
      ]
      self.vocab_size = font_arrays[0].shape[0]
      self.font_num = len(font_arrays)
      self.font_size = font_arrays[0].shape[-1]
      # N, C, H, W
      font_array = np.stack(font_arrays, axis=1)
      self.embedding = nn.Embedding(
          num_embeddings=self.vocab_size,
          embedding_dim=self.font_size ** 2 * self.font_num,
          _weight=torch.from_numpy(font_array.reshape([self.vocab_size, -1]))
      )

  def forward(self, input_ids):
      """
          get glyph images for batch inputs
      Args:
          input_ids: [batch, sentence_length]
      Returns:
          images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
      """
      # return self.embedding(input_ids).view([-1, self.font_num, self.font_size, self.font_size])
      return self.embedding(input_ids)

class BertGlyphBiLSTMNer(nn.Module):
  def __init__(self, args):
    super(BertGlyphBiLSTMNer, self).__init__()
    self.bert = BertModel.from_pretrained(args.bert_dir)
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    self.glyphEmbedding = GlyphEmbedding(font_npy_files=args.glyph_dir)
    self.glyph_map = nn.Linear(1728, self.bert_config.hidden_size)
    hidden_size = self.bert_config.hidden_size
    self.lstm_hiden = 128
    self.max_seq_len = args.max_seq_len
    self.bilstm = nn.LSTM(hidden_size*2, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
               dropout=0.1)
    self.linear = nn.Linear(self.lstm_hiden * 2, args.num_labels)
    self.crf = CRF(args.num_labels, batch_first=True)

  def forward(self, input_ids, attention_mask, labels=None):
    glyph_out = self.glyphEmbedding(input_ids) #[12, 512, 1728]
    glyph_embeddings = self.glyph_map(glyph_out) #[12, 512, 768]
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    bert_embedding = bert_output[0]  # [batchsize, max_len, 768]
    
    concat_embeddings = torch.cat((bert_embedding, glyph_embeddings), 2) #[12, 512, 1536]
    batch_size = concat_embeddings.size(0)
    seq_out, _ = self.bilstm(concat_embeddings)
    
    seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
    seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
    seq_out = self.linear(seq_out)
    logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    loss = None
    if labels is not None:
      labels=torch.tensor(labels,dtype=torch.int64)
      loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    model_output = ModelOutput(logits, labels, loss)
    return model_output