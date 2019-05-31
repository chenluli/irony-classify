# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: env3.5
#     language: python
#     name: env3.5
# ---

# +
import os
import os.path as osp

import json
import numpy as np
import torch
import torch.nn as nn
# -
# #### Word Embedding




# #### Positional Encoding  
# $P E_{2 i}(p)=\sin \left(p / 10000^{2 i / d_{model}}\right)$   
# $P E_{2 i+1}(p)=\cos \left(p / 10000^{2 i / d_{model}}\right)$

def positional_embd(dim, sentence_len):
    base = 10000
    vec = np.array([pos / np.power(base, 2 * i / dim) for pos in range(sentence_len) for i in range(dim)], dtype=np.float32)
    vec[::2] = np.sin(vec[::2])
    vec[1::2] = np.cos(vec[1::2])
    return torch.from_numpy(vec.reshape([1, sentence_len, dim]))

# obtain input encoding by adding position embd and word embd


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


# #### Attention
#  $\text { Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V$

# self-attention using scaled dot-product 
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    dim_key = query.size(-1)
    attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_key)
    if mask is not None:
        attn = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim = -1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights, value), attn_weights


class MultiHeadedAttention():
    def __init__(self, num_heads, dim_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # make sure input word embedding dimension divides by the number of desired heads
        assert dim_model % num_heads == 0
        # assume dim of key,query,values are equal
        self.dim_k = dim_model // num_heads
        self.dim_v = dim_model // num_heads 
        self.dim_model = dim_model
        self.num_h = num_heads
        self.w_q = nn.Linear(dim_model, dim_model) # self.w_qs = nn.Linear(d_model, n_head * d_k) 
        self.w_v = nn.Linear(dim_model, dim_model)
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        n_batch = query.size(0)
        residual = query
        
        # linear projections
        query = self.w_q(query).view(n_batch, -1, self.num_h, self.dim_qkv)
        key = self.w_q(key).view(n_batch, -1, self.num_h, self.dim_qkv)
        value = self.w_v(value).view(n_batch, -1, self.num_h, self.dim_qkv)
        
        # Apply attention on all the projected vectors in batch 
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # Concat(head1, ..., headh) 
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.num_h * self.dim_qkv)
#         x = nn.Linear(d_model, d_model, bias=False)(x)
        x = self.layer_norm(x + residual)
        return x


# +
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

class AttnClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def set_embedding(self, vectors):
        self.embedding.weight.data.copy_(vectors)
        
    def forward(self, inputs, lengths):
        batch_size = inputs.size(1)
        # (L, B)
        embedded = self.embedding(inputs)
        # (L, B, E)
        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        out, hidden = self.lstm(packed_emb)
        out = nn.utils.rnn.pad_packed_sequence(out)[0]
        out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        # (L, B, H)
        embedding, attn_weights = self.attention(out.transpose(0, 1))
        # (B, HOP, H)
        outputs = self.fc(embedding.view(batch_size, -1))
        # (B, 1)
        return outputs, attn_weights


# -

# #### Position-wise feed forward network

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_model, d_ff, 1)
        self.w_2 = nn.Conv1d(d_ff, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# #### Add & Norm
# `Residual connection`是对于较为深层的神经网络有比较好的作用，比如网络层很深时，数值的传播随着weight不断的减弱，`Residual connection`是从输入的部分，连到它输出层的部分，把输入的信息原封不动copy到输出的部分，减少信息的损失。
# `layer-normalization`这种归一化层是为了防止在某些层中由于某些位置过大或者过小导致数值过大或过小，对神经网络梯度回传时有训练的问题，保证训练的稳定性。基本在每个子网络后面都要加上`layer-normalization`、加上`Residual connection`，加上这两个部分能够使深层神经网络训练更加顺利。  
# (本实验中也许不需要)

class AddNorm():
    def __init__(self, size, dropout, eps=1e-6):
        super(AddNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        norm = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x + self.dropout(sublayer(norm))


# #### Encoder

# 一层Encoder: self-atten --> add&norm --> feed-forward --> add&norm
class EncoderLayer(nn.Module):
    def __init__(self, size, attention, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.feed_forward = feed_forward
        self.self_atten = attention
        self.add_norm_1 = AddNorm(size, dropout)
        self.add_norm_2 = AddNorm(size, dropout)
        self.size = size

    def forward(self, x, mask):
        output = self.add_norm_1(x, lambda x: self.self_atten(x, x, x, mask))
        output = self.add_norm_2(output, self.feed_forward)
        return output


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N) # clone the layer for N times
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# #### Full Model

num_encode_layers = 2
class transformerClassifier():
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        def __init__(self, input_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def set_embedding(self, vectors):
        self.embedding.weight.data.copy_(vectors)
        
    def forward(self, inputs, lengths):
        batch_size = inputs.size(1)
        # (L, B)
        embedded = self.embedding(inputs)
        # (L, B, E)
        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        out, hidden = self.lstm(packed_emb)
        out = nn.utils.rnn.pad_packed_sequence(out)[0]
        out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        # (L, B, H)
        embedding, attn_weights = self.attention(out.transpose(0, 1))
        # (B, HOP, H)
        outputs = self.fc(embedding.view(batch_size, -1))
        # (B, 1)
        return outputs, attn_weights


def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def transformerClassifier(x_tensor, output_dim, wordIndxToVec_tensor, dropoutKeep_tensor, max_sentence_length):
    with tf.variable_scope("Embedding_Layer"):
        emb = tf.nn.embedding_lookup(wordIndxToVec_tensor, x_tensor)

    # Add positional encodings to the embeddings we feed to the encoder.
    if hp.include_positional_encoding:
        with tf.variable_scope("Add_Position_Encoding"):
            posEnc = positional_encoding(hp.model_dim, max_sentence_length)
            emb = tf.add(emb, posEnc, name="Add_Positional_Encoding")
            
    if hp.input_emb_apply_dropout:
        with tf.variable_scope("Input_Embeddings_Dropout"):
            emb = tf.nn.dropout(emb, keep_prob=dropoutKeep_tensor)          # ignore some input info to regularize the model

    for i in range(1, hp.num_layers + 1):
        with tf.variable_scope("Stack-Layer-{0}".format(i)):
            encoder_output = encoder_layer(emb, dropout_keep_prob_tensor=dropoutKeep_tensor)
            emb = encoder_output

    # Simply average the final sequence position representations to create a fixed size "sentence representation".
    sentence_representation = tf.reduce_mean(encoder_output, axis=1)    # [batch_size, model_dim]

    with tf.variable_scope("Sentence_Representation_And_Output"):
        sentence_representation = tf.layers.dense(sentence_representation, hp.model_dim, activation=tf.nn.relu, use_bias=True,
                                          kernel_initializer=tf.glorot_normal_initializer(), bias_initializer=tf.zeros_initializer())
        if hp.sentence_representation_dropout:
            sentence_representation = tf.nn.dropout(sentence_representation, keep_prob=dropoutKeep_tensor)          # ignore some input info to regularize the model

        prediction_logits = tf.layers.dense(sentence_representation, output_dim, activation=None, use_bias=False, kernel_initializer=tf.glorot_normal_initializer())

    return prediction_logits

# ### Train



# ### Evaluation


