#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        pad_token_idx = vocab.char2id['<pad>']
        char_embed_dim = 50
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(len(vocab.char2id), char_embed_dim, padding_idx=pad_token_idx)

        ### YOUR CODE HERE for part 1j
        self.cnn_layer = CNN(embed_size)
        self.highway_layer = Highway(embed_size)
        self.dropout_layer = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """

        ### YOUR CODE HERE for part 1j
        # X_embed: shape: (sentence_length, batch_size, max_word_length, char_embed_dim)
        X_embed = self.embeddings(input)
        X_word_emb = []
        for x_embed in torch.split(X_embed, split_size_or_sections=1, dim=0):
            x_embed = torch.squeeze(x_embed, dim=0)
            # x_embed shape: (batch_size, max_word_length, char_embed_dim)
            #   needs to reshape x_embed to (batch_size, char_embed_dim, max_word_length)
            # x_conv_out: shape: (batch_size, word_embed_dim)
            x_conv_out = self.cnn_layer(torch.transpose(x_embed, 1, 2))

            # x_highway: shape: (batch_size, word_embed_dim)
            x_highway = self.highway_layer(x_conv_out)

            x_word_emb = self.dropout_layer(x_highway)

            X_word_emb.append(x_word_emb)

        # X_word_embed: shape: (sentence_length, batch_size, word_embed_dim)
        X_word_emb = torch.stack(X_word_emb, dim=0)
        return X_word_emb

        ### END YOUR CODE

