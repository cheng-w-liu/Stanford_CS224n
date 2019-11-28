#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    """
    Implemented 1D Convolutional layer on text data
    """

    def __init__(self, word_embed_dim=256, char_embed_dim=50, kernel_size=5):
        """
        1D Convolutional Network
        :param char_embed_dim: dimension for the character embedding
        :param word_embed_dim: dimension for the word embedding
        :param kernel_size: size of the kernel/filter
        """
        super(CNN, self).__init__()

        self.cnn_layer = nn.Conv1d(
            in_channels=char_embed_dim,
            out_channels=word_embed_dim,
            kernel_size=kernel_size,
            bias=False
        )
        self.kernel_size = kernel_size

    def forward(self, X_reshaped: torch.Tensor) -> torch.Tensor:
        """
        :param X_reshaped: shape: (batch_size, char_embed, m_word)
        :return: X_conv_out: shape: (batch_size, word_embed)
        """
        max_word_len = X_reshaped.size(2)

        # X_conv shape: (batch_size, word_embed, m_word-k+1)
        X_conv = self.cnn_layer(X_reshaped)
        X_conv_out = torch.squeeze(
            F.max_pool1d(
                F.relu(X_conv),
                max_word_len-self.kernel_size+1
            )
        )
        return X_conv_out

### END YOUR CODE

