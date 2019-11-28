#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, emb_dim):
        """
        Highway network defined in Eq.(8) - Eq.(10)

        :param emb_dim: embedding dimension
        :returns X_highway (Tensor)
        """
        super(Highway, self).__init__()

        self.w_projection = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=True)
        self.gate_projection = nn.Linear(in_features=emb_dim, out_features=emb_dim, bias=True)

    def forward(self, X_conv_out: torch.Tensor) -> torch.Tensor:
        """
        Perform operations defined in Eq.(8) - Eq.(10)
        :param X_conv_out (Tensor): shape (batch_size, embed_size)
        :return: X_highway (Tensor): shape (batch_size, embed_size)
        """

        X_proj = F.relu(self.w_projection(X_conv_out))
        X_gate = torch.sigmoid(self.gate_projection(X_conv_out))
        X_highway = torch.mul(X_gate, X_proj) + torch.mul(1 - X_gate, X_conv_out)
        return X_highway

### END YOUR CODE 

