#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()

        self.target_vocab = target_vocab

        self.decoderCharEmb = nn.Embedding(
            len(target_vocab.char2id),
            char_embedding_size,
            padding_idx=target_vocab.char2id['<pad>']
        )

        self.charDecoder = nn.LSTM(
            input_size=char_embedding_size,
            hidden_size=hidden_size,
            batch_first=False
        )

        self.char_output_projection = nn.Linear(
            in_features=hidden_size,
            out_features=len(target_vocab.char2id),
            bias=True
        )
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        X = self.decoderCharEmb(input)  # (length, batch, char_embedding_size)
        output, dec_hidden_n = self.charDecoder(X, dec_hidden)
        # output shape: (seq_len, batch, hidden_size)
        S_t = self.char_output_projection(output) # (seq_len, batch, V_char)

        return S_t, dec_hidden_n
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        S_t, dec_hidden_n = self.forward(char_sequence[:-1, :], dec_hidden)  # extract [x_1, x_2, ...,x_n]
        # s_t shape: (seq_len-1, batch_size, V_char)
        # dec_hidden_n: a tuple of two tensors of shape (1, batch, hidden_size)

        loss_char_dec = 0.0
        loss_func = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        t1 = 1
        for s_t in torch.split(S_t, split_size_or_sections=1, dim=0):
            s_t = torch.squeeze(s_t, dim=0) # shape: (batch_size, V_char)
            loss_char_dec += loss_func(s_t, char_sequence[t1, :])
            t1 += 1
        return loss_char_dec

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        #model = self.charDecoder.to(device)
        batch_size = initialStates[0].shape[1]

        char_start_idx = self.target_vocab.char2id['{']
        curr_input = torch.tensor([char_start_idx] * batch_size, device=device).view(1, -1)
        output_words = [''] * batch_size
        dec_hidden = initialStates
        for t in range(max_length-1):
            s_t, dec_hidden = self.forward(curr_input, dec_hidden)
            s_t = torch.squeeze(s_t, dim=0) # (batch_size, V_char)
            indices = torch.max(s_t, dim=1)[1]  # (batch_size, )

            curr_chars = [self.target_vocab.id2char[idx.item()] for idx in indices]
            for i, w in enumerate(output_words):
                output_words[i] = output_words[i] + curr_chars[i]

            curr_input = torch.tensor(indices, device=device).view(1, -1)

        for i in range(len(output_words)):
            e = output_words[i].find('}')
            output_words[i] = output_words[i][:e]

        return output_words
        
        ### END YOUR CODE

