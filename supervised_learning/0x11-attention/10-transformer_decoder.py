#!/usr/bin/env python3
"""Module Decoder class."""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Create the decoder for a transformer.

    Args:
        tf (tensor): class inherits from
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor.

        Args:
            N (int): number of blocks in the encoder
            dm (int): dimensionality of the model
            h (int): number of heads
            hidden (int): number of hidden units in the fully connected layer
            target_vocab (int): size of the target vocabulary
            max_seq_len (int): maximum sequence length possible
            drop_rate (float, optional): dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
                       ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Call method.

        Args:
            x (tensor): contains the input to the decoder
            encoder_output (tensor): contains the output of the encoder
            training (boolean): determines if the model is training
            look_ahead_mask (tensor): mask to be applied to the first multi
                                      head attention layer
            padding_mask (tensor): mask to be applied to the second multi head
                                   attention layer
        Returns:
            tensor containing the decoder output
        """
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training, look_ahead_mask,
                               padding_mask)
        return x
