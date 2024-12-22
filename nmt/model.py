import math
import torch
from torch import nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        return x


# Positional Encoding is added to the input embeddings to introduce a notion of order to the tokens


class positionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super(positionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2, )

        # Compute the positional encoding

        # Apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to the positional encoding matrix
        # shape: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].requires_grad_(False)
        return self.dropout(x)


# Layer Normalization


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.aplha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Check if d_model is divisible by n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_k = d_model // n_heads

        # Linear layers to project the queries, keys and values

        # Linear layer to project the queries
        self.W_q = nn.Linear(d_model, d_model)

        # Linear layer to project the keys
        self.W_k = nn.Linear(d_model, d_model)

        # Linear layer to project the values
        self.W_v = nn.Linear(d_model, d_model)

        # Linear layer to project the concatenated output of all the heads
        self.W_o = nn.Linear(d_model, d_model)

    def attention(query, key, value, mask, dropout: nn.Dropout):

        d_k = query.shape[-1]

        # Shape = (Batch_size, n_heads, seq_len, d_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # (Batch_size, n_heads, seq_len, seq_len)
        attention_scores = torch.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores) @ value, attention_scores

    def forward(self, q, k, v, mask=None):

        # Shape = (batch_size, seq_len, d_model)
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        # (Batch_size, seq_len, d_model) -> (Batch_size, seq_len, n_heads, d_k) -> (Batch_size, n_heads, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.n_heads, self.d_k
        ).transpose(1, 2)

        x, attention_scores = self.attention(query, key, value, mask, self.dropout)

        # shape (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)

        return self.W_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout=0.1):
        super(ResidualConnection, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))


class EncoderBlock(nn.Module):

    def __init__(
        self,
        self_attention: MultiHeadAttentionBlock,
        feed_forward: FeedForward,
        dropout: float,
    ):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention(x, x, x, src_mask)
        )
        x = self.residual_connection[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.encoder_layers = encoder_layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttentionBlock,
        cross_attention: MultiHeadAttentionBlock,
        feed_forward: FeedForward,
        dropout: float,
    ):
        super().__init__(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask)
        )
        x = self.residual_connection[1](
            x,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask),
        )
        x = self.residual_connection[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.decoder_layers = decoder_layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


class ProjectionBlock(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(ProjectionBlock, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch_size, seq_len, d_model) -> (Batch_size, seq_len, vocab_size)
        x = self.linear(x)
        x = torch.log_softmax(x, dim=-1)
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos: positionalEncoding,
        tgt_pos: positionalEncoding,
        projection_layer: ProjectionBlock,
    ):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):

        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size,
    tgt_vocab_size,
    src_seq_len,
    tgt_seq_len,
    d_model=512,
    n=6,
    n_heads=8,
    dropout=0.1,
    d_ff=2048,
):

    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    src_pos = positionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = positionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []

    for _ in range(n):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        encoder_feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention, encoder_feed_forward, dropout
        )
        encoder_blocks.append(encoder_block)

    decoder_blocks = []

    for _ in range(n):
        decoder_self_attention = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        decoder_feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention,
            decoder_cross_attention,
            decoder_feed_forward,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer

    projection_layer = ProjectionBlock(d_model, tgt_vocab_size)

    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the weights of the model

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
