from dataclasses import dataclass
import torch 
import torch.nn as nn 
import math 

@dataclass
class TransformerConfig:
    source_vocab_size: int
    target_vocab_size: int 
    source_sequence_length: int 
    target_sequence_length: int 
    d_model: int = 512 
    layers: int = 6 
    heads: int = 8 
    dropout: float = 0.1 
    d_ff: int = 2048 

class SentenceEmbeddingLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model  # d_model = model dimension = vector  
        self.vocab_size = vocab_size 
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model: int, sequence_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model 
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        PE = torch.zeros(sequence_length, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze()
        deviation_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        PE[:, 0::2] = torch.sin(position * deviation_term)
        PE[:, 1::2] = torch.cos(position * deviation_term)
        PE = PE.unsqueeze()
        self.register_buffer(PE, 'PE')

    def forward(self, x):
        x = x + (self.PE[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class NormalizationLayer(nn.Module):
    def __init__(self, Epsilon: float = 10**-6) -> None:
        super().__init__()
        self.Epsilon = Epsilon
        self.Alpha = nn.Parameter(torch.ones(1)) # learnable parameters (optional) 
        self.Bias = nn.Parameter(torch.ones(1)) # learnable parameters  (optional)
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.Alpha * (x - mean) / (std + self.Epsilon) + self.Bias

# TransformerMLP = Transformer Multi-Layer Preceptorn
class TransformerMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # d_ff = dimensions of feed-forward neural network
        self.Linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.Linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.Linear_2(self.dropout(torch.relu(self.Linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads 
        assert d_model % heads == 0, "d_model is not divisable by heads"
        self.d_k = d_model // heads # d_k = dimensions of the model
        
        self.W_Q = nn.Linear(d_model, d_model) # Query weights
        self.W_K = nn.Linear(d_model, d_model) # Key weights
        self.W_V = nn.Linear(d_model, d_model) # Value weights
        self.W_O = nn.Linear(d_model, d_model) # output weights

    @staticmethod
    def AttentionFormula(Query, Key, Value, mask, dropout: nn.Dropout):
        d_k = Query.shape[-1]
        self_attention_scores = (Query @ Key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            self_attention_scores = self_attention_scores.masked_fill_(mask == 0, -1e9)
        self_attention_scores = self_attention_scores.softmax(dim = -1)
        if dropout is not None:
            self_attention_scores = dropout(self_attention_scores)
        return self_attention_scores @ Value 

    def forward(self, query, key, value, mask):
        Query = self.W_Q(query)
        Key = self.W_K(key)
        Value = self.W_V(value)

        Query = Query.view(Query.shape[0], Query.shape[1], self.heads, self.d_k).transpose(1,2)
        Key = Key.view(Key.shape[0], Key.shape[1], self.heads, self.d_k).transpose(1,2)
        Value = Value.view(Value.shape[0], Value.shape[1], self.heads, self.d_k).transpose(1,2)

        x, self.self_attention_scores = MultiHeadAttentionBlock.AttentionFormula(Query, Key, Value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)
        return self.W_O(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = NormalizationLayer()
    
    def forward(self, x, subLayer):
        return x + self.dropout(subLayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, 
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: TransformerMLP,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, source_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x 

class SequentialEncoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = NormalizationLayer()
    
    def forward(self, x, source_mask):
        for layer in self.layers:
            x = layer(x, source_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, 
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: TransformerMLP,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, target_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class SequentialDecoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers 
        self.norm = NormalizationLayer()
    
    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)

class LinearLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.Linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.Linear(x)

class TransformerBlock(nn.Module):
    def __init__(self,
                 encoder: SequentialEncoder,
                 decoder: SequentialDecoder,
                 source_embedding: SentenceEmbeddingLayer,
                 target_embedding: SentenceEmbeddingLayer,
                 source_position: PositionalEncodingLayer,
                 target_position: PositionalEncodingLayer,
                 linear: LinearLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_position = source_position
        self.target_position = target_position
        self.linear = linear 

    def encode(self, source_language, source_mask):
        source_language = self.source_embedding(source_language)
        source_language = self.source_position(source_language)
        return self.encoder(source_language, source_mask)
    
    def decode(self, encoder_output, source_mask, target_language, target_mask):
        target_language = self.target_embedding(target_language)
        target_language = self.target_position(target_language)
        return self.decoder(target_language, encoder_output, source_mask, target_mask)

    def linear_layer(self, x):
        return self.linear(x)

def TransformerModel(transformer: TransformerConfig)->TransformerBlock:

    source_embedding = SentenceEmbeddingLayer(transformer.d_model, transformer.source_vocab_size)
    source_position = PositionalEncodingLayer(transformer.d_model, transformer.source_sequence_length, transformer.dropout)
    
    target_embedding = SentenceEmbeddingLayer(transformer.d_model, transformer.target_vocab_size)
    target_position = PositionalEncodingLayer(transformer.d_model, transformer.target_sequence_length, transformer.dropout)

    encoder_blocks = []
    for _ in range(transformer.layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(transformer.d_model, transformer.heads, transformer.dropout)
        encoder_feed_forward_block = TransformerMLP(transformer.d_model, transformer.d_ff, transformer.dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, transformer.dropout)
        encoder_blocks.append(encoder_block)

    
    DecoderBlocks = []
    for _ in range(transformer.layers):
        decoder_self_attention_block = MultiHeadAttentionBlock(transformer.d_model, transformer.heads, transformer.dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(transformer.d_model, transformer.heads, transformer.dropout)
        decoder_feed_forward_block = TransformerMLP(transformer.d_model, transformer.d_ff, transformer.dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, transformer.dropout)
        DecoderBlocks.append(decoder_block)


    encoder = SequentialEncoder(nn.ModuleList(encoder_blocks))
    decoder = SequentialDecoder(nn.ModuleList(DecoderBlocks))

    linear = LinearLayer(transformer.d_model, transformer.target_vocab_size)

    Transformer = TransformerBlock(encoder, decoder, source_embedding, target_embedding, source_position, target_position, linear)

    for T in Transformer.parameters():
        if T.dim() > 1:
            nn.init.xavier_uniform(T)

    return Transformer
