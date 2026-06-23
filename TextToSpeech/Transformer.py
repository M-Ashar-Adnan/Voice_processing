import torch;
import torch.nn as nn;
import math;

#Position Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        
        position  = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x +self.pe[:, :x.size(1), :]
        return x
    
    
# The Core TTS Transformer Model
class TTSBrain(nn.Module):
    def __init__(self, text_vocab_size, audio_vocab_size, embed_dim=256, num_heads=8, num_layers=4):
        super(TTSBrain, self).__init__()
        
        self.text_embedding = nn.Embedding(text_vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.audio_predictor = nn.Linear(embed_dim, audio_vocab_size)
        
    def forward(self, text_tokens):
        x = self.text_embedding(text_tokens)
        x = self.pos_encoder(x)
        x=self.transformer_encoder(x)
        
        audio_logits = self.audio_predictor(x)
        return audio_logits
    
#Simulating a Forward Pass
TEXT_VOCAB_SIZE = 100
AUDIO_VOCAB_SIZE = 1024
EMBED_DIM = 256

tts_model = TTSBrain(
 text_vocab_size=TEXT_VOCAB_SIZE,
    audio_vocab_size=AUDIO_VOCAB_SIZE,
    embed_dim=EMBED_DIM   
)
        
        
#Creating dummy data
text_batch = torch.randint(0, TEXT_VOCAB_SIZE, (2, 10))  
print(f"Input Text shape: {text_batch.shape}")

with torch.no_grad():
    predicted_audio_logits = tts_model(text_batch)
    
print(f"Output Audio Logits shape: {predicted_audio_logits.shape} -> (Batch Size: 2, Sequence Length: 10, Audio Vocabulary Size: {AUDIO_VOCAB_SIZE})")

predicted_audio_tokens = torch.argmax(predicted_audio_logits, dim = -1)
print(f"\nPredicted Audio Tokens Shape: {predicted_audio_tokens.shape} -> (Batch Size :2, Audio length : 15)")
print(f"Sentence 1 Predicted Audio Tokens: \n{predicted_audio_tokens[0].tolist()}")