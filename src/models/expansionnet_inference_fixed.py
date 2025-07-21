import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class DeviceAssignment:
    device_id: int
    layer_start: int
    layer_end: int
    device_type: str
    estimated_time: float = 0.0

@dataclass
class SplitConfiguration:
    device_assignments: List[DeviceAssignment]
    total_devices: int
    strategy: str
    estimated_total_time: float = 0.0

class VisualEncoderTransform(nn.Module):
    """Fixed visual encoder transform"""
    
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.projection = nn.Linear(128, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        # Input: (B, 128, 14, 14)
        B, C, H, W = x.shape
        
        # Reshape to (B, H*W, C) for transformer format
        x = x.flatten(2).permute(0, 2, 1)  # (B, 196, 128)
        
        # Project and normalize
        x = self.projection(x)  # (B, 196, model_dim)
        x = self.layer_norm(x)
        
        return x

class InferenceExpansionNet(nn.Module):
    """Fixed ExpansionNet v2 for multi-device inference"""
    
    def __init__(self, model_dim=512, N_enc=6, N_dec=6, max_seq_len=100, 
                 dropout=0.3, vocab_size=10201):
        super().__init__()
        self.model_dim = model_dim
        self.N_enc = N_enc
        self.N_dec = N_dec
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # Fixed visual encoder
        self.visual_encoder = self._build_visual_encoder()
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, 8, model_dim*4, dropout)
            for _ in range(N_enc)
        ])
        
        # Decoder layers  
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(model_dim, 8, model_dim*4, dropout)
            for _ in range(N_dec)
        ])
        
        # Output layers
        self.output_projection = nn.Linear(model_dim, vocab_size)
        self.pos_encoding = PositionalEncoding(model_dim, max_seq_len)
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        
    def _build_visual_encoder(self):
        """Fixed visual feature extraction"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((14, 14)),
            VisualEncoderTransform(self.model_dim)
        )
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor = None) -> torch.Tensor:
        """Fixed forward pass"""
        # Visual encoding - now outputs (B, 196, model_dim)
        visual_features = self.visual_encoder(images)
        
        # Encoder
        encoder_output = visual_features
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        
        if captions is not None:
            # Decoder
            token_embeddings = self.token_embedding(captions)
            decoder_input = self.pos_encoding(token_embeddings)
            
            for layer in self.decoder_layers:
                decoder_input = layer(decoder_input, encoder_output)
            
            # Output
            logits = self.output_projection(decoder_input)
            return logits
        
        return encoder_output
    
    def get_total_layers(self) -> int:
        return self.N_enc + self.N_dec
    
    def estimate_layer_complexity(self) -> List[float]:
        """Estimate computational complexity for each layer"""
        complexities = []
        
        # Encoder complexities
        for i in range(self.N_enc):
            attention_ops = 196 * 196 * self.model_dim
            ffn_ops = 196 * self.model_dim * (self.model_dim * 4)
            total_ops = (attention_ops + ffn_ops) / 1e9
            complexities.append(total_ops)
        
        # Decoder complexities
        for i in range(self.N_dec):
            seq_len = 20
            self_attention_ops = seq_len * seq_len * self.model_dim
            cross_attention_ops = seq_len * 196 * self.model_dim
            ffn_ops = seq_len * self.model_dim * (self.model_dim * 4)
            total_ops = (self_attention_ops + cross_attention_ops + ffn_ops) / 1e9
            complexities.append(total_ops)
        
        return complexities

# Import component classes
from .components.transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer
from .components.positional_encoding import PositionalEncoding
