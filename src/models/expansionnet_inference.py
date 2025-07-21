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
    def estimate_layer_complexity(self) -> List[float]:
        """Estimate computational complexity for each layer"""
        complexities = []
        
        # Encoder complexities (attention + FFN)
        for i in range(self.N_enc):
            # Self-attention: O(seq_len^2 * model_dim)
            # FFN: O(seq_len * model_dim * ffn_dim)
            attention_ops = 196 * 196 * self.model_dim  # Assuming 14x14 visual features
            ffn_ops = 196 * self.model_dim * (self.model_dim * 4)
            total_ops = (attention_ops + ffn_ops) / 1e9  # Convert to GFLOPs
            complexities.append(total_ops)
        
        # Decoder complexities (self-attention + cross-attention + FFN)
        for i in range(self.N_dec):
            seq_len = 20  # Typical caption length
            self_attention_ops = seq_len * seq_len * self.model_dim
            cross_attention_ops = seq_len * 196 * self.model_dim
            ffn_ops = seq_len * self.model_dim * (self.model_dim * 4)
            total_ops = (self_attention_ops + cross_attention_ops + ffn_ops) / 1e9
            complexities.append(total_ops)
        
        return complexities
    def forward_device_segment(self, device_id: int, input_data: Dict[str, torch.Tensor], split_config: SplitConfiguration) -> Dict[str, torch.Tensor]:
        """
        Execute forward pass for a specific device segment
        """
        # Find this device's assignment
        assignment = None
        for assign in split_config.device_assignments:
            if assign.device_id == device_id:
                assignment = assign
                break
        
        if assignment is None:
            raise ValueError(f"No assignment found for device {device_id}")
        
        output_data = {}
        
        # Visual encoding (device 0 only) - FIX: Handle tensor dimensions properly
        if device_id == 0 and 'images' in input_data:
            images = input_data['images']
            
            # Ensure correct input dimensions for conv2d
            if images.dim() == 5:  # Remove extra dimension if present
                images = images.squeeze(1)
            elif images.dim() == 3:  # Add batch dimension if missing
                images = images.unsqueeze(0)
            
            # Apply visual encoder
            visual_features = self.visual_encoder(images)
            
            # Ensure correct output shape
            if visual_features.dim() == 2:
                visual_features = visual_features.unsqueeze(0)
            
            output_data['visual_features'] = visual_features
            output_data['encoder_input'] = visual_features
        
        # Encoder processing - FIX: Handle tensor shapes consistently
        if assignment.layer_start < self.N_enc:
            encoder_start = max(0, assignment.layer_start)
            encoder_end = min(self.N_enc, assignment.layer_end + 1)
            
            encoder_input = input_data.get('encoder_input', input_data.get('visual_features'))
            if encoder_input is not None:
                # Ensure correct tensor dimensions
                if encoder_input.dim() == 2:
                    encoder_input = encoder_input.unsqueeze(0)
                
                for i in range(encoder_start, encoder_end):
                    encoder_input = self.encoder_layers[i](encoder_input)
                
                output_data['encoder_output'] = encoder_input
                
                # Mark if encoder is complete
                if encoder_end == self.N_enc:
                    output_data['encoder_complete'] = encoder_input
        
        # Decoder processing - FIX: Handle cross-attention properly
        decoder_start_layer = assignment.layer_start - self.N_enc
        decoder_end_layer = assignment.layer_end - self.N_enc
        
        if decoder_start_layer < self.N_dec and decoder_end_layer >= 0:
            dec_start = max(0, decoder_start_layer)
            dec_end = min(self.N_dec, decoder_end_layer + 1)
            
            # Initialize decoder if this is the first decoder device
            if dec_start == 0 and 'captions' in input_data:
                captions = input_data['captions']
                if captions.dim() == 1:
                    captions = captions.unsqueeze(0)
                
                token_embeddings = self.token_embedding(captions)
                decoder_input = self.pos_encoding(token_embeddings)
            elif 'decoder_output' in input_data:
                decoder_input = input_data['decoder_output']
            else:
                decoder_input = None
            
            # Process decoder layers
            if decoder_input is not None:
                encoder_memory = input_data.get('encoder_complete', input_data.get('encoder_output'))
                
                if encoder_memory is not None:
                    # Ensure compatible dimensions for cross-attention
                    if encoder_memory.dim() == 2:
                        encoder_memory = encoder_memory.unsqueeze(0)
                    if decoder_input.dim() == 2:
                        decoder_input = decoder_input.unsqueeze(0)
                    
                    for i in range(dec_start, dec_end):
                        decoder_input = self.decoder_layers[i](decoder_input, encoder_memory)
                    
                    output_data['decoder_output'] = decoder_input
                    
                    # Generate final output if this is the last decoder device
                    if dec_end == self.N_dec:
                        logits = self.output_projection(decoder_input)
                        output_data['final_logits'] = logits
        
        # Add metadata
        output_data['device_id'] = device_id
        output_data['processing_complete'] = True
        
        return output_data


    
    def get_total_layers(self) -> int:
        return self.N_enc + self.N_dec

# Import remaining components from the original files
from .components.transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer
from .components.positional_encoding import PositionalEncoding
