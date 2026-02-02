"""
Tactile Encoder Module for VTLA-ACT

This module provides tactile encoding capabilities for the ACT model,
enabling the model to process tactile sensor data (contact forces)
alongside visual and proprioceptive observations.
"""

import torch
import torch.nn as nn
from typing import Optional


class TactileEncoder(nn.Module):
    """
    MLP-based tactile encoder for contact force data.
    
    This encoder processes tactile data (contact forces from gripper fingers)
    and produces embeddings compatible with the ACT transformer architecture.
    """
    
    def __init__(
        self,
        tactile_dim: int = 6,      # 2 fingers * 3D force
        hidden_dim: int = 256,     # Output dimension (matches transformer dim)
        num_layers: int = 2,       # Number of MLP layers
        dropout: float = 0.1,      # Dropout rate
        use_layer_norm: bool = True,
    ):
        """
        Initialize the tactile encoder.
        
        Args:
            tactile_dim: Dimension of input tactile data
            hidden_dim: Output embedding dimension
            num_layers: Number of MLP layers
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.tactile_dim = tactile_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build MLP layers
        layers = []
        in_dim = tactile_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Final projection (optional, for matching exact hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tactile: Tactile data tensor of shape (batch, tactile_dim)
            
        Returns:
            tactile_embed: Tactile embeddings of shape (batch, hidden_dim)
        """
        x = self.mlp(tactile)
        x = self.output_proj(x)
        return x


class TactileCNNEncoder(nn.Module):
    """
    CNN-based tactile encoder for tactile image data.
    
    This encoder is designed for tactile sensors that produce image-like data
    (e.g., GelSight, DIGIT sensors). Use this when tactile data is in image form.
    
    Note: This is provided for future extensibility. Current implementation
    uses contact forces, which should use TactileEncoder instead.
    """
    
    def __init__(
        self,
        in_channels: int = 3,       # RGB tactile image
        hidden_dim: int = 256,
        image_size: tuple = (64, 64),
    ):
        """
        Initialize the CNN tactile encoder.
        
        Args:
            in_channels: Number of input channels
            hidden_dim: Output embedding dimension
            image_size: Input image size (H, W)
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        # Calculate flatten size
        self.flatten_size = 128 * 4 * 4
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, tactile_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tactile_image: Tactile image tensor of shape (batch, C, H, W)
            
        Returns:
            tactile_embed: Tactile embeddings of shape (batch, hidden_dim)
        """
        x = self.encoder(tactile_image)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class TactilePositionEmbedding(nn.Module):
    """
    Learnable position embedding for tactile tokens in the transformer.
    """
    
    def __init__(self, hidden_dim: int = 256):
        """
        Initialize position embedding.
        
        Args:
            hidden_dim: Embedding dimension
        """
        super().__init__()
        self.pos_embed = nn.Embedding(1, hidden_dim)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get position embeddings.
        
        Args:
            batch_size: Batch size
            
        Returns:
            pos: Position embeddings of shape (batch, 1, hidden_dim)
        """
        return self.pos_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)


def build_tactile_encoder(
    tactile_type: str = "force",
    tactile_dim: int = 6,
    hidden_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to build tactile encoder.
    
    Args:
        tactile_type: Type of tactile data ("force" or "image")
        tactile_dim: Dimension of tactile input (for force type)
        hidden_dim: Output embedding dimension
        **kwargs: Additional arguments passed to encoder
        
    Returns:
        encoder: Tactile encoder module
    """
    if tactile_type == "force":
        return TactileEncoder(
            tactile_dim=tactile_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
    elif tactile_type == "image":
        return TactileCNNEncoder(
            hidden_dim=hidden_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown tactile type: {tactile_type}")


if __name__ == "__main__":
    # Test tactile encoders
    batch_size = 4
    tactile_dim = 6
    hidden_dim = 256
    
    # Test MLP encoder (for force data)
    encoder = TactileEncoder(tactile_dim=tactile_dim, hidden_dim=hidden_dim)
    tactile_input = torch.randn(batch_size, tactile_dim)
    output = encoder(tactile_input)
    print(f"MLP Encoder output shape: {output.shape}")
    assert output.shape == (batch_size, hidden_dim)
    
    # Test CNN encoder (for image data)
    cnn_encoder = TactileCNNEncoder(hidden_dim=hidden_dim)
    image_input = torch.randn(batch_size, 3, 64, 64)
    output = cnn_encoder(image_input)
    print(f"CNN Encoder output shape: {output.shape}")
    assert output.shape == (batch_size, hidden_dim)
    
    # Test factory function
    encoder = build_tactile_encoder("force", tactile_dim=6, hidden_dim=256)
    output = encoder(torch.randn(2, 6))
    print(f"Factory encoder output shape: {output.shape}")
    
    print("\nAll tactile encoder tests passed!")
