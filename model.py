import torch.nn as nn
import timm

class DRModel(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        # Backbone: EfficientNet-B3 (features only)
        self.cnn = timm.create_model("efficientnet_b3", pretrained=False, features_only=True)
        cnn_channels = self.cnn.feature_info[-1]["num_chs"]
        
        # Mapping 384 -> 256
        self.proj = nn.Conv2d(cnn_channels, embed_dim, 1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CORAL: 5 levels -> 4 logits
        self.head = nn.Linear(embed_dim, 4)

    def forward(self, x):
        feats = self.cnn(x)[-1]
        feats = self.proj(feats)
        B, E, H, W = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)
