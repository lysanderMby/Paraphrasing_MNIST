import torch
import torch.nn as nn

class BaseInterpretableClassifier(nn.Module):
    """Base class for interpretable classifiers"""
    def get_param_count(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())

class DefaultInterpretableClassifier(BaseInterpretableClassifier):
    """Original interpretable classifier architecture with residual connections"""
    def __init__(self, layer_configs, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        for in_channels, out_channels in layer_configs:
            # Main convolutional block
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*2, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels*2),
                nn.ReLU(),
                nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.layers.append(layer)
            
            # Residual projection if needed
            if in_channels != out_channels:
                proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                proj = nn.Identity()
            self.residual_projections.append(proj)
        
        # Final classifier
        final_channels = layer_configs[-1][1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(final_channels * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, return_intermediates=False):
        intermediates = []
        current = x
        
        for layer, proj in zip(self.layers, self.residual_projections):
            layer_out = layer(current)
            residual = proj(current)
            current = layer_out + residual
            intermediates.append(current)
        
        logits = self.classifier(current)
        
        if return_intermediates:
            return logits, intermediates
        return logits

class SimpleInterpretableClassifier(BaseInterpretableClassifier):
    """Simpler architecture without residual connections"""
    def __init__(self, layer_configs, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for in_channels, out_channels in layer_configs:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.layers.append(layer)
        
        final_channels = layer_configs[-1][1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(final_channels * 7 * 7, num_classes)
        )

    def forward(self, x, return_intermediates=False):
        intermediates = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            intermediates.append(current)
        
        logits = self.classifier(current)
        
        if return_intermediates:
            return logits, intermediates
        return logits

# Dictionary mapping model names to their classes
MODEL_ARCHITECTURES = {
    'default': DefaultInterpretableClassifier,
    'simple': SimpleInterpretableClassifier
} 