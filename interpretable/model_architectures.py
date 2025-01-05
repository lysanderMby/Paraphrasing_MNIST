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

class TransformedInterpretableClassifier(BaseInterpretableClassifier):
    """Classifier that deliberately transforms intermediate states to be unlike inputs"""
    def __init__(self, layer_configs, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.transformations = nn.ModuleList()
        
        for in_channels, out_channels in layer_configs:
            # Main processing layer with frequency-space transformation
            layer = nn.Sequential(
                # Expand channels and apply spatial mixing
                nn.Conv2d(in_channels, out_channels*4, kernel_size=1),
                nn.BatchNorm2d(out_channels*4),
                nn.GELU(),  # Different activation for more non-linearity
                
                # Spatial mixing with different kernel sizes
                nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, 
                         padding=1, groups=out_channels*4),  # Depthwise conv
                nn.BatchNorm2d(out_channels*4),
                nn.GELU(),
                
                # Channel mixing and dimension reduction
                nn.Conv2d(out_channels*4, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            self.layers.append(layer)
            
            # Additional transformation to make intermediates more abstract
            transform = nn.Sequential(
                # Expand channels to ensure divisibility by 4 for pixel shuffle
                nn.Conv2d(out_channels, out_channels*4, kernel_size=1),
                nn.BatchNorm2d(out_channels*4),
                nn.GELU(),
                
                # Spatial shuffling (now guaranteed to have channel count divisible by 4)
                nn.PixelShuffle(2),  # Reorganize spatial dimensions
                
                # Restore original dimensions
                nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                         padding=1, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                
                # Phase shift in frequency domain
                nn.Conv2d(out_channels, out_channels, kernel_size=5, 
                         padding=2, groups=out_channels),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            self.transformations.append(transform)
        
        # Final classifier
        final_channels = layer_configs[-1][1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(final_channels * 7 * 7, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights to encourage transformation
        self._init_weights()
    
    def _init_weights(self):
        """Custom initialization to encourage transformation"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize convolutions to mix spatial information
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Small positive bias
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize batch norms with slight offset
                nn.init.constant_(m.weight, 1.1)  # Slightly above 1
                nn.init.constant_(m.bias, 0.1)  # Small positive bias

    def forward(self, x, return_intermediates=False):
        intermediates = []
        current = x
        
        for layer, transform in zip(self.layers, self.transformations):
            # Apply main processing
            current = layer(current)
            # Apply additional transformation
            current = transform(current)
            intermediates.append(current)
        
        logits = self.classifier(current)
        
        if return_intermediates:
            return logits, intermediates
        return logits

# Dictionary mapping model names to their classes
MODEL_ARCHITECTURES = {
    'default': DefaultInterpretableClassifier,
    'simple': SimpleInterpretableClassifier,
    'transformed': TransformedInterpretableClassifier
} 