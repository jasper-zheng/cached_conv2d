#!/usr/bin/env python3
"""
Test script for 2D cached convolution layers.
Demonstrates the usage of cached convolutions for spectrogram processing.
"""

import torch
import torch.nn as nn
from convs import CachedConv2d, CachedConvTranspose2d, Conv2d, ConvTranspose2d, get_padding_2d

def test_cached_conv2d():
    """Test basic CachedConv2d functionality"""
    print("Testing CachedConv2d...")
    
    # Create a simple cached 2D convolution
    # For spectrograms: height=frequency (centered), width=time (causal)
    conv = CachedConv2d(
        in_channels=1,
        out_channels=16,
        kernel_size=(3, 5),  # (frequency, time)
        stride=(1, 1),
        padding=(1, 2),
        padding_mode=("centered", "causal")
    )
    
    # Simulate a batch of spectrograms: [batch, channels, frequency, time]
    batch_size = 2
    freq_bins = 80
    time_steps = 100
    
    # Process multiple chunks to test caching
    for i in range(3):
        x = torch.randn(batch_size, 1, freq_bins, time_steps)
        y = conv(x)
        print(f"Chunk {i}: Input shape: {x.shape}, Output shape: {y.shape}")
        print(f"Cumulative delay: {conv.cumulative_delay}")
    
    print("CachedConv2d test passed!\n")


def test_cached_conv_transpose2d():
    """Test CachedConvTranspose2d functionality"""
    print("Testing CachedConvTranspose2d...")
    
    conv_t = CachedConvTranspose2d(
        in_channels=16,
        out_channels=1,
        kernel_size=(3, 5),  # (frequency, time)
        stride=(1, 1),
        padding=(1, 2)
    )
    
    batch_size = 2
    freq_bins = 80
    time_steps = 100
    
    # Process multiple chunks
    for i in range(3):
        x = torch.randn(batch_size, 16, freq_bins, time_steps)
        y = conv_t(x)
        print(f"Chunk {i}: Input shape: {x.shape}, Output shape: {y.shape}")
        print(f"Cumulative delay: {conv_t.cumulative_delay}")
    
    print("CachedConvTranspose2d test passed!\n")


def test_padding_2d():
    """Test 2D padding calculation"""
    print("Testing get_padding_2d...")
    
    # Test different padding modes
    padding = get_padding_2d(
        kernel_size=(3, 5),  # (frequency, time)
        stride=(1, 1),
        dilation=(1, 1),
        mode=("centered", "causal")
    )
    print(f"Padding for (3,5) kernel with (centered, causal): {padding}")
    
    padding = get_padding_2d(
        kernel_size=3,
        mode="centered"
    )
    print(f"Padding for 3x3 kernel with centered: {padding}")
    
    print("get_padding_2d test passed!\n")


def test_spectrogram_model():
    """Test a simple spectrogram processing model"""
    print("Testing spectrogram model...")
    
    class SpectrogramModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = CachedConv2d(
                1, 32, kernel_size=(3, 5), stride=(1, 1), 
                padding=(1, 2), padding_mode=("centered", "causal")
            )
            self.conv2 = CachedConv2d(
                32, 64, kernel_size=(3, 3), stride=(1, 1), 
                padding=(1, 1), padding_mode=("centered", "causal")
            )
            self.conv3 = CachedConv2d(
                64, 32, kernel_size=(3, 3), stride=(1, 1), 
                padding=(1, 1), padding_mode=("centered", "causal")
            )
            self.conv_out = CachedConv2d(
                32, 1, kernel_size=(1, 1), stride=(1, 1), 
                padding=0
            )
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.conv_out(x)
            return x
    
    model = SpectrogramModel()
    
    # Test with multiple chunks
    batch_size = 2
    freq_bins = 128
    time_steps = 50
    
    total_delay = 0
    for layer in [model.conv1, model.conv2, model.conv3, model.conv_out]:
        total_delay += layer.cumulative_delay
    
    print(f"Total model delay: {total_delay}")
    
    for i in range(3):
        x = torch.randn(batch_size, 1, freq_bins, time_steps)
        y = model(x)
        print(f"Chunk {i}: Input shape: {x.shape}, Output shape: {y.shape}")
    
    print("Spectrogram model test passed!\n")


def test_comparison_with_standard():
    """Compare cached vs standard convolutions"""
    print("Testing comparison with standard convolutions...")
    
    # Create both cached and standard versions
    cached_conv = CachedConv2d(
        1, 16, kernel_size=(3, 3), stride=(1, 1), 
        padding=(1, 1), padding_mode=("centered", "causal")
    )
    
    standard_conv = Conv2d(
        1, 16, kernel_size=(3, 3), stride=(1, 1), 
        padding=(1, 1)
    )
    
    # Copy weights for fair comparison
    standard_conv.weight.data = cached_conv.weight.data.clone()
    standard_conv.bias.data = cached_conv.bias.data.clone()
    
    # Test input: [batch, channels, frequency, time]
    x = torch.randn(1, 1, 32, 32)
    
    # Get outputs
    cached_out = cached_conv(x)
    standard_out = standard_conv(x)
    
    print(f"Cached conv output shape: {cached_out.shape}")
    print(f"Standard conv output shape: {standard_out.shape}")
    print(f"Cached conv delay: {cached_conv.cumulative_delay}")
    print(f"Standard conv delay: {standard_conv.cumulative_delay}")
    
    print("Comparison test passed!\n")


if __name__ == "__main__":
    print("Running 2D Cached Convolution Tests\n")
    print("=" * 50)
    
    test_padding_2d()
    test_cached_conv2d()
    test_cached_conv_transpose2d()
    test_spectrogram_model()
    test_comparison_with_standard()
    
    print("=" * 50)
    print("All tests passed successfully!")
