"""
Baseline Comparator - Simple Linear Transformation

Provides a trivial baseline for comparison with the QuICSE engine.
This is a simple linear transformation that serves as a control.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class BaselineState:
    """
    Simple state representation for baseline comparator.

    Attributes:
        features: Real-valued feature vector
    """
    features: np.ndarray

    def probability_distribution(self) -> np.ndarray:
        """Return softmax probability distribution over features."""
        exp_features = np.exp(self.features - np.max(self.features))  # For numerical stability
        return exp_features / np.sum(exp_features)


class BaselineModule(nn.Module):
    """
    Baseline comparator - simple linear transformation.

    This provides a trivial baseline for comparison with QuICSE.
    It's a single linear layer with ReLU activation.
    """

    def __init__(self, hidden_dim: int = 64, output_dim: int = 16):
        """
        Initialize baseline module.

        Args:
            hidden_dim: Dimension of hidden representations
            output_dim: Output dimension (matched to QuICSE cognitive states)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Simple linear transformation
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        # Output projection
        self.output_projection = nn.Linear(output_dim, hidden_dim)

        # Transformation layer for baseline state (output_dim -> output_dim)
        self.transformation = nn.Linear(output_dim, output_dim)

    def encode(self, hidden_states: torch.Tensor) -> BaselineState:
        """
        Encode hidden states to baseline state.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            BaselineState with features derived from input
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Average pooling over sequence
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)

        # Linear transformation
        transformed = self.linear(pooled)  # (batch, output_dim)
        activated = self.relu(transformed)

        # For MVP, return baseline state for first batch element
        features_numpy = activated[0].detach().numpy()

        return BaselineState(features=features_numpy)

    def apply_transformation(self, state: BaselineState) -> BaselineState:
        """
        Apply simple linear transformation (baseline equivalent of attention).

        Args:
            state: Input baseline state

        Returns:
            Transformed baseline state
        """
        features_tensor = torch.tensor(state.features, dtype=torch.float32).unsqueeze(0)  # (1, output_dim)

        # Apply transformation layer (output_dim -> output_dim)
        transformed = self.transformation(features_tensor)
        activated = self.relu(transformed)

        return BaselineState(features=activated[0].detach().numpy())

    def measure(self, state: BaselineState) -> int:
        """
        Measure baseline state to produce inference result.

        Args:
            state: Baseline state to measure

        Returns:
            Index of measured outcome (argmax over probability distribution)
        """
        probs = state.probability_distribution()
        return int(np.argmax(probs))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through baseline module.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
            attention_mask: Optional attention mask (not used in baseline)

        Returns:
            Transformed hidden states of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Encode to baseline state
        baseline_state = self.encode(hidden_states)

        # Apply transformation
        transformed_state = self.apply_transformation(baseline_state)

        # Project back to hidden dimension
        probs = torch.tensor(transformed_state.probability_distribution(), dtype=torch.float32).unsqueeze(0)
        output = self.output_projection(probs)  # (1, hidden_dim)

        # Broadcast to match input shape
        output_expanded = output.unsqueeze(0).expand(batch_size, seq_len, -1)

        return output_expanded

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_memory_footprint(self, batch_size: int = 1, seq_len: int = 512) -> dict:
        """
        Estimate memory footprint for given batch and sequence length.

        Returns:
            Dictionary with memory metrics in bytes
        """
        # Parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters())

        # Activation memory (rough estimate)
        # Baseline state: batch * output_dim * 8 bytes
        state_memory = batch_size * self.output_dim * 8

        # Hidden states: batch * seq_len * hidden_dim * 4 bytes (float32)
        hidden_memory = batch_size * seq_len * self.hidden_dim * 4

        return {
            'parameter_bytes': param_memory,
            'parameter_mb': param_memory / (1024 ** 2),
            'activation_bytes': state_memory + hidden_memory,
            'activation_mb': (state_memory + hidden_memory) / (1024 ** 2),
            'total_bytes': param_memory + state_memory + hidden_memory,
            'total_mb': (param_memory + state_memory + hidden_memory) / (1024 ** 2)
        }


def create_baseline_model(
    hidden_dim: int = 64,
    output_dim: int = 16
) -> BaselineModule:
    """
    Factory function to create a baseline model with default configuration.

    Args:
        hidden_dim: Hidden dimension
        output_dim: Output dimension (should match QuICSE cognitive states for fair comparison)

    Returns:
        Initialized BaselineModule
    """
    return BaselineModule(
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )


if __name__ == "__main__":
    # Simple test to verify the baseline module works
    print("Testing Baseline Comparator...")

    # Create model
    model = create_baseline_model(hidden_dim=64, output_dim=16)

    # Print parameter count
    param_count = model.get_parameter_count()
    print(f"Parameter count: {param_count:,}")

    # Print memory footprint
    footprint = model.get_memory_footprint(batch_size=1, seq_len=128)
    print("Memory footprint (batch=1, seq_len=128):")
    print(f"  Parameters: {footprint['parameter_mb']:.3f} MB")
    print(f"  Activations: {footprint['activation_mb']:.3f} MB")
    print(f"  Total: {footprint['total_mb']:.3f} MB")

    # Create dummy input
    batch_size = 2
    seq_len = 32
    hidden_dim = 64
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

    # Forward pass
    output = model(hidden_states)
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")

    # Test baseline state operations
    baseline_state = model.encode(hidden_states)
    print(f"Baseline state features shape: {baseline_state.features.shape}")

    # Test transformation
    transformed = model.apply_transformation(baseline_state)
    print(f"Transformed state probabilities sum: {np.sum(transformed.probability_distribution()):.6f}")

    # Test measurement
    result = model.measure(transformed)
    print(f"Measurement result: {result}")

    print("\nBaseline Comparator test completed successfully!")
