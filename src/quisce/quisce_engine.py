"""
QuICSE Engine - Quantum-Infused Cognitive Synthesis Engine

Minimal viable prototype implementing the core inference mechanism.
Based on docs/architecture/quisce_engine_specification.md
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SpectralContext:
    """
    Context provided by the spectral behavioral system.

    Attributes:
        eigenvalues: Graph Laplacian eigenvalues
        eigenvectors: Graph Laplacian eigenvectors
        fiedler_vector: Second eigenvector for bipartitioning
        spectral_clusters: Cluster assignments from spectral clustering
        cheeger_constant: Graph expansion measure
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    fiedler_vector: np.ndarray | None = None
    spectral_clusters: np.ndarray | None = None
    cheeger_constant: float | None = None


@dataclass
class CognitiveState:
    """
    Represents a cognitive state in the Hilbert space.

    Attributes:
        amplitudes: Complex amplitude coefficients (alpha_i)
        basis_states: Number of basis cognitive states (n)
    """
    amplitudes: np.ndarray  # Complex coefficients
    basis_states: int

    def __post_init__(self):
        # Ensure normalization: sum of |alpha_i|^2 = 1
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 0:
            self.amplitudes = self.amplitudes / norm

    def probability_distribution(self) -> np.ndarray:
        """Return probability distribution P(k) = |alpha_k|^2"""
        return np.abs(self.amplitudes) ** 2


class QuICSEModule(nn.Module):
    """
    QuICSE engine module - minimal viable prototype.

    Implements:
    1. State Encoder: Basic token-to-state mapping
    2. Single Attention Layer: One QuICSE attention head
    3. Simple Measurement: Argmax over state amplitudes
    4. Spectral Context Integration: Accept precomputed spectral features
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_cognitive_states: int = 16,
        attention_heads: int = 1,
        spectral_dim: int | None = None
    ):
        """
        Initialize QuICSE module.

        Args:
            hidden_dim: Dimension of hidden representations
            num_cognitive_states: Number of basis cognitive states
            attention_heads: Number of attention heads (default 1 for MVP)
            spectral_dim: Dimension for spectral decomposition (default: hidden_dim)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_cognitive_states = num_cognitive_states
        self.attention_heads = attention_heads
        self.spectral_dim = spectral_dim if spectral_dim else hidden_dim

        # State encoder: maps hidden states to cognitive state amplitudes
        self.state_encoder = nn.Linear(hidden_dim, num_cognitive_states * 2)  # *2 for complex (real, imag)

        # Attention operator parameter (Hermitian matrix parameterization)
        # H is Hermitian, so we parameterize upper triangle and diagonal
        self.attention_param = nn.Parameter(torch.randn(num_cognitive_states, num_cognitive_states))

        # Synthesis weights
        self.synthesis_weights = nn.Linear(num_cognitive_states, num_cognitive_states)

        # Output projection back to hidden dim
        self.output_projection = nn.Linear(num_cognitive_states, hidden_dim)

        # Spectral context projection (if provided)
        self.spectral_projection = nn.Linear(self.spectral_dim, num_cognitive_states) if self.spectral_dim != hidden_dim else None

    def _make_hermitian(self, param: torch.Tensor) -> torch.Tensor:
        """Convert parameter matrix to Hermitian matrix: H = (A + A†) / 2"""
        return (param + param.t()) / 2

    def _unitary_from_hermitian(self, H: torch.Tensor, theta: float = 1.0) -> torch.Tensor:
        """
        Create unitary operator from Hermitian matrix: U = exp(-i * theta * H)
        Uses eigendecomposition for matrix exponential.
        """
        # Eigendecomposition of Hermitian matrix
        eigenvalues, eigenvectors = torch.linalg.eigh(H)

        # U = V * diag(exp(-i * theta * lambda)) * V†
        # Convert eigenvalues to complex for the exponential
        exp_eigenvalues = torch.exp(-1j * theta * eigenvalues.to(torch.complex64))
        diag_exp = torch.diag(exp_eigenvalues)

        # Convert eigenvectors to complex for matrix multiplication
        eigenvectors_complex = eigenvectors.to(torch.complex64)
        U = eigenvectors_complex @ diag_exp @ eigenvectors_complex.conj().t()
        return U

    def encode(self, hidden_states: torch.Tensor) -> CognitiveState:
        """
        Encode hidden states to cognitive state.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            CognitiveState with amplitudes derived from input
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Average pooling over sequence to get batch-level representation
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)

        # Encode to cognitive state amplitudes
        encoded = self.state_encoder(pooled)  # (batch, num_cognitive_states * 2)

        # Split into real and imaginary parts
        real_part = encoded[:, :self.num_cognitive_states]
        imag_part = encoded[:, self.num_cognitive_states:]

        # Create complex amplitudes
        amplitudes = real_part + 1j * imag_part  # (batch, num_cognitive_states)

        # Normalize each batch element
        norms = torch.sqrt(torch.sum(torch.abs(amplitudes) ** 2, dim=1, keepdim=True))
        amplitudes = amplitudes / (norms + 1e-8)

        # For MVP, return cognitive state for first batch element
        # In full implementation, would handle batch properly
        amp_numpy = amplitudes[0].detach().numpy()

        return CognitiveState(amplitudes=amp_numpy, basis_states=self.num_cognitive_states)

    def apply_attention(self, state: CognitiveState, spectral_context: SpectralContext | None = None) -> CognitiveState:
        """
        Apply attention as unitary rotation in cognitive state space.

        Args:
            state: Input cognitive state
            spectral_context: Optional spectral context from spectral behavioral system

        Returns:
            Transformed cognitive state
        """
        # Create Hermitian operator from parameters
        H = self._make_hermitian(self.attention_param)

        # If spectral context provided, modulate attention
        if spectral_context is not None and spectral_context.eigenvalues is not None:
            # Use eigenvalues to modulate the Hermitian operator
            eigenvalues = torch.tensor(spectral_context.eigenvalues[:self.num_cognitive_states], dtype=torch.float32)
            if len(eigenvalues) < self.num_cognitive_states:
                # Pad with zeros if needed
                eigenvalues = torch.cat([eigenvalues, torch.zeros(self.num_cognitive_states - len(eigenvalues))])
            # Modulate diagonal of H
            H = H + torch.diag(eigenvalues) * 0.1

        # Create unitary attention operator
        U = self._unitary_from_hermitian(H, theta=1.0)

        # Apply unitary transformation: |psi'> = U |psi>
        state_vector = torch.tensor(state.amplitudes, dtype=torch.complex64)
        transformed = U @ state_vector

        # Normalize
        transformed = transformed / (torch.norm(transformed) + 1e-8)

        return CognitiveState(amplitudes=transformed.detach().numpy(), basis_states=self.num_cognitive_states)

    def synthesize(self, states: list[CognitiveState]) -> CognitiveState:
        """
        Combine multiple cognitive states through coherent superposition.

        Args:
            states: List of cognitive states to combine

        Returns:
            Synthesized cognitive state
        """
        if not states:
            raise ValueError("Cannot synthesize empty list of states")

        # Stack amplitudes
        stacked = np.stack([s.amplitudes for s in states], axis=0)  # (num_states, num_cognitive_states)

        # Apply learned weights
        weights_tensor = torch.tensor(stacked, dtype=torch.complex64)
        weighted = self.synthesis_weights(weights_tensor.real) + 1j * self.synthesis_weights(weights_tensor.imag)

        # Sum and normalize
        synthesized = torch.sum(weighted, dim=0)
        synthesized = synthesized / (torch.norm(synthesized) + 1e-8)

        return CognitiveState(amplitudes=synthesized.detach().numpy(), basis_states=self.num_cognitive_states)

    def measure(self, state: CognitiveState) -> int:
        """
        Measure cognitive state to produce inference result.

        Args:
            state: Cognitive state to measure

        Returns:
            Index of measured outcome (argmax over probability distribution)
        """
        probs = state.probability_distribution()
        return int(np.argmax(probs))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        spectral_context: SpectralContext | None = None
    ) -> torch.Tensor:
        """
        Forward pass through QuICSE engine.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
            attention_mask: Optional attention mask (not used in MVP)
            spectral_context: Optional spectral context from spectral behavioral system

        Returns:
            Transformed hidden states of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Encode to cognitive state
        cognitive_state = self.encode(hidden_states)

        # Apply attention
        attended_state = self.apply_attention(cognitive_state, spectral_context)

        # Project back to hidden dimension
        # Use probability distribution as features
        probs = torch.tensor(attended_state.probability_distribution(), dtype=torch.float32)
        output = self.output_projection(probs)  # (hidden_dim,)

        # Broadcast to match input shape
        output_expanded = output.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

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
        # Cognitive state: batch * num_cognitive_states * 2 (complex) * 8 bytes
        state_memory = batch_size * self.num_cognitive_states * 2 * 8

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


def create_quisce_model(
    hidden_dim: int = 64,
    num_cognitive_states: int = 16
) -> QuICSEModule:
    """
    Factory function to create a QuICSE model with default configuration.

    Args:
        hidden_dim: Hidden dimension
        num_cognitive_states: Number of cognitive basis states

    Returns:
        Initialized QuICSEModule
    """
    return QuICSEModule(
        hidden_dim=hidden_dim,
        num_cognitive_states=num_cognitive_states,
        attention_heads=1
    )


if __name__ == "__main__":
    # Simple test to verify the module works
    print("Testing QuICSE Engine MVP...")

    # Create model
    model = create_quisce_model(hidden_dim=64, num_cognitive_states=16)

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

    # Test cognitive state operations
    cognitive_state = model.encode(hidden_states)
    print(f"Cognitive state basis: {cognitive_state.basis_states}")
    print(f"Cognitive state amplitudes shape: {cognitive_state.amplitudes.shape}")

    # Test attention
    attended = model.apply_attention(cognitive_state)
    print(f"Attended state probabilities sum: {np.sum(attended.probability_distribution()):.6f}")

    # Test measurement
    result = model.measure(attended)
    print(f"Measurement result: {result}")

    # Test synthesis with multiple states
    state1 = CognitiveState(amplitudes=np.random.randn(16) + 1j * np.random.randn(16), basis_states=16)
    state2 = CognitiveState(amplitudes=np.random.randn(16) + 1j * np.random.randn(16), basis_states=16)
    synthesized = model.synthesize([state1, state2])
    print(f"Synthesized state probabilities sum: {np.sum(synthesized.probability_distribution()):.6f}")

    print("\nQuICSE Engine MVP test completed successfully!")
