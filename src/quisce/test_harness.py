"""
Test Harness for QuICSE Engine vs Baseline Comparator

Runs both components on identical input and outputs comparison metrics.
"""

import time
from typing import Any

import numpy as np
import torch

from quisce.baseline import create_baseline_model
from quisce.quisce_engine import (
    CognitiveState,
    QuICSEModule,
    SpectralContext,
    create_quisce_model,
)


def generate_test_input(
    batch_size: int = 1,
    seq_len: int = 32,
    hidden_dim: int = 64,
    seed: int = 42
) -> torch.Tensor:
    """
    Generate reproducible test input.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        seed: Random seed for reproducibility

    Returns:
        Random tensor of shape (batch_size, seq_len, hidden_dim)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.randn(batch_size, seq_len, hidden_dim)


def generate_spectral_context(
    num_cognitive_states: int = 16,
    seed: int = 42
) -> SpectralContext:
    """
    Generate synthetic spectral context for testing.

    Args:
        num_cognitive_states: Number of cognitive states (determines eigenvalue count)
        seed: Random seed for reproducibility

    Returns:
        SpectralContext with synthetic data
    """
    np.random.seed(seed)

    # Generate synthetic eigenvalues (sorted, positive for Laplacian)
    eigenvalues = np.sort(np.abs(np.random.randn(num_cognitive_states * 2)))

    # Generate synthetic eigenvectors (orthonormal)
    random_matrix = np.random.randn(num_cognitive_states * 2, num_cognitive_states * 2)
    Q, R = np.linalg.qr(random_matrix)
    eigenvectors = Q

    # Fiedler vector (second eigenvector)
    fiedler_vector = eigenvectors[:, 1]

    # Synthetic spectral clusters
    spectral_clusters = np.random.randint(0, 3, size=num_cognitive_states * 2)

    # Synthetic Cheeger constant
    cheeger_constant = np.random.uniform(0.1, 0.5)

    return SpectralContext(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        fiedler_vector=fiedler_vector,
        spectral_clusters=spectral_clusters,
        cheeger_constant=cheeger_constant
    )


def measure_inference_time(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    spectral_context: SpectralContext = None,
    num_runs: int = 10
) -> dict[str, float]:
    """
    Measure inference time for a model.

    Args:
        model: Model to test
        input_tensor: Input tensor
        spectral_context: Optional spectral context
        num_runs: Number of runs for averaging

    Returns:
        Dictionary with timing metrics
    """
    times = []

    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            if isinstance(model, QuICSEModule):
                _ = model(input_tensor, spectral_context=spectral_context)
            else:
                _ = model(input_tensor)
        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000
    }


def measure_state_coherence(state: CognitiveState) -> float:
    """
    Measure cognitive coherence - novelty metric for QuICSE.

    This measures how "spread out" the probability distribution is,
    which indicates superposition quality.

    Args:
        state: CognitiveState to measure

    Returns:
        Coherence score (higher = more coherent superposition)
    """
    probs = state.probability_distribution()

    # Entropy-based coherence measure
    # Higher entropy = more uniform distribution = better superposition
    probs_clean = probs + 1e-10  # Avoid log(0)
    entropy = -np.sum(probs_clean * np.log(probs_clean))

    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(len(probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return normalized_entropy


def run_comparison(
    hidden_dim: int = 64,
    num_cognitive_states: int = 16,
    batch_size: int = 1,
    seq_len: int = 32,
    num_runs: int = 10
) -> dict[str, Any]:
    """
    Run full comparison between QuICSE and baseline.

    Args:
        hidden_dim: Hidden dimension
        num_cognitive_states: Number of cognitive states
        batch_size: Batch size
        seq_len: Sequence length
        num_runs: Number of runs for timing

    Returns:
        Dictionary with all comparison metrics
    """
    print("=" * 60)
    print("QuICSE Engine vs Baseline Comparison")
    print("=" * 60)
    print()

    # Create models
    print("Initializing models...")
    quicse_model = create_quisce_model(hidden_dim=hidden_dim, num_cognitive_states=num_cognitive_states)
    baseline_model = create_baseline_model(hidden_dim=hidden_dim, output_dim=num_cognitive_states)

    # Generate test input
    print("Generating test input...")
    test_input = generate_test_input(batch_size=batch_size, seq_len=seq_len, hidden_dim=hidden_dim)

    # Generate spectral context
    print("Generating spectral context...")
    spectral_context = generate_spectral_context(num_cognitive_states=num_cognitive_states)

    results = {}

    # Parameter count comparison
    print("\n--- Parameter Count ---")
    quicse_params = quicse_model.get_parameter_count()
    baseline_params = baseline_model.get_parameter_count()
    print(f"QuICSE:     {quicse_params:,} parameters")
    print(f"Baseline:   {baseline_params:,} parameters")
    print(f"Difference: {quicse_params - baseline_params:+,} ({(quicse_params/baseline_params - 1)*100:+.1f}%)")

    results['parameter_count'] = {
        'quicse': quicse_params,
        'baseline': baseline_params,
        'difference': quicse_params - baseline_params,
        'ratio': quicse_params / baseline_params
    }

    # Memory footprint comparison
    print("\n--- Memory Footprint (batch=1, seq_len=128) ---")
    quicse_footprint = quicse_model.get_memory_footprint(batch_size=1, seq_len=128)
    baseline_footprint = baseline_model.get_memory_footprint(batch_size=1, seq_len=128)
    print(f"QuICSE:     {quicse_footprint['total_mb']:.3f} MB total")
    print(f"Baseline:   {baseline_footprint['total_mb']:.3f} MB total")
    print(f"Difference: {quicse_footprint['total_mb'] - baseline_footprint['total_mb']:+.3f} MB")

    results['memory_footprint'] = {
        'quicse': quicse_footprint,
        'baseline': baseline_footprint
    }

    # Inference time comparison
    print("\n--- Inference Time ---")
    quicse_timing = measure_inference_time(quicse_model, test_input, spectral_context, num_runs=num_runs)
    baseline_timing = measure_inference_time(baseline_model, test_input, None, num_runs=num_runs)
    print(f"QuICSE:     {quicse_timing['mean_time_ms']:.3f} ± {quicse_timing['std_time_ms']:.3f} ms")
    print(f"Baseline:   {baseline_timing['mean_time_ms']:.3f} ± {baseline_timing['std_time_ms']:.3f} ms")
    print(f"Difference: {quicse_timing['mean_time_ms'] - baseline_timing['mean_time_ms']:+.3f} ms")

    results['inference_time'] = {
        'quicse': quicse_timing,
        'baseline': baseline_timing
    }

    # Forward pass output comparison
    print("\n--- Forward Pass Output ---")
    with torch.no_grad():
        quicse_output = quicse_model(test_input, spectral_context=spectral_context)
        baseline_output = baseline_model(test_input)

    print(f"QuICSE output shape:   {quicse_output.shape}")
    print(f"Baseline output shape: {baseline_output.shape}")
    print(f"QuICSE output mean:    {quicse_output.mean().item():.6f}")
    print(f"Baseline output mean:  {baseline_output.mean().item():.6f}")

    results['forward_output'] = {
        'quicse_shape': list(quicse_output.shape),
        'baseline_shape': list(baseline_output.shape),
        'quicse_mean': float(quicse_output.mean().item()),
        'baseline_mean': float(baseline_output.mean().item())
    }

    # Cognitive state analysis (QuICSE only)
    print("\n--- Cognitive State Analysis (QuICSE) ---")
    quicse_state = quicse_model.encode(test_input)
    attended_state = quicse_model.apply_attention(quicse_state, spectral_context)

    print(f"Initial state coherence:    {measure_state_coherence(quicse_state):.4f}")
    print(f"Attended state coherence:   {measure_state_coherence(attended_state):.4f}")
    print(f"Measurement result:         {quicse_model.measure(attended_state)}")
    print(f"Probability distribution:   {attended_state.probability_distribution()}")

    results['cognitive_state'] = {
        'initial_coherence': measure_state_coherence(quicse_state),
        'attended_coherence': measure_state_coherence(attended_state),
        'measurement_result': quicse_model.measure(attended_state),
        'probability_distribution': attended_state.probability_distribution().tolist()
    }

    # Baseline state analysis
    print("\n--- Baseline State Analysis ---")
    baseline_state = baseline_model.encode(test_input)
    transformed_state = baseline_model.apply_transformation(baseline_state)

    print(f"Initial state entropy:      {-np.sum(baseline_state.probability_distribution() * np.log(baseline_state.probability_distribution() + 1e-10)):.4f}")
    print(f"Transformed state entropy:  {-np.sum(transformed_state.probability_distribution() * np.log(transformed_state.probability_distribution() + 1e-10)):.4f}")
    print(f"Measurement result:         {baseline_model.measure(transformed_state)}")

    results['baseline_state'] = {
        'measurement_result': baseline_model.measure(transformed_state)
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"QuICSE parameters: {quicse_params:,} vs Baseline: {baseline_params:,}")
    print(f"QuICSE memory:     {quicse_footprint['total_mb']:.3f} MB vs Baseline: {baseline_footprint['total_mb']:.3f} MB")
    print(f"QuICSE latency:    {quicse_timing['mean_time_ms']:.3f} ms vs Baseline: {baseline_timing['mean_time_ms']:.3f} ms")
    print(f"QuICSE coherence:  {measure_state_coherence(attended_state):.4f} (novel metric)")
    print()
    print("Note: This is a minimal viable prototype. Performance characteristics")
    print("will change with optimization and larger-scale testing.")
    print("=" * 60)

    return results


def main():
    """Main entry point for test harness."""
    print("QuICSE Engine Test Harness")
    print("Comparing QuICSE vs Baseline on identical inputs")
    print()

    # Run comparison with default settings
    results = run_comparison(
        hidden_dim=64,
        num_cognitive_states=16,
        batch_size=1,
        seq_len=32,
        num_runs=10
    )

    return results


if __name__ == "__main__":
    main()
