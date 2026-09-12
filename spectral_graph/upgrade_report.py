#!/usr/bin/env python3
"""
Spectral analysis report generation for dolphin model upgrades.

This module analyzes the spectral properties of a dolphin model's connectivity
graph and produces a structured report mapping spectral gaps to proposed
architectural upgrades.

The spectral gap (difference between consecutive eigenvalues) reveals:
- Large gaps: Natural cluster boundaries or bottlenecks in the graph
- Small gaps: Smooth transitions or highly connected regions
- Algebraic connectivity (λ₂): Overall connectivity strength

Based on these properties, the module suggests logical upgrades such as:
- Adding skip connections across bottlenecks
- Increasing capacity in weakly connected regions
- Splitting the model at natural community boundaries
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from spectral_graph.dolphin_model import DolphinModel, SpectralAnalyzer


@dataclass
class SpectralGap:
    """Represents a spectral gap between consecutive eigenvalues."""

    index: int  # Gap between λ_index and λ_{index+1}
    lambda_low: float  # Lower eigenvalue
    lambda_high: float  # Higher eigenvalue
    gap_size: float  # Difference (lambda_high - lambda_low)
    relative_gap: float  # Gap relative to lambda_high (if > 0)

    def is_significant(self, threshold: float = 0.5) -> bool:
        """Check if this gap is significant (larger than threshold)."""
        return self.relative_gap > threshold


@dataclass
class UpgradeRecommendation:
    """A recommended upgrade based on spectral analysis."""

    upgrade_type: str
    description: str
    justification: str
    priority: str  # "high", "medium", "low"
    affected_regions: list[int] = field(default_factory=list)
    spectral_evidence: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "upgrade_type": self.upgrade_type,
            "description": self.description,
            "justification": self.justification,
            "priority": self.priority,
            "affected_regions": self.affected_regions,
            "spectral_evidence": self.spectral_evidence,
        }


@dataclass
class SpectralReport:
    """Complete spectral analysis report with recommendations."""

    model_info: dict[str, Any]
    eigenvalues: list[float]
    significant_gaps: list[dict[str, float]]
    algebraic_connectivity: float
    condition_number: float
    recommendations: list[dict[str, Any]]
    summary: str

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to JSON."""
        return json.dumps(
            {
                "model_info": self.model_info,
                "eigenvalues": self.eigenvalues,
                "significant_gaps": self.significant_gaps,
                "algebraic_connectivity": self.algebraic_connectivity,
                "condition_number": self.condition_number,
                "recommendations": self.recommendations,
                "summary": self.summary,
            },
            indent=indent,
        )

    def print_summary(self) -> None:
        """Print a human-readable summary of the report."""
        print("=" * 70)
        print("SPECTRAL ANALYSIS REPORT")
        print("=" * 70)

        print(f"\nModel: {self.model_info.get('type', 'Unknown')}")
        print(f"Vertices: {self.model_info.get('n_vertices', 'N/A')}")
        print(f"Edges: {self.model_info.get('n_edges', 'N/A')}")

        print(f"\n--- Spectral Properties ---")
        print(f"Algebraic Connectivity (λ₂): {self.algebraic_connectivity:.6f}")
        print(f"Condition Number (λ_max/λ₂): {self.condition_number:.2f}")

        if self.significant_gaps:
            print(f"\n--- Significant Spectral Gaps ({len(self.significant_gaps)} found) ---")
            for gap in self.significant_gaps[:5]:  # Show top 5
                print(
                    f"  Gap after λ_{gap['index']}: "
                    f"{gap['lambda_low']:.4f} → {gap['lambda_high']:.4f} "
                    f"(Δ={gap['gap_size']:.4f}, relative={gap['relative_gap']:.2%})"
                )
        else:
            print("\n--- No Significant Spectral Gaps Detected ---")

        print(f"\n--- Recommendations ({len(self.recommendations)}) ---")
        for i, rec in enumerate(self.recommendations, 1):
            print(f"\n{i}. [{rec['priority'].upper()}] {rec['upgrade_type']}")
            print(f"   {rec['description']}")
            print(f"   Justification: {rec['justification']}")

        print(f"\n--- Summary ---")
        print(self.summary)
        print("=" * 70)


class UpgradeAnalyzer:
    """
    Analyzes spectral properties and generates upgrade recommendations.

    This class takes a DolphinModel or SpectralAnalyzer and produces
    a structured report mapping spectral gaps to architectural upgrades.
    """

    # Thresholds for gap significance
    GAP_THRESHOLD = 0.3  # Relative gap size considered significant
    LOW_CONNECTIVITY_THRESHOLD = 0.1  # Algebraic connectivity below this is "low"
    HIGH_CONDITION_THRESHOLD = 100.0  # Condition number above this is "high"

    def __init__(
        self,
        analyzer: Optional[SpectralAnalyzer] = None,
        model: Optional[DolphinModel] = None,
    ) -> None:
        """
        Initialize the analyzer.

        Parameters
        ----------
        analyzer : SpectralAnalyzer, optional
            Pre-existing spectral analyzer
        model : DolphinModel, optional
            Dolphin model to analyze (creates analyzer if not provided)

        Raises
        ------
        ValueError
            If neither analyzer nor model is provided
        """
        if analyzer is not None:
            self.analyzer = analyzer
        elif model is not None:
            self.analyzer = SpectralAnalyzer.from_dolphin_model(model)
            self.model = model
        else:
            raise ValueError("Must provide either analyzer or model")

        self._eigenvalues: Optional[np.ndarray] = None
        self._gaps: list[SpectralGap] = []

    def _compute_gaps(self) -> list[SpectralGap]:
        """Compute all spectral gaps from eigenvalues."""
        if self._eigenvalues is None:
            self._eigenvalues = self.analyzer.get_sorted_eigenvalues()

        gaps = []
        eigenvalues = self._eigenvalues

        for i in range(len(eigenvalues) - 1):
            lambda_low = eigenvalues[i]
            lambda_high = eigenvalues[i + 1]
            gap_size = lambda_high - lambda_low

            # Relative gap (avoid division by zero)
            if lambda_high > 1e-10:
                relative_gap = gap_size / lambda_high
            else:
                relative_gap = 0.0

            gaps.append(
                SpectralGap(
                    index=i + 1,  # Gap after λ_i (1-indexed)
                    lambda_low=lambda_low,
                    lambda_high=lambda_high,
                    gap_size=gap_size,
                    relative_gap=relative_gap,
                )
            )

        self._gaps = gaps
        return gaps

    def _identify_significant_gaps(
        self, threshold: float = GAP_THRESHOLD
    ) -> list[SpectralGap]:
        """Identify gaps larger than the threshold."""
        if not self._gaps:
            self._compute_gaps()

        return [gap for gap in self._gaps if gap.is_significant(threshold)]

    def _generate_recommendations(self) -> list[UpgradeRecommendation]:
        """Generate upgrade recommendations based on spectral analysis."""
        recommendations = []

        if self._eigenvalues is None:
            self._eigenvalues = self.analyzer.get_sorted_eigenvalues()

        if not self._gaps:
            self._compute_gaps()

        eigenvalues = self._eigenvalues
        n_vertices = len(eigenvalues)

        # Get key metrics
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        lambda_max = eigenvalues[-1] if len(eigenvalues) > 0 else 0.0
        condition_number = lambda_max / lambda_2 if lambda_2 > 1e-10 else float("inf")

        # Recommendation 1: Low algebraic connectivity
        if lambda_2 < self.LOW_CONNECTIVITY_THRESHOLD:
            recommendations.append(
                UpgradeRecommendation(
                    upgrade_type="Add Skip Connections",
                    description=(
                        "Introduce long-range connections or skip layers to improve "
                        "information flow across the network"
                    ),
                    justification=(
                        f"Low algebraic connectivity (λ₂={lambda_2:.6f}) indicates "
                        "the presence of bottlenecks. The graph has weak connections "
                        "between some regions, which may cause vanishing gradients "
                        "or slow convergence."
                    ),
                    priority="high",
                    spectral_evidence={
                        "algebraic_connectivity": float(lambda_2),
                        "threshold": self.LOW_CONNECTIVITY_THRESHOLD,
                    },
                )
            )

        # Recommendation 2: High condition number
        if condition_number > self.HIGH_CONDITION_THRESHOLD:
            recommendations.append(
                UpgradeRecommendation(
                    upgrade_type="Normalize Layer Interactions",
                    description=(
                        "Apply batch normalization or layer normalization to stabilize "
                        "the spectrum and improve training dynamics"
                    ),
                    justification=(
                        f"High condition number ({condition_number:.2f}) suggests "
                        "the Laplacian spectrum spans many orders of magnitude. "
                        "This can lead to numerical instability and slow convergence."
                    ),
                    priority="high",
                    spectral_evidence={
                        "condition_number": float(condition_number),
                        "lambda_2": float(lambda_2),
                        "lambda_max": float(lambda_max),
                    },
                )
            )

        # Recommendation 3: Significant spectral gaps (community structure)
        significant_gaps = self._identify_significant_gaps()
        for gap in significant_gaps[:3]:  # Top 3 gaps
            if gap.index == 1:
                # Gap after first eigenvalue - indicates disconnected components
                recommendations.append(
                    UpgradeRecommendation(
                        upgrade_type="Merge Disconnected Components",
                        description=(
                            "The model appears to have nearly disconnected subgraphs. "
                            "Consider merging these components or adding bridging connections"
                        ),
                        justification=(
                            f"Large gap after λ₁ (Δ={gap.gap_size:.4f}, "
                            f"relative={gap.relative_gap:.2%}) suggests the graph "
                            "may be nearly disconnected. This could indicate "
                            "isolated feature clusters that don't interact."
                        ),
                        priority="high",
                        spectral_evidence={
                            "gap_index": gap.index,
                            "gap_size": float(gap.gap_size),
                            "relative_gap": float(gap.relative_gap),
                        },
                    )
                )
            elif gap.index <= 3:
                # Early gaps indicate major community structure
                k_clusters = gap.index
                recommendations.append(
                    UpgradeRecommendation(
                        upgrade_type=f"Split into {k_clusters} Parallel Branches",
                        description=(
                            f"The spectral gap after λ_{gap.index} suggests {k_clusters} "
                            "natural communities. Consider restructuring the model "
                            "with parallel processing branches for each community"
                        ),
                        justification=(
                            f"Significant spectral gap (Δ={gap.gap_size:.4f}, "
                            f"relative={gap.relative_gap:.2%}) indicates natural "
                            "cluster boundaries. Processing these clusters separately "
                            "may improve efficiency and interpretability."
                        ),
                        priority="medium",
                        affected_regions=list(range(k_clusters)),
                        spectral_evidence={
                            "gap_index": gap.index,
                            "gap_size": float(gap.gap_size),
                            "relative_gap": float(gap.relative_gap),
                            "suggested_clusters": k_clusters,
                        },
                    )
                )
            else:
                # Later gaps indicate finer structure
                recommendations.append(
                    UpgradeRecommendation(
                        upgrade_type="Refine Feature Granularity",
                        description=(
                            "Add intermediate layers or increase feature dimension "
                            "to capture finer-grained patterns revealed by spectral analysis"
                        ),
                        justification=(
                            f"Spectral gap after λ_{gap.index} (Δ={gap.gap_size:.4f}) "
                            "reveals substructure within communities. This suggests "
                            "the current architecture may be missing intermediate-scale features."
                        ),
                        priority="low",
                        spectral_evidence={
                            "gap_index": gap.index,
                            "gap_size": float(gap.gap_size),
                            "relative_gap": float(gap.relative_gap),
                        },
                    )
                )

        # Recommendation 4: Dense spectrum (no significant gaps)
        if not significant_gaps and lambda_2 > self.LOW_CONNECTIVITY_THRESHOLD:
            recommendations.append(
                UpgradeRecommendation(
                    upgrade_type="Maintain Current Architecture",
                    description=(
                        "The spectrum shows no major bottlenecks or community structure. "
                        "The current connectivity pattern is well-balanced."
                    ),
                    justification=(
                        f"Uniform spectral distribution (λ₂={lambda_2:.4f}, "
                        f"no significant gaps) indicates good connectivity throughout. "
                        "No structural changes are strongly indicated by spectral analysis."
                    ),
                    priority="low",
                    spectral_evidence={
                        "algebraic_connectivity": float(lambda_2),
                        "significant_gaps_found": 0,
                    },
                )
            )

        return recommendations

    def generate_report(self) -> SpectralReport:
        """
        Generate a complete spectral analysis report.

        Returns
        -------
        SpectralReport
            Complete report with metrics and recommendations
        """
        if self._eigenvalues is None:
            self._eigenvalues = self.analyzer.get_sorted_eigenvalues()

        if not self._gaps:
            self._compute_gaps()

        eigenvalues = self._eigenvalues
        n_vertices = len(eigenvalues)

        # Get graph info
        G = self.analyzer.graph
        model_info = {
            "type": "DolphinModel",
            "n_vertices": n_vertices,
            "n_edges": G.number_of_edges(),
            "density": 2 * G.number_of_edges() / (n_vertices * (n_vertices - 1))
            if n_vertices > 1
            else 0.0,
        }

        # Add model-specific info if available
        if hasattr(self, "model") and self.model is not None:
            model_info["dimension"] = self.model.dimension
            model_info["has_faces"] = self.model.faces is not None
            model_info["n_faces"] = (
                len(self.model.faces) if self.model.faces is not None else 0
            )

        # Compute metrics
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        lambda_max = eigenvalues[-1] if len(eigenvalues) > 0 else 0.0
        condition_number = lambda_max / lambda_2 if lambda_2 > 1e-10 else float("inf")

        # Get significant gaps
        significant_gaps = self._identify_significant_gaps()
        gap_data = [
            {
                "index": gap.index,
                "lambda_low": float(gap.lambda_low),
                "lambda_high": float(gap.lambda_high),
                "gap_size": float(gap.gap_size),
                "relative_gap": float(gap.relative_gap),
            }
            for gap in significant_gaps
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Create summary
        summary_parts = []
        if lambda_2 < self.LOW_CONNECTIVITY_THRESHOLD:
            summary_parts.append(
                f"Low algebraic connectivity ({lambda_2:.4f}) indicates bottlenecks."
            )
        if condition_number > self.HIGH_CONDITION_THRESHOLD:
            summary_parts.append(
                f"High condition number ({condition_number:.1f}) suggests numerical instability."
            )
        if significant_gaps:
            summary_parts.append(
                f"Found {len(significant_gaps)} significant spectral gap(s) indicating community structure."
            )
        else:
            summary_parts.append("Spectrum is relatively uniform with no major bottlenecks.")

        summary = " ".join(summary_parts)

        return SpectralReport(
            model_info=model_info,
            eigenvalues=[float(e) for e in eigenvalues],
            significant_gaps=gap_data,
            algebraic_connectivity=float(lambda_2),
            condition_number=float(condition_number) if np.isfinite(condition_number) else -1.0,
            recommendations=[rec.to_dict() for rec in recommendations],
            summary=summary,
        )


def analyze_and_report(
    model: DolphinModel,
    output_format: str = "print",
    k_eigenvalues: Optional[int] = None,
) -> SpectralReport:
    """
    Convenience function to analyze a model and generate a report.

    Parameters
    ----------
    model : DolphinModel
        The dolphin model to analyze
    output_format : str, default "print"
        How to output the report: "print", "json", or "return"
    k_eigenvalues : int, optional
        Number of eigenvalues to include in report (None = all)

    Returns
    -------
    SpectralReport
        The generated report (also printed or serialized based on output_format)
    """
    analyzer = UpgradeAnalyzer(model=model)
    report = analyzer.generate_report()

    # Limit eigenvalues in output if requested
    if k_eigenvalues is not None and k_eigenvalues < len(report.eigenvalues):
        report.eigenvalues = report.eigenvalues[:k_eigenvalues]

    if output_format == "print":
        report.print_summary()
    elif output_format == "json":
        print(report.to_json())

    return report


if __name__ == "__main__":
    # Run demonstration
    import numpy as np

    print("Creating sample DolphinModel...")

    # Create a model with known structure (two clusters connected by a bridge)
    n_per_cluster = 10
    vertex_rows: list[list[float]] = []

    # Cluster 1: vertices along x-axis
    for i in range(n_per_cluster):
        vertex_rows.append([i, 0, 0])

    # Cluster 2: vertices along y-axis
    for i in range(n_per_cluster):
        vertex_rows.append([0, i + n_per_cluster, 0])

    vertices = np.array(vertex_rows, dtype=float)

    # Create faces within each cluster
    face_rows: list[list[int]] = []
    # Cluster 1 faces
    for i in range(n_per_cluster - 2):
        face_rows.append([i, i + 1, i + 2])
    # Cluster 2 faces
    for i in range(n_per_cluster - 2):
        face_rows.append([n_per_cluster + i, n_per_cluster + i + 1, n_per_cluster + i + 2])

    # Add a bridge edge (single connection between clusters)
    # This creates a bottleneck
    face_rows.append([n_per_cluster - 1, n_per_cluster, n_per_cluster + 1])

    faces = np.array(face_rows)

    model = DolphinModel(vertices, faces)
    print(f"Created model with {model.n_vertices} vertices, {len(model.edges)} edges\n")

    # Generate and print report
    report = analyze_and_report(model, output_format="print")

    # Also show JSON output
    print("\n\nJSON Output:")
    print(report.to_json(indent=2))
