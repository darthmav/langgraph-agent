"""QuICSE engine prototype: the cognitive-synthesis module and its baseline.

This file is what makes `quisce` a package rather than a PEP 420 namespace
package. The difference was not cosmetic: `src/` is on the path, so `import
quisce` already succeeded and returned a module whose `__file__` was `None` and
which exported nothing. `docs/architecture/spectral_behavioral_system_specification.md`
documents `from quisce import QuICSEModule`, and that line raised ImportError
against the namespace package -- the documented entry point was a promise the
code did not keep.

Note that importing this package imports torch, since both modules below need
it at definition time. Nothing else in the project imports `quisce`, so that
cost falls only on a caller that asked for it.
"""

from quisce.baseline import BaselineModule, BaselineState, create_baseline_model
from quisce.quisce_engine import (
    CognitiveState,
    QuICSEModule,
    SpectralContext,
    create_quisce_model,
)

__all__ = [
    "BaselineModule",
    "BaselineState",
    "CognitiveState",
    "QuICSEModule",
    "SpectralContext",
    "create_baseline_model",
    "create_quisce_model",
]
