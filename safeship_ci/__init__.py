"""
safeship_ci — the pipeline-side half of SafeShip.

Runs inside the customer's CI, measures the build using that CI's own token, and
POSTs the result to /score. SafeShip stores no third-party credentials and needs
no network path into customer infrastructure — which is the only arrangement that
works for a firewalled or on-prem Jenkins.

    from safeship_ci import collect, detect
    result = collect(detect())
    print(result.summary())

stdlib only, on purpose: this ships into other people's build agents.
"""
from .collect import Collection, apply_overrides, collect, env_overrides
from .contract import FEATURES, SERVER_DERIVED
from .adapters import detect

__version__ = "1.0.0"
__all__ = ["collect", "detect", "Collection", "FEATURES", "SERVER_DERIVED",
           "apply_overrides", "env_overrides", "__version__"]
