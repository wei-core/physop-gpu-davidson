"""GPU Davidson components."""

from .contract import DavidsonShape, refresh_dynamic_hamiltonian, validate_graph

__all__ = ["DavidsonShape", "GenericDavidson", "refresh_dynamic_hamiltonian", "validate_graph"]


def __getattr__(name):
    if name == "GenericDavidson":
        from .solver import GenericDavidson
        return GenericDavidson
    raise AttributeError(name)
