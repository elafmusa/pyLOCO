"""Control-system abstractions used by pyLOCO acquisition tools.

Real control-system implementations are intentionally not imported here, so
pyLOCO remains usable without optional machine-control packages.
"""

from .adapters import (
    AdapterCapability,
    ChannelSample,
    ControlSystemAdapter,
    MockAdapter,
    ReadOnlyAdapter,
    WritableAdapter,
)
from .petra import PETRAReadOnlyAdapter, OptionalDependencyUnavailable
from .backends import (AbstractInterfaceAdapter, BackendDescriptor, BackendSession,
                       InterfaceRegistry)

__all__ = [
    "AdapterCapability",
    "ChannelSample",
    "ControlSystemAdapter",
    "MockAdapter",
    "ReadOnlyAdapter",
    "WritableAdapter",
    "PETRAReadOnlyAdapter",
    "AbstractInterfaceAdapter", "BackendDescriptor", "BackendSession", "InterfaceRegistry",
    "OptionalDependencyUnavailable",
]
