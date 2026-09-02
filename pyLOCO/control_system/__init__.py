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
from .pysc_profiles import (PySCMachineProfile, available_pysc_profiles,
                            load_pysc_catalog, load_pysc_profile)

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
    "PySCMachineProfile", "available_pysc_profiles", "load_pysc_catalog", "load_pysc_profile",
]
