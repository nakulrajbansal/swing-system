"""Edge signals. Each implements the EdgeSignal interface in base.py."""

from harness.signals.base import EdgeSignal
from harness.signals.edge01_filing import Edge01Filing
from harness.signals.edge02_8k import Edge02EightK
from harness.signals.edge06_insider import Edge06Insider
from harness.signals.edge07_links import Edge07Links
from harness.signals.edge08_momo import Edge08Momentum

ALL_FREE_EDGES = [
    Edge01Filing, Edge02EightK, Edge06Insider, Edge07Links, Edge08Momentum,
]

__all__ = [
    "EdgeSignal", "Edge01Filing", "Edge02EightK", "Edge06Insider",
    "Edge07Links", "Edge08Momentum", "ALL_FREE_EDGES",
]
