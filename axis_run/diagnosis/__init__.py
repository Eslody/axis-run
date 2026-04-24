"""axis-run 诊断与容错扩展。"""

from axis_run.diagnosis.fault_config import FaultConfigDiagnostician
from axis_run.diagnosis.fault_failover import FaultConfigFailover

__all__ = ["FaultConfigDiagnostician", "FaultConfigFailover"]
