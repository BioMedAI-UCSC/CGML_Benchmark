from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass(frozen=True)
class MachineConf:
    """
    Machine-specific paths, loaded from a single JSON file.

    Keys are intentionally verbose/descriptive to avoid ambiguity:
      - native_data_300k_dir: folder containing per-protein native simulation runs at 300K
      - native_data_350k_dir: folder containing 350K numpy native coords (Majewski dataset)
      - analysis_cache_dir: cache for TICA/PCA/MSM and other intermediates
      - benchmark_outputs_dir: where new benchmark runs are written
      - experimental_structure_rmsd_dir: RMSD reference structures / experimental structure dir
    """
    native_data_300k_dir: Path
    native_data_350k_dir: Path
    analysis_cache_dir: Path
    benchmark_outputs_dir: Path
    experimental_structure_rmsd_dir: Path

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MachineConf":
        """
        Supports:
          - new verbose keys (preferred)
          - legacy keys for backward compatibility
        """
        # New keys
        if "native_data_300k_dir" in d:
            return MachineConf(
                native_data_300k_dir=Path(d["native_data_300k_dir"]),
                native_data_350k_dir=Path(d["native_data_350k_dir"]),
                analysis_cache_dir=Path(d["analysis_cache_dir"]),
                benchmark_outputs_dir=Path(d["benchmark_outputs_dir"]),
                experimental_structure_rmsd_dir=Path(d["experimental_structure_rmsd_dir"]),
            )

        # Legacy compatibility (your old fields)
        # data_300_path, data_350_path, cache_path, sims_store_dir, rmsd_dir
        return MachineConf(
            native_data_300k_dir=Path(d["data_300_path"]),
            native_data_350k_dir=Path(d["data_350_path"]),
            analysis_cache_dir=Path(d["cache_path"]),
            benchmark_outputs_dir=Path(d["sims_store_dir"]),
            experimental_structure_rmsd_dir=Path(d["rmsd_dir"]),
        )

    @staticmethod
    def load_json(path: Path) -> "MachineConf":
        with open(path, "r") as f:
            data = json.load(f)
        return MachineConf.from_dict(data)

    def with_overrides(
        self,
        *,
        native_data_300k_dir: Optional[Path] = None,
        native_data_350k_dir: Optional[Path] = None,
        analysis_cache_dir: Optional[Path] = None,
        benchmark_outputs_dir: Optional[Path] = None,
        experimental_structure_rmsd_dir: Optional[Path] = None,
    ) -> "MachineConf":
        """
        Return a new MachineConf with any provided fields overridden.
        """
        return replace(
            self,
            native_data_300k_dir=native_data_300k_dir or self.native_data_300k_dir,
            native_data_350k_dir=native_data_350k_dir or self.native_data_350k_dir,
            analysis_cache_dir=analysis_cache_dir or self.analysis_cache_dir,
            benchmark_outputs_dir=benchmark_outputs_dir or self.benchmark_outputs_dir,
            experimental_structure_rmsd_dir=experimental_structure_rmsd_dir or self.experimental_structure_rmsd_dir,
        )


def list_machine_configs(machine_config_dir: Path) -> dict[str, Path]:
    """
    Finds *.json in machine_config_dir and returns {machine_name: path}
    where machine_name is the filename stem (e.g., nersc.json -> 'nersc').
    """
    out: dict[str, Path] = {}
    for p in sorted(machine_config_dir.glob("*.json")):
        out[p.stem] = p
    return out

