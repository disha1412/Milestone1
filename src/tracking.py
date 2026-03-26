"""
Simple JSON-based run tracker.
Each run records: id, timestamp, commit, config, split, data_version,
threshold, metrics, artifact paths, and a human-readable note.
"""

import json
import os
import subprocess
from datetime import datetime


def _get_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


class RunTracker:
    def __init__(self, runs_file: str):
        self.runs_file = runs_file
        parent = os.path.dirname(runs_file)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _load(self) -> list:
        if os.path.exists(self.runs_file):
            with open(self.runs_file) as f:
                return json.load(f)
        return []

    def _save(self, runs: list):
        with open(self.runs_file, "w") as f:
            json.dump(runs, f, indent=2)

    def log_run(
        self,
        run_id: str,
        config_name: str,
        split: str,
        data_version: str,
        threshold,
        metrics: dict,
        artifacts: list,
        note: str = "",
    ) -> dict:
        """
        Log or update a run entry.
        threshold: float for a specific threshold, or "sweep" for sweep runs.
        metrics: dict of scalar metric values.
        artifacts: list of output file paths produced by this run.
        """
        runs = self._load()
        run = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "commit_hash": _get_commit_hash(),
            "config_name": config_name,
            "split": split,
            "data_version": data_version,
            "threshold": threshold,
            "metrics": metrics,
            "artifacts": artifacts,
            "note": note,
        }
        idx = next((i for i, r in enumerate(runs) if r["run_id"] == run_id), None)
        if idx is not None:
            runs[idx] = run
        else:
            runs.append(run)
        self._save(runs)
        print(f"[tracking] Logged run '{run_id}' -> {self.runs_file}")
        return run

    def get_run(self, run_id: str) -> dict:
        for r in self._load():
            if r["run_id"] == run_id:
                return r
        raise KeyError(f"Run '{run_id}' not found in {self.runs_file}")

    def get_selected_threshold(self, run_id: str) -> float:
        """Return the scalar threshold logged for a specific run."""
        run = self.get_run(run_id)
        t = run["threshold"]
        if isinstance(t, (int, float)):
            return float(t)
        raise ValueError(f"Run '{run_id}' has non-scalar threshold: {t!r}")

    def list_runs(self) -> list:
        return self._load()

    def print_summary(self):
        runs = self._load()
        if not runs:
            print("[tracking] No runs logged yet.")
            return
        header = f"{'Run ID':<30} {'Split':<6} {'Data':<10} {'Threshold':<12} {'Bal.Acc':<10} {'AUC':<8} Note"
        print(header)
        print("-" * len(header))
        for r in runs:
            thresh = r["threshold"]
            thresh_str = f"{thresh:.4f}" if isinstance(thresh, float) else str(thresh)
            bal_acc = r["metrics"].get("balanced_accuracy", float("nan"))
            auc = r["metrics"].get("auc", float("nan"))
            bal_str = f"{bal_acc:.4f}" if not isinstance(bal_acc, float) or bal_acc == bal_acc else "  n/a"
            auc_str = f"{auc:.4f}" if not isinstance(auc, float) or auc == auc else "  n/a"
            print(f"{r['run_id']:<30} {r['split']:<6} {r['data_version']:<10} {thresh_str:<12} {bal_str:<10} {auc_str:<8} {r['note']}")
