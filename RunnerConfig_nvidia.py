from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from Plugins.Profilers.EnergiBridge import EnergiBridge

from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath
import sys
import os
import time
import subprocess
import psutil


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER CONFIG ================================
    name: str = "logreg_energy_experiment"
    results_output_path: Path = ROOT_DIR / 'experiments'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 1000

    def get_cpu_temp(self):
        temps = psutil.sensors_temperatures()
        if "k10temp" in temps:
            return max([t.current for t in temps["k10temp"]])
        elif "coretemp" in temps:
            return max([t.current for t in temps["coretemp"]])
        else:
            return None

    def get_gpu_temp(self): # get temperature of nvidia gpu
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            temp_str = result.stdout.strip()
            if temp_str:
                return float(temp_str)
            else:
                return None
        except Exception:
            return None


    def __init__(self):
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN, self.before_run),
            (RunnerEvents.START_RUN, self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT, self.interact),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.STOP_RUN, self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment)
        ])
        self.run_table_model = None
        output.console_log("Custom config loaded (Logistic Regression Energy Benchmark)")

    # ---------------------- Run Table ----------------------
    def create_run_table_model(self) -> RunTableModel:
        factor1 = FactorModel("impl", ['skl', 'xgb', 'trh'])
        factor2 = FactorModel("dataset_size", [1000, 10000, 50000])

        self.run_table_model = RunTableModel(
            factors=[factor1, factor2],
            repetitions=5,
            data_columns=[
                "energy", "runtime", "memory",
                "cpu_temp_before", "gpu_temp_before",
                "cpu_temp_after", "gpu_temp_after"
            ]
        )
        return self.run_table_model

    # ---------------------- Experiment Lifecycle ----------------------

    def before_experiment(self) -> None:
        """Warm-up before the whole experiment"""
        python_path = getattr(self, "VENV_PYTHON", sys.executable)
        output.console_log("[Warm-up] Running each implementation twice before experiment...")

        for impl in ['skl', 'xgb', 'trh']: # Warm up
            for i in range(1):
                cmd = f"{python_path} ml/LR_{impl}.py 1000"
                output.console_log(f"  Warm-up {impl} ({i+1}/1)")
                os.system(cmd)
                time.sleep(10)

        output.console_log("[Warm-up] Completed.\n")

    def before_run(self) -> None:
        pass

    def start_run(self, context: RunnerContext) -> None:
        pass

    def start_measurement(self, context: RunnerContext) -> None:
        impl = context.execute_run["impl"]
        dataset_size = context.execute_run["dataset_size"]
        python_path = getattr(self, "VENV_PYTHON", sys.executable)

        # Record start temps
        cpu_temp_before = self.get_cpu_temp()
        gpu_temp_before = self.get_gpu_temp()
        context.extra_data = {
            "cpu_temp_before": cpu_temp_before,
            "gpu_temp_before": gpu_temp_before
        }
        output.console_log(f"[Run Start] {impl} ({dataset_size}) | "
                           f"CPU={cpu_temp_before}°C | GPU={gpu_temp_before}°C")

        # Start EnergiBridge
        self.profiler = EnergiBridge(
            target_program=f"{python_path} ml/LR_{impl}.py {dataset_size}",
            out_file=context.run_dir / "energibridge.csv"
        )
        self.profiler.start()

    def interact(self, context: RunnerContext) -> None:
        pass

    def stop_measurement(self, context: RunnerContext) -> None:
        stdout = self.profiler.stop(wait=True)

        # Record temps after run
        cpu_temp_after = self.get_cpu_temp()
        gpu_temp_after = self.get_gpu_temp()
        context.extra_data["cpu_temp_after"] = cpu_temp_after
        context.extra_data["gpu_temp_after"] = gpu_temp_after

        output.console_log(f"[Run End] CPU={cpu_temp_after}°C | GPU={gpu_temp_after}°C")
        output.console_log("[Cool-down] Waiting for 3 minutes...")
        time.sleep(180) # Cool down for 3 minutes 

    def stop_run(self, context: RunnerContext) -> None:
        pass

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        eb_log, eb_summary = self.profiler.parse_log(
            self.profiler.logfile,
            self.profiler.summary_logfile
        )

        return {
            "energy": list(eb_log["CPU_ENERGY (J)"].values())[-1]
            - list(eb_log["CPU_ENERGY (J)"].values())[0],
            "runtime": eb_summary["runtime_seconds"],
            "memory": max(eb_log["USED_MEMORY"].values()),
            "cpu_temp_before": context.extra_data.get("cpu_temp_before"),
            "gpu_temp_before": context.extra_data.get("gpu_temp_before"),
            "cpu_temp_after": context.extra_data.get("cpu_temp_after"),
            "gpu_temp_after": context.extra_data.get("gpu_temp_after"),
        }

    def after_experiment(self) -> None:
        output.console_log("[Experiment Completed] All runs finished successfully.")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None
