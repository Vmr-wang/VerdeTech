from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from Plugins.Profilers.EnergiBridge import EnergiBridge
from Plugins.Profilers.NvidiaML import NvidiaML, NVML_Field, NVML_Sample, NVML_IDs

from typing import Dict, Any, Optional
from pathlib import Path
from os.path import dirname, realpath
import numpy as np
import time
import sys
import os
import psutil
import subprocess


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))
    name: str = "VerdeTech_experiment"
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
        self.profiler_stdout = ""
        self.profiler_cpu = None
        self.profiler_gpu = None
        self.gpu_enabled = False

        self.valid_pairs = {
        # Logistic Regression
        ("LR", "skl_cpu"),
        ("LR", "trh_gpu"),
        ("LR", "intelex_cpu"),

        # KMeans
        ("KMeans", "skl_cpu"),
        ("KMeans", "intelex_cpu"),
        ("KMeans", "tf_gpu"),

        # Decision Tree
        ("DT", "skl_cpu"),
        ("DT", "xgb_cpu"),
        ("DT", "xgb_gpu"),

        # Ridge Regression
        ("RR", "skl_cpu"),
        ("RR", "intelex_cpu"),
        ("RR", "trh_gpu"),}

        output.console_log("Custom config loaded (LogReg Auto CPU/GPU Energy Monitoring)")

    def create_run_table_model(self) -> RunTableModel:
        factor_alg = FactorModel("alg", ['LR', 'RR', 'DT', 'KMeans'])
        factor_iml = FactorModel("impl", ['skl_cpu', 'intelex_cpu', 'xgb_cpu', 'xgb_gpu', 'tf_gpu', 'trh_gpu'])
        factor_dataset = FactorModel("dataset", ['iris', 'wine', 'breast_cancer', 'digits'])
        # TODO: experiment such as LR__xgb_cpu.py does not exist,we need to make the Cartesian product nice and correct，the code could run now,but generate ugly csvs with missing line

        allowed_pairs = [('LR', 'skl_cpu'), ('LR', 'intelex_cpu'), ('LR', 'trh_gpu'),
                         ('KMeans', 'skl_cpu'), ('KMeans', 'intelex_cpu'), ('KMeans', 'tf_gpu'),
                         ('DT', 'skl_cpu'), ('DT', 'xgb_cpu'), ('DT', 'xgb_gpu'),
                         ('RR', 'skl_cpu'), ('RR', 'intelex_cpu'), ('RR', 'trh_gpu'), ('RR', 'xgb_gpu')]
        

        my_exclude_combinations = []
        for alg in ['LR', 'RR', 'DT', 'KMeans']:
            for impl in ['skl_cpu', 'intelex_cpu', 'xgb_cpu', 'xgb_gpu', 'tf_gpu', 'trh_gpu']:
                if (alg, impl) not in allowed_pairs:
                    my_exclude_combinations.append({
                        factor_alg: [alg],
                        factor_iml: [impl]
                    })

        self.run_table_model = RunTableModel(
            factors=[factor_alg, factor_iml, factor_dataset],
            exclude_combinations = my_exclude_combinations,
            repetitions=5,
            data_columns=[
                #"dataset_name",
                "actual_size", "n_features",
                "cpu_energy", "cpu_util",
                "gpu_energy", "gpu_util",
                "runtime", "memory", "accuracy", "mse",
                "cpu_temp_before", "gpu_temp_before",
                "cpu_temp_after", "gpu_temp_after"
            ]
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Initialize GPU profiler if available and warm-up before the whole experiment"""
        # --- GPU initialization ---
        try:
            self.profiler_gpu = NvidiaML(
                fields=[
                    NVML_Field.NVML_FI_DEV_POWER_INSTANT,
                    NVML_Field.NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
                    NVML_Field.NVML_FI_DEV_GPU_UTILIZATION,
                    NVML_Field.NVML_FI_DEV_MEMORY_USED
                ],
                samples=[NVML_Sample.NVML_GPU_UTILIZATION_SAMPLES]
            )
            devices = self.profiler_gpu.list_devices(print_dev=False)
            self.profiler_gpu.open_device(0, NVML_IDs.NVML_ID_INDEX)
            output.console_log(f"NVML initialized: GPU detected - {devices[0]['name']}")
        except Exception as e:
            output.console_log(f"[INFO] No GPU or NVML init failed: {e}")
            self.profiler_gpu = None

        # --- Warm-up ---
        output.console_log("[Warm-up] Running each implementation twice before experiment...")

        for impl in ['skl', 'intelex']:  # Warm up
            for i in range(3):
                cmd = f"{sys.executable} ml/LR_{impl}_cpu.py iris"
                output.console_log(f"  Warm-up {impl} ({i+1}/3)")

                cpu_temp = self.get_cpu_temp()
                gpu_temp = self.get_gpu_temp()
                output.console_log(f"[Warm-up Temps] CPU={cpu_temp}°C | GPU={gpu_temp}°C")
                os.system(cmd)
                time.sleep(10) # Cool down for 10s

        output.console_log("[Warm-up] Completed.\n")


            
    def before_run(self) -> None:
        pass

    def start_run(self, context: RunnerContext) -> None:
        pass       

    def start_measurement(self, context: RunnerContext) -> None:
        """Automatically decide whether to enable GPU profiling"""
        alg = context.execute_run["alg"]
        impl = context.execute_run["impl"]
        dataset = context.execute_run["dataset"]

        # Record start temps
        cpu_temp_before = self.get_cpu_temp()
        gpu_temp_before = self.get_gpu_temp()
        context.extra_data = {
            "cpu_temp_before": cpu_temp_before,
            "gpu_temp_before": gpu_temp_before
        }
        output.console_log(f"[Run Start] {impl} ({dataset}) | "
                           f"CPU={cpu_temp_before}°C | GPU={gpu_temp_before}°C")


        if (alg, impl) not in self.valid_pairs:
            output.console_log(f"Skipping invalid combination: {alg}_{impl}")
            context.skip_run = True
            # 有的 Runner 支持 skip_run，有的不支持；保险起见直接 return
            return


        gpu_keywords = ["gpu", "cuda", "torch"]
        self.gpu_enabled = any(k in impl.lower() for k in gpu_keywords)

        # --- CPU profiler (EnergiBridge) ---
        self.profiler_cpu = EnergiBridge(
            target_program=f"{sys.executable} ml/{alg}_{impl}.py {dataset}",
            out_file=context.run_dir / "energibridge.csv"
        )
        self.profiler_cpu.start()

        # --- GPU profiler (only if needed) ---
        if self.gpu_enabled and self.profiler_gpu is not None:
            output.console_log(f"GPU profiling enabled for implementation: {impl}")
            self.profiler_gpu.logfile = context.run_dir / "nvml_log.json"
            self.profiler_gpu.start()
        else:
            output.console_log(f"GPU profiling skipped for implementation: {impl}")

    def interact(self, context: RunnerContext) -> None:
        time.sleep(2)

    def stop_measurement(self, context: RunnerContext) -> None:
        """Stop both CPU and GPU profilers and perform cool-down"""
        # --- Stop CPU profiler ---
        self.profiler_stdout = self.profiler_cpu.stop(wait=True)

        # --- Stop GPU profiler if enabled ---
        if self.gpu_enabled and self.profiler_gpu:
            try:
                self.profiler_gpu.stop()
            except Exception as e:
                output.console_log(f"Failed to stop GPU profiler: {e}")

        # --- Record temperatures after run ---
        cpu_temp_after = getattr(self, "get_cpu_temp", lambda: None)()
        gpu_temp_after = getattr(self, "get_gpu_temp", lambda: None)()
        context.extra_data["cpu_temp_after"] = cpu_temp_after
        context.extra_data["gpu_temp_after"] = gpu_temp_after

        output.console_log(f"[Run End] CPU={cpu_temp_after}°C | GPU={gpu_temp_after}°C")

        # --- Cool-down ---
        output.console_log("[Cool-down] Waiting for 1 minute...")
        time.sleep(60)  # Cool down for 1 minute


    def stop_run(self, context: RunnerContext) -> None:
        pass

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        """Parse both CPU (EnergiBridge) and GPU (NVML) data"""
        eb_log, eb_summary = self.profiler_cpu.parse_log(
            self.profiler_cpu.logfile,
            self.profiler_cpu.summary_logfile
        )

        # --- CPU energy ---
        cpu_energy = 0.0
        if "PACKAGE_ENERGY (J)" in eb_log:
            vals = list(eb_log["PACKAGE_ENERGY (J)"].values())
            cpu_energy = vals[-1] - vals[0]

        # --- CPU utilization (avg across cores) ---
        cpu_util_cols = [c for c in eb_log.keys() if c.startswith("CPU_USAGE_")]
        cpu_util = None
        if cpu_util_cols:
            # 计算所有 CPU 核心的平均利用率
            all_values = []
            for col in cpu_util_cols:
                all_values.extend(list(eb_log[col].values()))
            cpu_util = np.mean(all_values) if all_values else None

        runtime = eb_summary.get("runtime_seconds", 0)
        
        # --- Memory handling ---
        max_memory = 0
        if "USED_MEMORY" in eb_log:
            memory_values = list(eb_log["USED_MEMORY"].values())
            max_memory = max(memory_values) if memory_values else 0

        dataset_info = self._parse_dataset_info(self.profiler_stdout)

        # --- GPU Data ---
        gpu_energy = None
        gpu_util = None
        if self.gpu_enabled and self.profiler_gpu and hasattr(self.profiler_gpu, "logfile") and self.profiler_gpu.logfile.exists():
            nvml_log = self.profiler_gpu.parse_log(self.profiler_gpu.logfile, remove_errors=True)
            try:
                if "NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION" in nvml_log:
                    energy_vals = [v[1] for v in nvml_log["NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION"]]
                    gpu_energy = (energy_vals[-1] - energy_vals[0]) / 1000.0  # mJ → J
                if "NVML_FI_DEV_GPU_UTILIZATION" in nvml_log:
                    gpu_util = np.mean([v[1] for v in nvml_log["NVML_FI_DEV_GPU_UTILIZATION"]])
            except Exception as e:
                output.console_log(f"GPU log parse failed: {e}")

        return {
            "dataset_name": dataset_info.get("dataset_name", "Unknown"),
            "actual_size": dataset_info.get("actual_size", 0),
            "n_features": dataset_info.get("n_features", 0),
            "cpu_energy": cpu_energy,
            "cpu_util": cpu_util,
            "gpu_energy": gpu_energy,
            "gpu_util": gpu_util,
            "runtime": runtime,
            "memory": max_memory,
            "accuracy": dataset_info.get("accuracy", 0.0),
            "mse": dataset_info.get("mse", 0.0),
            "cpu_temp_before": context.extra_data.get("cpu_temp_before"),
            "gpu_temp_before": context.extra_data.get("gpu_temp_before"),
            "cpu_temp_after": context.extra_data.get("cpu_temp_after"),
            "gpu_temp_after": context.extra_data.get("gpu_temp_after"),
        }

    def _parse_dataset_info(self, stdout: str) -> Dict[str, Any]:
        info = {}
        if not stdout:
            return info
        for line in stdout.splitlines():
            line = line.strip()
            if "DATASET_NAME:" in line:
                info["dataset_name"] = line.split(":", 1)[1].strip()
            elif "ACTUAL_SIZE:" in line:
                info["actual_size"] = int(line.split(":", 1)[1].strip())
            elif "N_FEATURES:" in line:
                info["n_features"] = int(line.split(":", 1)[1].strip())
            elif "ACCURACY:" in line:
                info["accuracy"] = float(line.split(":", 1)[1].strip())
            elif "MSE:" in line: 
                info["mse"] = float(line.split(":", 1)[1].strip())
        return info

    def after_experiment(self) -> None:
        output.console_log("Experiment completed! Check the results in the experiments folder.")
        if self.profiler_gpu:
            self.profiler_gpu.close_device()

    experiment_path: Path = None