from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from Plugins.Profilers.EnergiBridge import EnergiBridge

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from os.path import dirname, realpath
import numpy as np
import time
import sys
import subprocess
import threading


class NvidiaSMIPowerMonitor:
    """通过 nvidia-smi 实时监控 GPU 功耗并计算能耗"""
    
    def __init__(self, gpu_id: int = 0, sample_interval: float = 0.1):
        self.gpu_id = gpu_id
        self.sample_interval = sample_interval
        self.power_samples: List[Tuple[float, float]] = []
        self.monitoring = False
        self.monitor_thread = None
        self.supports_power = None
        
    def test_power_support(self) -> bool:
        """测试 nvidia-smi 是否支持功耗查询"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', 
                 '--format=csv,noheader,nounits', f'--id={self.gpu_id}'],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                power_str = result.stdout.strip()
                if power_str and power_str not in ['[N/A]', 'N/A', '[Not Supported]']:
                    try:
                        float(power_str)
                        return True
                    except ValueError:
                        return False
            return False
        except Exception:
            return False
    
    def start(self):
        """开始监控 GPU 功耗"""
        self.supports_power = self.test_power_support()
        if not self.supports_power:
            output.console_log("[GPU Monitor] nvidia-smi does not support power monitoring")
            return
        
        self.power_samples = []
        self.monitoring = True
        self._collect_sample()  # 立即采集第一个样本
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        output.console_log(f"[GPU Monitor] Started (sampling every {self.sample_interval}s)")
        
    def _collect_sample(self):
        """采集一次功耗样本"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw',
                 '--format=csv,noheader,nounits', f'--id={self.gpu_id}'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                power_str = result.stdout.strip()
                if power_str and power_str not in ['[N/A]', 'N/A']:
                    power_watts = float(power_str)
                    self.power_samples.append((time.time(), power_watts))
        except Exception:
            pass
            
    def _monitor_loop(self):
        """后台持续采样"""
        while self.monitoring:
            self._collect_sample()
            time.sleep(self.sample_interval)
    
    def stop(self) -> Dict[str, Any]:
        """停止监控并计算能耗"""
        if self.monitoring:
            self._collect_sample()  # 最后一次采样
        
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3)
        
        if not self.supports_power:
            return {'energy_j': None, 'avg_power_w': None, 'samples': 0, 'method': 'unsupported'}
        
        if len(self.power_samples) < 2:
            output.console_log(f"[GPU Monitor] Insufficient samples ({len(self.power_samples)})")
            return {'energy_j': None, 'avg_power_w': None, 'samples': len(self.power_samples), 'method': 'insufficient'}
        
        # 梯形积分计算能耗
        energy_j = 0.0
        for i in range(len(self.power_samples) - 1):
            t1, p1 = self.power_samples[i]
            t2, p2 = self.power_samples[i + 1]
            energy_j += (p1 + p2) / 2.0 * (t2 - t1)
        
        powers = [p for _, p in self.power_samples]
        result = {
            'energy_j': energy_j,
            'avg_power_w': np.mean(powers),
            'samples': len(self.power_samples),
            'method': 'nvidia_smi'
        }
        output.console_log(f"[GPU Monitor] Energy: {energy_j:.2f}J, Samples: {len(self.power_samples)}")
        return result


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))
    name: str = "VerdeTech_experiment"
    results_output_path: Path = ROOT_DIR / 'experiments'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 1000

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
        
        # GPU monitoring variables
        self.gpu_available = False
        self.gpu_handle = None
        self.gpu_tdp_watts = None
        self.gpu_power_monitor = None
        self.gpu_enabled = False
        self.gpu_power_data = None
        self.gpu_utilization_samples = []
        self.gpu_memory_samples = []

        output.console_log("Custom config loaded (LogReg Auto CPU/GPU Energy Monitoring)")

    def create_run_table_model(self) -> RunTableModel:
        factor_alg = FactorModel("alg", ['LR', 'RR', 'DT', 'KMeans'])
        factor_iml = FactorModel("impl", ['skl_cpu', 'intelex_cpu', 'xgb_cpu', 'xgb_gpu', 'tf_gpu', 'trh_gpu'])
        factor_dataset = FactorModel("dataset", ['iris', 'wine', 'breast_cancer', 'digits'])

        self.run_table_model = RunTableModel(
            factors=[factor_alg, factor_iml, factor_dataset],
            repetitions=5,
            data_columns=[
                "actual_size", "n_features",
                "cpu_energy", "cpu_util",
                "gpu_energy", "gpu_util",
                "runtime", "memory", "accuracy"
            ]
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Initialize GPU monitoring using pynvml"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
            
            # Get GPU TDP for backup estimation
            try:
                power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle)
                self.gpu_tdp_watts = power_limit_mw / 1000.0
                output.console_log(f"GPU TDP: {self.gpu_tdp_watts:.1f}W")
            except:
                self.gpu_tdp_watts = None
            
            self.gpu_available = True
            output.console_log(f"GPU detected: {gpu_name}")
            
        except Exception as e:
            output.console_log(f"[INFO] No GPU or NVML init failed: {e}")
            self.gpu_available = False

    def before_run(self) -> None:
        pass

    def start_run(self, context: RunnerContext) -> None:
        pass       

    def start_measurement(self, context: RunnerContext) -> None:
        """Automatically decide whether to enable GPU profiling"""
        alg = context.execute_run["alg"]
        impl = context.execute_run["impl"]
        dataset = context.execute_run["dataset"]

        gpu_keywords = ["gpu", "cuda", "torch"]
        self.gpu_enabled = any(k in impl.lower() for k in gpu_keywords)

        # --- CPU profiler (EnergiBridge) ---
        self.profiler_cpu = EnergiBridge(
            target_program=f"{sys.executable} ml/{alg}_{impl}.py {dataset}",
            out_file=context.run_dir / "energibridge.csv"
        )
        self.profiler_cpu.start()

        # --- GPU profiler (only if needed) ---
        if self.gpu_enabled and self.gpu_available:
            output.console_log(f"GPU profiling enabled for implementation: {impl}")
            
            # Start nvidia-smi power monitoring
            self.gpu_power_monitor = NvidiaSMIPowerMonitor(gpu_id=0, sample_interval=0.1)
            self.gpu_power_monitor.start()
            
            # Reset sampling arrays
            self.gpu_utilization_samples = []
            self.gpu_memory_samples = []
        else:
            output.console_log(f"GPU profiling skipped for implementation: {impl}")
            self.gpu_power_monitor = None

    def interact(self, context: RunnerContext) -> None:
        """Collect GPU utilization samples during execution"""
        if self.gpu_enabled and self.gpu_available:
            self._collect_gpu_sample()
        time.sleep(2)

    def _collect_gpu_sample(self) -> None:
        """Collect one GPU utilization and memory sample"""
        try:
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            self.gpu_utilization_samples.append(util.gpu)
            
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            self.gpu_memory_samples.append(mem.used / 1024**2)
        except Exception:
            pass

    def stop_measurement(self, context: RunnerContext) -> None:
        """Stop both CPU and GPU profilers"""
        # Final GPU sample
        if self.gpu_enabled and self.gpu_available:
            self._collect_gpu_sample()
        
        # Stop GPU power monitoring
        if self.gpu_power_monitor:
            self.gpu_power_data = self.gpu_power_monitor.stop()
        
        # Stop CPU profiler
        self.profiler_stdout = self.profiler_cpu.stop(wait=True)

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
        
        if self.gpu_enabled and self.gpu_available:
            # GPU Utilization (from pynvml samples)
            if len(self.gpu_utilization_samples) > 0:
                gpu_util = np.mean(self.gpu_utilization_samples)
            
            # GPU Energy
            if self.gpu_power_data:
                method = self.gpu_power_data.get('method', 'unknown')
                
                # Method 1: nvidia-smi power sampling
                if method == 'nvidia_smi' and self.gpu_power_data['energy_j'] is not None:
                    gpu_energy = self.gpu_power_data['energy_j']
                    output.console_log(f"GPU Energy: {gpu_energy:.2f}J (from nvidia-smi)")
                
                # Method 2: Estimate from TDP and utilization
                elif method in ['unsupported', 'insufficient']:
                    if gpu_util is not None and self.gpu_tdp_watts and runtime > 0:
                        gpu_energy = self.gpu_tdp_watts * (gpu_util / 100.0) * runtime
                        output.console_log(f"GPU Energy: {gpu_energy:.2f}J (estimated: TDP={self.gpu_tdp_watts:.1f}W × util={gpu_util:.1f}% × time={runtime:.1f}s)")
                    else:
                        output.console_log("[WARNING] Cannot estimate GPU energy: missing data")

        return {
            "actual_size": dataset_info.get("actual_size", 0),
            "n_features": dataset_info.get("n_features", 0),
            "cpu_energy": cpu_energy,
            "cpu_util": cpu_util,
            "gpu_energy": gpu_energy,
            "gpu_util": gpu_util,
            "runtime": runtime,
            "memory": max_memory,
            "accuracy": dataset_info.get("accuracy", 0.0)
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
        return info

    def after_experiment(self) -> None:
        output.console_log("Experiment completed! Check the results in the experiments folder.")
        if self.gpu_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except:
                pass

    experiment_path: Path = None