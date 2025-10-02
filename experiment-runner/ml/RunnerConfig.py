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


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name:                       str             = "logreg_energy_experiment"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path:        Path             = ROOT_DIR / 'experiments'

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type:             OperationType   = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms:    int             = 1000

    def __init__(self):
        """Executes immediately after program start, on config load"""

        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN       , self.before_run       ),
            (RunnerEvents.START_RUN        , self.start_run        ),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT         , self.interact         ),
            (RunnerEvents.STOP_MEASUREMENT , self.stop_measurement ),
            (RunnerEvents.STOP_RUN         , self.stop_run         ),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT , self.after_experiment )
        ])
        self.run_table_model = None  # Initialized later
        self.profiler_stdout = ""     # 用于存储 profiler 的输出

        output.console_log("Custom config loaded (Logistic Regression - Real Datasets)")

    def create_run_table_model(self) -> RunTableModel:
        """Create and return the run_table model here."""
        # 使用数据集名称作为因子
        factor1 = FactorModel("impl", ['skl', 'xgb', 'trh'])
        factor2 = FactorModel("dataset", ['iris', 'wine', 'breast_cancer', 'digits'])
        
        self.run_table_model = RunTableModel(
            factors=[factor1, factor2],
            exclude_combinations=[
                # 如果需要排除某些组合，在这里添加
            ],
            repetitions=5,
            data_columns=["dataset_name", "actual_size", "n_features", "energy", "runtime", "memory", "accuracy"]
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        pass

    def before_run(self) -> None:
        pass

    def start_run(self, context: RunnerContext) -> None:
        pass       

    def start_measurement(self, context: RunnerContext) -> None:
        """启动逻辑回归的不同实现"""
        # 正确：使用 execute_run（根据你的原始代码）
        impl = context.execute_run["impl"]
        dataset = context.execute_run["dataset"]

        self.profiler = EnergiBridge(
            target_program=f"/home/abigale/anaconda3/envs/experiment-runner/bin/python ml/LR_{impl}.py {dataset}",
            out_file=context.run_dir / "energibridge.csv"
        )
        self.profiler.start()

    def interact(self, context: RunnerContext) -> None:
        pass

    def stop_measurement(self, context: RunnerContext) -> None:
        self.profiler_stdout = self.profiler.stop(wait=True)

    def stop_run(self, context: RunnerContext) -> None:
        pass

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        """收集运行数据并返回"""
        eb_log, eb_summary = self.profiler.parse_log(
            self.profiler.logfile, 
            self.profiler.summary_logfile
        )

        # 使用 PACKAGE_ENERGY 代替 SYSTEM_POWER
        energy_values = list(eb_log["PACKAGE_ENERGY (J)"].values())
        
        # 从 stdout 解析数据集信息
        dataset_info = self._parse_dataset_info(self.profiler_stdout)
        
        return {
            "dataset_name": dataset_info.get("dataset_name", "Unknown"),
            "actual_size": dataset_info.get("actual_size", 0),
            "n_features": dataset_info.get("n_features", 0),
            "energy": energy_values[-1] - energy_values[0],
            "runtime": eb_summary["runtime_seconds"], 
            "memory": max(eb_log["USED_MEMORY"].values()),
            "accuracy": dataset_info.get("accuracy", 0.0)
        }

    def _parse_dataset_info(self, stdout: str) -> Dict[str, Any]:
        """从 stdout 解析数据集信息"""
        info = {}
        
        if not stdout:
            return info
        
        lines = stdout.split('\n')
        for line in lines:
            line = line.strip()
            
            if "DATASET_NAME:" in line:
                info["dataset_name"] = line.split(":", 1)[1].strip()
            elif "ACTUAL_SIZE:" in line:
                try:
                    info["actual_size"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    info["actual_size"] = 0
            elif "N_FEATURES:" in line:
                try:
                    info["n_features"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    info["n_features"] = 0
            elif "ACCURACY:" in line:
                try:
                    info["accuracy"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    info["accuracy"] = 0.0
        
        return info

    def after_experiment(self) -> None:
        output.console_log("Experiment completed! Check the results in the experiments folder.")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None