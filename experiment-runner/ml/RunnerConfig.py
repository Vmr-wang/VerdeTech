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

        output.console_log("Custom config loaded (Logistic Regression)")

    def create_run_table_model(self) -> RunTableModel:
        """Create and return the run_table model here."""
        factor1 = FactorModel("impl", ['skl', 'xgb', 'trh'])
        factor2 = FactorModel("dataset_size", [1000, 10000, 50000])  # 可以根据需求调整

        self.run_table_model = RunTableModel(
            factors=[factor1, factor2],
            exclude_combinations=[
                # 举例：torch 在超大数据集上可能排除
                {factor1: ['torch'], factor2: [50000]},
            ],
            repetitions=5,
            data_columns=["energy", "runtime", "memory"]
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
        impl = context.execute_run["impl"]
        dataset_size = context.execute_run["dataset_size"]

        self.profiler = EnergiBridge(
            target_program=f"/home/abigale/anaconda3/envs/experiment-runner/bin/python ml/LR_{impl}.py {dataset_size}",
            out_file=context.run_dir / "energibridge.csv"
        )
        self.profiler.start()

    def interact(self, context: RunnerContext) -> None:
        pass

    def stop_measurement(self, context: RunnerContext) -> None:
        stdout = self.profiler.stop(wait=True)

    def stop_run(self, context: RunnerContext) -> None:
        pass

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        eb_log, eb_summary = self.profiler.parse_log(
            self.profiler.logfile, 
            self.profiler.summary_logfile
        )

        # 使用 PACKAGE_ENERGY 代替 SYSTEM_POWER
        energy_values = list(eb_log["PACKAGE_ENERGY (J)"].values())
        
        return {
            "energy": energy_values[-1] - energy_values[0],  # 总能耗差值
            "runtime": eb_summary["runtime_seconds"], 
            "memory": max(eb_log["USED_MEMORY"].values())
        }

    def after_experiment(self) -> None:
        pass

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None
