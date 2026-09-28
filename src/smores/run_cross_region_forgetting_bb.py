import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from experiments.single_step.cross_region_forgetting import CrossRegionForgetting

def main():
    """Same as run_cross_region_forgetting.py, extended with SMORES BB folds
    (SMORES_preprocessing_plan_v2.md). train_order picks up "bb1".."bbN"
    once liveness classification runs on the passed-in data_dir_smores --
    see folds.REPORTED_LIVE_CHANNELS for how many folds to expect.
    """
    runs = 1
    exp = CrossRegionForgetting(
        data_path_sb='DO_allsites_allyears_20250611.csv',
        data_path_fl='3OEC_current_flow.csv',
        num_train_epochs=100,
        sampling_freq='5min',
        sequence_len=12,
        device='mps',
        data_dir_smores='/path/to/smores/raw/chunk/dir',
    )
    save_dir = f"./out/cross_region_forgetting_bb_{runs}runs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    exp.run_experiment(
        num_runs=1,
        save_dir=save_dir,
        train_order=["FL1", "FL2", "FL3", "FL4", "SB1", "SB2", "SB3", "SB4", "BB1", "BB2", "BB3"],
        eval_all=False,
    )

if __name__=='__main__':
    main()
