from models.CRUW_finetune.cruw import CRUW
from models.CRUW_finetune.cruw.eval import evaluate_rod2021


data_root = "/mnt/truenas_datasets/Datasets_mmRadar/ROD2021"
submit_dir = r"ft_CRUW_output_dir/20250528_202053_RADTR_cruw_tiny_finetune_eval/17/evaluate"
truth_dir = r"models/CRUW_finetune/gt"

if __name__ == '__main__':

    dataset = CRUW(data_root=data_root, sensor_config_name='sensor_config_rod2021')
    print(dataset)
    evaluate_rod2021(submit_dir, truth_dir, dataset)