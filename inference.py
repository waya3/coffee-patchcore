from anomaly_detector import AnomalyDetector
from patchcore import PatchCore
from dataset import Dataset
from params import MODELNAME, DATASET, BACKBONE, THRESHOLD, DIRNAME
import torch
from utils import *
import time

def main():
    make_dir(f"{DIRNAME}/result/good")
    make_dir(f"{DIRNAME}/result/bad")
    model = PatchCore(backbone_name=BACKBONE)
    model.load_state_dict(torch.load(MODELNAME))
    model.standby()
    anomaly_detector = AnomalyDetector(model, THRESHOLD)
    dataset = Dataset(DATASET)
    _, test_loader = dataset.get_dataloaders()

    start = time.perf_counter()
    anomaly_detector.run(test_loader)
    end = time.perf_counter()
    print(f"実行時間:" + "{:2f}".format(end - start) + "秒")
    print(f"一画像に対する時間" + "{:5f}".format((end - start) / anomaly_detector.img_num) + "秒")

if __name__ == "__main__":
    main()