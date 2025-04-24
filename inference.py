from anomaly_detector import AnomalyDetector
from patchcore import PatchCore
from dataset import Dataset
from params import MODELNAME, BACKBONE, THRESHOLD, RESULTDIR
import torch
from utils import *
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    make_dir(f"{RESULTDIR}/good")
    make_dir(f"{RESULTDIR}/bad")
    make_dir(f"{RESULTDIR}/heatmap")
    model = PatchCore(backbone_name=BACKBONE)
    model.load_state_dict(torch.load(MODELNAME))
    model.standby()
    anomaly_detector = AnomalyDetector(model, THRESHOLD)
    dataset = Dataset()
    _, test_loader = dataset.get_dataloaders()

    start = time.perf_counter()
    anomaly_detector.run(test_loader)
    end = time.perf_counter()

    print(f"実行時間:" + "{:2f}".format(end - start) + "秒")
    print(f"一画像に対する時間" + "{:5f}".format((end - start) / anomaly_detector.img_num) + "秒")
    print(f"TP:{anomaly_detector.tp}, TN:{anomaly_detector.tn}, FP:{anomaly_detector.fp}, fn:{anomaly_detector.fn}")
    print(f"正解率:{anomaly_detector.accuracy}, 適合率:{anomaly_detector.precision}, 再現率:{anomaly_detector.recall}, 特異率:{anomaly_detector.specificity}, F値:{anomaly_detector.Fmeasure}")

if __name__ == "__main__":
    main()