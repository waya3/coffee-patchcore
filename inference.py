from anomaly_detector import AnomalyDetector
from patchcore import PatchCore
from dataset import Dataset
from params import MODELNAME, DATASET, BACKBONE, THRESHOLD, DIRNAME
import torch

def main():
    model = PatchCore(backbone_name=BACKBONE)
    model.load_state_dict(torch.load(MODELNAME))
    model.standby()
    anomaly_detector = AnomalyDetector(model, THRESHOLD)
    dataset = Dataset(DATASET)
    _, test_loader = dataset.get_dataloaders()
    anomaly_detector.run(test_loader)

if __name__ == "__main__":
    main()