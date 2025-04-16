import os
import cv2
import numpy as np
from params import HEADER, RESULTDIR
from dataset import SIZE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
from patchcore import PatchCore
from utils import *

class AnomalyDetector:
    def __init__(self, model:PatchCore, threshold):
        self.threshold = threshold
        self.model = model
        self.ave_time = 0.0
        self.csvPath = f"{RESULTDIR}/score.csv"
        self.img_org =[]
        self.img_num = 0
        self.patch_lib = []
        self.true = []
        self.scores = []

        self.good_num = 0
        self.bad_num = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def make_anomalymap(self, s_map, score, filename):
        # ----------------------------------------------------- #
        #---            異常ヒートマップを作成したい              ---#
        # ----------------------------------------------------- #
        s_map = s_map.squeeze(0)        #tensorのshapeが[1,1,224,224] -> [224,224]
        s_map = s_map.squeeze(0)
        s_map = s_map.detach().cpu().numpy()
        s_map = (s_map - s_map.min()) / (s_map.max() - s_map.min())

        #異常マップをカラー形式に変換
        heatmap = cv2.applyColorMap((s_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #元画像と重ね合わせ
        overlay = cv2.addWeighted(self.img_org, 0.8, heatmap, 0.2, 0)
        #異常度描画
        overlay = cv2.putText(overlay, str(score), (0,10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
        #オーバーレイ画像を保存
        cv2.imwrite(f"{RESULTDIR}/heatmap/{filename}", overlay)

    def memory_bank_visualization(self, patch):
        # ----------------------------------------------------- #
        #---              メモリバンクを可視化したい             ---#
        # ----------------------------------------------------- #
        memory_bank = torch.cat(self.patch_lib, 0)
        print(self.patch_lib.shape)
        memory_bank = self.patch_lib.detach().cpu().numpy()

        pca = PCA(n_components=2)
        memory_bank_2d = pca.fit_transform(self.patch_lib)
        print(memory_bank_2d.shape)
        
        plt.scatter(memory_bank_2d[:, 0], memory_bank_2d[:, 1], s=10, c="red", alpha=0.7)
        plt.title("Memoly Bank")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig(f"{RESULTDIR}/memorybank_inf.jpg")

    def run(self, dataloader):
        self.save_data = []
        cnt = 0
        for images, paths, _ in dataloader:
            for i in range(images.size(0)):
                path = paths[i]
                self.dir_name = os.path.basename(os.path.dirname(path))
                self.img_org = cv2.imread(path)  #元画像
                self.img_org = cv2.resize(self.img_org, (SIZE,SIZE), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
                image_tensor = images[i].unsqueeze(0)
                filename = os.path.basename(path)
                #異常スコア，ピクセルごとの異常度，メモリバンク
                img_lvl_anom_score, s_map, patch = self.model.predict(image_tensor)
                
                score = round(img_lvl_anom_score.item(), 2)
                self.scores.append(score)
                img_score = cv2.putText(self.img_org, str(score), (0,10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
                #異常検知
                self.anomaly_detect(filename, score, img_score)

                #異常ヒートマップ作成
                self.make_anomalymap(s_map, score, filename)

                self.patch_lib.append(patch)
        
        self.img_num = self.good_num + self.bad_num
        #予測の評価
        self.eval()
        Save2Csv(self.save_data, HEADER, save_path=self.csvPath, save_mode="w")
        #ROC曲線
        self.make_roc()

    def anomaly_detect(self, filename, score, img_score):
        #異常判定
        is_anomaly = score < self.threshold
        print(f"ファイル名:{filename}, 異常値スコア: {score}, 異常フラグ: {not is_anomaly}")
        self.save_data.append([filename, score, is_anomaly])
        if is_anomaly:  #正と予想
            cv2.imwrite(f"{RESULTDIR}/good/{filename}", img_score)
            #正常豆と欠点豆正解数を調査
            if self.dir_name == "good":  #正解は正
                self.good_num += 1
                self.tp += 1
                self.true.append(0)     #正常が0，異常が1
            else:                   #正解は負
                self.bad_num += 1   
                self.fp += 1
                self.true.append(1)
        else:           #負と予想
            cv2.imwrite(f"{RESULTDIR}/bad/{filename}", img_score)
            #正常豆と欠点豆正解数を調査
            if self.dir_name == "good":  #正解は正
                self.good_num += 1
                self.fn += 1
                self.true.append(0)
            else:                   #正解は負
                self.bad_num += 1
                self.tn += 1
                self.true.append(1)

        cv2.imwrite(f"{RESULTDIR}/test_all/{filename}", img_score)

    def eval(self):
        self.accuracy = round((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn), 4)
        self.precision = round(self.tp / (self.tp + self.fp), 4)
        self.recall = round(self.tp / (self.tp + self.fn), 4)
        self.specificity = round(self.tn / (self.fp + self.tn), 4)
        self.Fmeasure = round(2 * self.precision * self.recall / (self.precision + self.recall), 4)
    
    def make_roc(self):
        fpr, tpr, thresholds = roc_curve(self.true, self.scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1], [0,1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for PatchCore")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{RESULTDIR}/ROC.jpg")