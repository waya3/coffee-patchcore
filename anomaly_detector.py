import os
import cv2
import numpy as np
from params import DIRNAME, HEADER
from dataset import SIZE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from utils import *

class AnomalyDetector:
    def __init__(self, model, threshold):
        self.threshold = threshold
        self.model = model
        self.ave_time = 0.0
        self.csvPath = DIRNAME + "/score.csv"
        self.img_org =[]
        self.img_num = 0
        self.patch_lib = []

    def make_anomalymap(self, s_map, filename):
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
        overlay = cv2.addWeighted(self.img_org, 0.6, heatmap, 0.4, 0)
        #オーバーレイ画像を保存
        cv2.imwrite(f"{DIRNAME}/heatmap/{filename}", overlay)

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
        plt.savefig(DIRNAME + "_memorybank_inf.jpg")

    def run(self, dataloader):
        save_data = []
        cnt = 0
        for images, paths, _ in dataloader:
            for i in range(images.size(0)):
                cnt += 1
                path = paths[i]
                self.img_org = cv2.imread(path)  #元画像
                self.img_org = cv2.resize(self.img_org, (SIZE,SIZE), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
                image_tensor = images[i].unsqueeze(0)
                filename = os.path.basename(path)
                #異常スコア，ピクセルごとの異常度，メモリバンク
                img_lvl_anom_score, s_map, patch = self.model.predict(image_tensor)
                
                score = img_lvl_anom_score.item()
                is_anomaly = score > self.threshold
                print(f"ファイル名:{filename}, 異常値スコア: {score}, 異常フラグ: {is_anomaly}")
                save_data.append([filename, score, is_anomaly])
                if is_anomaly:
                    cv2.imwrite(f"{DIRNAME}/result/bad/{filename}", self.img_org)
                else:
                    cv2.imwrite(f"{DIRNAME}/result/good/{filename}", self.img_org)
                
                self.make_anomalymap(s_map, filename)

                self.patch_lib.append(patch)


        self.img_num = cnt
        Save2Csv(save_data, HEADER, save_path=self.csvPath, save_mode="w")