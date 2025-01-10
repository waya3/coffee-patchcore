import os
import cv2
import numpy as np
from params import DIRNAME
from dataset import SIZE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch

class AnomalyDetector:
    def __init__(self, model, threshold):
        self.threshold = threshold
        self.model = model

    def run(self, dataloader):
        patch_lib = []
        for images, paths, _ in dataloader:
            for i in range(images.size(0)):
                path = paths[i]
                image_tensor = images[i].unsqueeze(0)
                filename = os.path.basename(path)
                #異常スコア，ピクセルごとの異常度，メモリバンク
                img_lvl_anom_score, s_map, patch = self.model.predict(image_tensor)
                
                score = img_lvl_anom_score.item()
                is_anomaly = score > self.threshold
                print(f"ファイル名:{filename}, 異常値スコア: {score}, 異常フラグ: {is_anomaly}")

                # ----------------------------------------------------- #
                #---            異常ヒートマップを作成したい              ---#
                # ----------------------------------------------------- #
                img_org = cv2.imread(path)  #元画像
                img_org = cv2.resize(img_org, (SIZE,SIZE), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
                s_map = s_map.squeeze(0)        #tensorのshapeが[1,1,224,224] -> [224,224]
                s_map = s_map.squeeze(0)
                s_map = s_map.detach().cpu().numpy()
                s_map = (s_map - s_map.min()) / (s_map.max() - s_map.min())

                #異常マップをカラー形式に変換
                heatmap = cv2.applyColorMap((s_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                #元画像と重ね合わせ
                overlay = cv2.addWeighted(img_org, 0.6, heatmap, 0.4, 0)
                #オーバーレイ画像を保存
                cv2.imwrite(DIRNAME + "/heatmap/" + filename, overlay)

        # ----------------------------------------------------- #
        #---              メモリバンクを可視化したい             ---#
        # ----------------------------------------------------- #
                patch_lib.append(patch)
        # memory_bank = torch.cat(patch_lib, 0)
        # print(patch_lib.shape)
        # memory_bank = patch_lib.detach().cpu().numpy()
        pca = PCA(n_components=2)
        memory_bank_2d = pca.fit_transform(patch_lib)
        print(memory_bank_2d.shape)
        
        plt.scatter(memory_bank_2d[:, 0], memory_bank_2d[:, 1], s=10, c="red", alpha=0.7)
        plt.title("Memoly Bank")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig(DIRNAME + "_memorybank_inf.jpg")