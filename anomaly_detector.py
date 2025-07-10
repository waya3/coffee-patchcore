import os
import cv2
import numpy as np
from params import HEADER, RESULTDIR, RED, END, ARG
from dataset import SIZE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from patchcore import PatchCore
from utils import *
from pathlib import Path
import visualization as vi

class AnomalyDetector:
    def __init__(self, model:PatchCore, threshold):
        self.threshold = threshold
        self.model = model
        self.ave_time = 0.0
        self.img_org =[]
        self.img_num = 0
        self.patch_lib = []
        self.true = []
        self.pred = []
        self.scores = []
        self.FB = None

        self.flag = False
        self.good_num = 0
        self.bad_num = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        self.good_minval = []
        self.bad_minval = []
        self.good_pred_minval = []
        self.bad_pred_minval = []
        self.omote_minval = []
        self.omote_good_minval = []
        self.omote_bad_minval = []
        self.ura_good_minval = []
        self.ura_bad_minval = []
        self.ura_minval = []
        self.graph = []

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

    def run(self, dataloader, num):
        self.num = num
        self.csvPath = f"{RESULTDIR}/score_{str(self.num)}.csv"
        self.save_data = []
        self.fig, self.ax = plt.subplots()
        cnt = 0
        for images, paths, _ in dataloader:
            self.print_BeanType(paths)
            for i in range(images.size(0)):
                path = paths[i]
                self.dir_name = os.path.basename(os.path.dirname(path))
                self.img_org = cv2.imread(path)  #元画像
                self.img_org = cv2.resize(self.img_org, (SIZE,SIZE), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
                image_tensor = images[i].unsqueeze(0)
                filename = os.path.basename(path)
                #異常スコア，ピクセルごとの異常度，メモリバンク
                img_lvl_anom_score, s_map, patch, min_val, m_test, m_star = self.model.predict(image_tensor)
                Nmin_val = min_val.to("cpu").detach().numpy().copy()
                m_test = m_test.to("cpu").detach().numpy().copy()
                m_star = m_star.to("cpu").detach().numpy().copy()
                pstdev = calc_pstdev(Nmin_val)      #標準偏差
                score = round(img_lvl_anom_score.item(), 2)
                self.scores.append(score)
                #表ならTrue，裏ならFalse
                self.acquisition_omote(path, Nmin_val)  #ついでに表裏のminvalも可視化
                # vi.kde_minval(filename, Nmin_val, FB=self.FB)
                #matplotでmin_val描画
                graph_img = self.graph_paint(min_val, score, filename)
                self.img_score = cv2.putText(self.img_org, str(score), (0,10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
                # self.img_score = cv2.putText(self.img_score, str(pstdev), (0,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))    #標準偏差追加
                self.img_score = cv2.resize(self.img_score, (480,480))
                #連結
                img_scoreAndminval = cv2.hconcat([self.img_score, graph_img])

                #m_test, m_star
                graph_mtest = self.graph_paint_test_star(m_test, score, filename)
                graph_mstar = self.graph_paint_test_star(m_star, score, filename)
                img_mtest_mstar = cv2.hconcat([graph_mtest, graph_mstar])
                cv2.imwrite(f"{RESULTDIR}/m_test_star/{filename}", img_mtest_mstar)

                #異常検知
                self.anomaly_detect(filename, score, img_scoreAndminval, Nmin_val)

                #異常ヒートマップ作成
                self.make_anomalymap(s_map, score, filename)

                self.patch_lib.append(patch)
        
        #kde可視化
        # vi.kde_minval("True", self.good_minval, bad_minval=self.bad_minval)
        # vi.kde_minval("Pred", self.good_pred_minval, bad_minval=self.bad_pred_minval)
        # vi.kde_minval("Front", self.omote_good_minval, bad_minval=self.omote_bad_minval)
        # vi.kde_minval("Back", self.ura_good_minval, bad_minval=self.ura_bad_minval)

        # vi.kde_minval("TwoSides", self.omote_minval, bad_minval=self.ura_minval)
        self.img_num = self.good_num + self.bad_num
        #予測の評価
        self.eval()
        acc_header = ["Threshold", "Accuracy", "Precision", "Recall", "Specificity", "F1"]
        Save2Csv([self.opt_th, self.accuracy, self.precision, self.recall, self.specificity, self.Fmeasure], acc_header, save_path=self.csvPath, save_mode="w")
        Save2Csv(self.save_data, HEADER, save_path=self.csvPath, save_mode="a")

        #matplotアニメーション
        # ani = animation.ArtistAnimation(self.fig, self.graph)
        # ani.save(f"{RESULTDIR}/graph.mp4", writer="ffmpeg")

    def anomaly_detect(self, filename, score, img_scoreAndminval, min_val):
        #異常判定
        is_anomaly = score < self.threshold
        print(f"ファイル名:{filename}, 異常値スコア: {score}, 異常フラグ: {not is_anomaly}")
        self.save_data.append([filename, score, is_anomaly])
        if is_anomaly:              #正と予想
            cv2.imwrite(f"{RESULTDIR}/good/{filename}", self.img_score)
            self.good_pred_minval.append(min_val)
            #正常豆と欠点豆正解数を調査
            if self.dir_name == "good":  #正解は正
                self.good_num += 1
                self.tp += 1
                self.true.append(0)     #正常が0，異常が1
                self.pred.append(0)
                self.good_minval.append(min_val)
                if self.FB == "omote":
                    self.omote_good_minval.append(min_val)
                elif self.FB == "ura":
                    self.ura_good_minval.append(min_val)
            else:                   #正解は負
                self.bad_num += 1   
                self.fp += 1
                self.true.append(1)
                self.pred.append(0)
                self.bad_minval.append(min_val)
                if self.FB == "omote":
                    self.omote_bad_minval.append(min_val)
                elif self.FB == "ura":
                    self.ura_bad_minval.append(min_val)
        else:                       #負と予想
            cv2.imwrite(f"{RESULTDIR}/bad/{filename}", self.img_score)
            self.bad_pred_minval.append(min_val)
            #正常豆と欠点豆正解数を調査
            if self.dir_name == "good":  #正解は正
                self.good_num += 1
                self.fn += 1
                self.true.append(0)
                self.pred.append(1)
                self.good_minval.append(min_val)
                if self.FB == "omote":
                    self.omote_good_minval.append(min_val)
                elif self.FB == "ura":
                    self.ura_good_minval.append(min_val)
            else:                   #正解は負
                self.bad_num += 1
                self.tn += 1
                self.true.append(1)
                self.pred.append(1)
                self.bad_minval.append(min_val)
                if self.FB == "omote":
                    self.omote_bad_minval.append(min_val)
                elif self.FB == "ura":
                    self.ura_bad_minval.append(min_val)

        cv2.imwrite(f"{RESULTDIR}/test_all/{self.FB}/{filename}", img_scoreAndminval)

    #表裏取得,表裏の異常度も追加
    def acquisition_omote(self, pa, minval):
        path = Path(pa)
        dir = path.parent.parent.name
        if dir == "omote":
            self.FB = "omote"
            self.omote_minval.append(minval)
        else:
            self.FB = "ura"
            self.ura_minval.append(minval)

    def eval(self):
        self.accuracy = round((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn), 4)
        self.precision = round(self.tp / (self.tp + self.fp), 4)
        self.recall = round(self.tp / (self.tp + self.fn), 4)
        self.specificity = round(self.tn / (self.fp + self.tn), 4)
        self.Fmeasure = round(2 * self.precision * self.recall / (self.precision + self.recall), 4)

        #混同行列
        self.confusion()
        #ROC曲線
        self.opt_th = self.make_roc()
    
    def confusion(self):
        cm = confusion_matrix(self.true, self.pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(f"{RESULTDIR}/ConfusionMatrix_{ARG}_{str(self.num)}.jpg")
        plt.close()

    def make_roc(self):
        fpr, tpr, thresholds = roc_curve(self.true, self.scores)
        roc_auc = auc(fpr, tpr)
        opt_th = Youden(fpr, tpr, thresholds)
        print(f"最適なしきい値は{opt_th}です")
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1], [0,1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for PatchCore")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{RESULTDIR}/ROC_{ARG}_{str(self.num)}.jpg")
        plt.close()
        return opt_th


    def print_BeanType(self, paths):
        dir = os.path.basename(os.path.dirname(paths[0]))
        if dir == "good" and not self.flag:
            print(f"{RED}正常豆{END}")
            self.flag = True
        elif dir != "good" and self.flag:
            print(f"{RED}欠点豆{END}")
            self.flag = False

    def graph_paint(self, min_val, score, filename):
        self.ax.cla()
        im, = self.ax.plot(min_val, color="blue")
        self.ax.set_ylim(10, 60)
        text = self.ax.text(2,62, f"{self.dir_name},{self.FB}:{score}", fontsize=15, color="black")
        self.graph.append([im,text])
        plt.savefig(f"{RESULTDIR}/min_val/{filename}")
        dst = cv2.imread(f"{RESULTDIR}/min_val/{filename}")
        # dst = cv2.resize(dst, (171,128))
        return dst
        # plt.pause(.1)

    def graph_paint_test_star(self, m_value, score, filename):
        m_value = np.squeeze(m_value)
        self.ax.cla()
        im, = self.ax.plot(m_value, color="blue")
        self.ax.set_ylim(10, 30)
        text = self.ax.text(2,32, f"{self.dir_name},{self.FB}:{score}", fontsize=15, color="black")
        self.graph.append([im,text])
        plt.savefig(f"{RESULTDIR}/m_test_star/{filename}")
        dst = cv2.imread(f"{RESULTDIR}/m_test_star/{filename}")

        return dst


