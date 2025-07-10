import pathlib
import shutil
import csv
import cv2
import numpy as np

def delete_dir(path):
    shutil.rmtree(path)
    #ファイル一個ずつ削除する
    # check_dir = pathlib.Path(path)
    # for file in check_dir.iterdir():
    #     if file.is_file():
    #         file.unlink

def make_dir(path):
    delete_dir(path)
    pathlib.Path(path).mkdir()

def Save2Csv(save_data, header, save_path : str, save_mode = 'w'):
    # 保存するデータの格納形式を判定
    if isinstance(save_data, list) and isinstance(save_data[0], list):
    # if isinstance(save_data, list):
        is2D = True
    elif isinstance(save_data, list) and not isinstance(save_data[0], list):
        is2D = False
    else:
        return -1
    if is2D:
        if save_mode == "w":
            with open(save_path, save_mode, encoding="utf-8", errors="replace") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i, data in enumerate(save_data):
                    writer.writerow(data)
        elif save_mode == "a":
            with open(save_path, save_mode, encoding="utf-8", errors="replace") as f:
                writer = csv.writer(f)
                writer.writerow("\n")
                writer.writerow(header)
                for i, data in enumerate(save_data):
                    writer.writerow([i] + data)
    else:
        with open(save_path, save_mode, encoding="utf-8", errors="replace") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(save_data)

    return 0

def calc_pstdev(data):
    data = np.array(data)
    data_flat = data.flatten()
    return round(np.std(data_flat), 3)

#AUCから最適なしきい値を決定
def Youden(fpr, tpr, th):
    youden_j = tpr -fpr
    optimal_idx = np.argmax(youden_j)
    optimal_th = th[optimal_idx]
    return optimal_th