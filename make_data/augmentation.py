import glob
import cv2
import random
import os
import pathlib
import shutil

DIRNAME = "/home/kby/mnt/hdd/coffee/PatchCore/coffee"
TRAINDIR = f"{DIRNAME}/train_"

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

def OflineExpansion(path):
    files = glob.glob(path + '*.png')
    for f in files:
        filename = os.path.splitext(os.path.basename(f))[0]
        img = cv2.imread(f)
        cv2.imwrite(f"{TRAINDIR}/good/{filename}.png", img)
        argment = [j for j in range(20, 360, 20)]
        angle = random.sample(argment, 1)
        for i, ang in enumerate(angle):
            exp = rotate_img(img, ang)
            cv2.imwrite(f"{TRAINDIR}/good/{filename}_{str(ang)}.png", exp)

def rotate_img(img, angle):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    #回転行列
    rotation_M = cv2.getRotationMatrix2D(center, angle, 1)
    rotateImg = cv2.warpAffine(img, rotation_M, (width, height), borderMode=cv2.BORDER_TRANSPARENT, dst=img.copy())
    return rotateImg

def main():
    rootdir = "/home/kby/mnt/hdd/coffee/PatchCore/data/seijo/OutContext/"
    make_dir(f"{TRAINDIR}/good")
    OflineExpansion(rootdir)

if __name__ == "__main__":
    main()