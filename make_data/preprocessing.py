import cv2
import os, sys, glob
import numpy as np

#前処理
def noiseDel(im):
    img = np.copy(im)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    median = cv2.medianBlur(opening, 9)

    return median

def main():
    files = glob.glob("/home/kby/mnt/hdd/coffee/PatchCore/data/*.png")
    for i, fi in enumerate(files):
        f_name = os.path.split(fi)[1]
        img = cv2.imread(fi)
        binary_img = noiseDel(img)
        cv2.imwrite(f"/home/kby/mnt/hdd/coffee/PatchCore/data/2/{f_name}", binary_img)

        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        idxs = [i for i, h in enumerate(hierarchy[0]) if h[2] == -1]
        for idx in idxs:
            x, y, w, h = cv2.boundingRect(contours[idx])
            if (w * h) > 2500:
                roi = img[y-30:y+h+30, x-30:x+w+30]
                resized = cv2.resize(roi, (128,128))
                cv2.imwrite(f"/home/kby/mnt/hdd/coffee/PatchCore/data/resize/{f_name}", resized)

if __name__ == "__main__":
    main()