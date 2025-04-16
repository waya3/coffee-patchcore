import cv2
import os, sys, glob
import numpy as np

#前処理
def noiseDel(im):
    img = np.copy(im)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # nlmean = cv2.fastNlMeansDenoising(gray, h=20)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    median = cv2.medianBlur(opening, 9)

    return median

def MakeMask(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL)
    not_beabhsv = cv2.inRange(hsv, (80,0,15), (250,90,150))
    beanhsv = cv2.bitwise_not(not_beabhsv)
    beanhsv = cv2.medianBlur(beanhsv, 9)
    beanhsv = cv2.morphologyEx(beanhsv, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    beanhsv = cv2.cvtColor(beanhsv, cv2.COLOR_GRAY2BGR)

    return beanhsv

def main():
    rootdir = "/home/kby/mnt/hdd/coffee/PatchCore/data"
    for subdir, dirs, files in os.walk(rootdir):
        for _, fi in enumerate(files):
            f_name = os.path.split(fi)[1]
            img = cv2.imread(os.path.join(subdir, fi))
            binary_img = noiseDel(img)
            cv2.imwrite(f"{subdir}/mask/{f_name}", binary_img)

            contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            idxs = [i for i, h in enumerate(hierarchy[0]) if h[2] == -1]
            for idx in idxs:
                x, y, w, h = cv2.boundingRect(contours[idx])
                if (w * h) > 2500:
                    roi = img[y-30:y+h+30, x-30:x+w+30]
                    resized = cv2.resize(roi, (128,128))
                    cv2.imwrite(f"{subdir}/resize/{f_name}", resized)
                    mask = MakeMask(resized)
                    cv2.imwrite(f"{subdir}/mask/{f_name}", mask)
                    OutContext = cv2.bitwise_and(mask, resized)
                    cv2.imwrite(f"{subdir}/OutContext/{f_name}", OutContext)

if __name__ == "__main__":
    main()