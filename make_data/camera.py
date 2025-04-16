import cv2
import os, sys
import datetime
sys.path.append("/home/kby/MasterThesis/patchcore/")
from params import DATADIR

cap = cv2.VideoCapture(2)

index = 0
# キャプチャがオープンしている間続ける
while(cap.isOpened()):
    # フレームを読み込む
    ret, frame = cap.read()
    if ret == True:
        # フレームを表示
        cv2.imshow('Webcam Live', frame)

        # 'q'キーが押されたらループから抜ける
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        #aが押された場合写真を撮る
        if cv2.waitKey(60)&0xFF == 32:  #space
            now = datetime.datetime.now()
            index += 1
            filename = DATADIR + "testseijo/" + now.strftime("%m%d_%H%M_") + str(index) + ".png"
            cv2.imwrite(filename, frame)
    else:
        break

# キャプチャをリリースし、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()