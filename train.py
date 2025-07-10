from patchcore import PatchCore
from dataset import Dataset
import torch
import time
from params import MODELNAME, DATASET, BACKBONE, DIRNAME
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from params import ARG

def main():
    model = PatchCore(backbone_name=BACKBONE, pretrained=True)
    train_ds, _ = Dataset().get_dataloaders()
    
    # ----------------------------------------------------- #
    #---     1.学習済みモデルに正常画像を通して特徴マップ抽出   ---#
    #---     2.特徴量マップをリサイズ                       ---#
    #---     3.画像の中で特徴的なピクセルをピックアップして保持 ---#
    # ----------------------------------------------------- #
    start = time.perf_counter()
    model.fit(train_ds)     
    end = time.perf_counter()
    torch.save(model.state_dict(), MODELNAME)
    print("モデル保存先：", f"{MODELNAME}, ../../PatchCore/npy_data/patch_lib_{ARG}.npy")
    print(model.patch_lib.shape)    #[50252,2]
    print(f"実行時間:" + "{:2f}".format(end - start) + "秒")
    
    #メモリバンク可視化したい
    memory_bank = model.patch_lib
    memory_bank = memory_bank.detach().cpu().numpy()

    pca = PCA(n_components=2)
    memory_bank_2d = pca.fit_transform(memory_bank)
    print(memory_bank_2d.shape)
    
    plt.scatter(memory_bank_2d[:, 0], memory_bank_2d[:, 1], s=10, c="blue", alpha=0.7)
    plt.title("Memoly Bank")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(f"{DIRNAME}/memolybank.jpg")

if __name__ == "__main__":
    main()
