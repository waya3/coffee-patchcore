from patchcore import PatchCore
from dataset import Dataset
import torch
from params import MODELNAME, DATASET, BACKBONE, DIRNAME
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    model = PatchCore(backbone_name=BACKBONE, pretrained=True)
    train_ds, _ = Dataset().get_dataloaders()
    
    # ----------------------------------------------------- #
    #---     1.学習済みモデルに正常画像を通して特徴マップ抽出   ---#
    #---     2.特徴量マップをリサイズ                       ---#
    #---     3.画像の中で特徴的なピクセルをピックアップして保持 ---#
    # ----------------------------------------------------- #
    model.fit(train_ds)     
    torch.save(model.state_dict(), MODELNAME)
    print("モデル保存先：", MODELNAME)

    print(model.patch_lib.shape)
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
