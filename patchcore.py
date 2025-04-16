from tqdm import tqdm
import torch
import timm
from dataset import SIZE
import numpy as np
import sys
from sklearn import random_projection

class PatchCore(torch.nn.Module):
    def __init__(
            self, 
            f_coreset = 0.1, 
            backbone_name = "resnet18", 
            coreset_eps = 0.90, 
            pool_last = False, 
            pretrained = True
            ):
        
        super().__init__()

        self.feature_extractor = self._initialize_feature_extractor(backbone_name, pretrained)
        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.backbone_name = backbone_name
        self.out_indices = (2, 3)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.n_reweight = 3
        self.patch_lib = []
        self.reduce_patch_lib = []
        self.resize = None

    def _initialize_feature_extractor(self, backbone_name, pretrained):
        feature_extractor = timm.create_model(
            backbone_name,
            out_indices=(2, 3),
            features_only=True,
            pretrained=pretrained, 
        )
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        return feature_extractor

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            feature_maps = self.feature_extractor(x.to(self.device))
        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        if self.pool:
            return feature_maps[:-1], self.pool(feature_maps[-1])
        else:
            return feature_maps


    #学習    
    def fit(self, train_dl):
        self.patch_lib = self._create_patch_library(train_dl)
        self.reduce_patch_lib = self._reduce_patch_library()        #最終的なpatch_libは下の特徴量マップからピクセルの情報のみを削っている

    #特徴マップの抽出
    def _create_patch_library(self, train_dl):
        patch_lib = []
        for sample, _, _ in tqdm(train_dl, **self.get_tqdm_params()):
            feature_maps = self(sample)
            resized_maps = self._resize_feature_maps(feature_maps)
            patch = self._reshape_and_concatenate(resized_maps)
            patch_lib.append(patch)
        return torch.cat(patch_lib, 0)

    #リサイズ
    def _resize_feature_maps(self, feature_maps):
        if self.resize is None:
            largest_fmap_size = feature_maps[0].shape[-2:]      #すべての層の出力を最も大きな画像サイズに揃える
            self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)     #画像サイズの小さなより深い層の出力を引き伸ばしている
        return [self.resize(self.average(fmap)) for fmap in feature_maps]   

    def _reshape_and_concatenate(self, resized_maps):
        patch = torch.cat(resized_maps, 1)
        return patch.reshape(patch.shape[1], -1).T      #特徴マップの構造をピクセル単位でバラし，2次元構造へ変形

    def _reduce_patch_library(self):
        if self.f_coreset < 1:
            self.coreset_idx = self._get_coreset_idx_randomp(   #Coresetが次元圧縮しても精度があんまり落ちないサンプリング手法らしい
                self.patch_lib,
                n=int(self.f_coreset * self.patch_lib.shape[0]),
                eps=self.coreset_eps,
            )
            self.patch_lib = self.patch_lib[self.coreset_idx]       #coreset_idxでピックアップされたピクセルを特徴量マップから取り出してpatch_libに保持
            x = self.patch_lib.to('cpu').detach().numpy().copy()
            np.save("./npy_data/patch_lib.npy", x)
            return x

    #スパース・ランダム射影による圧縮
    def _get_coreset_idx_randomp(self, z_lib, n = 1000, eps = 0.90, float16 = True, force_cpu = False):
        print(f"   ランダムプロジェクション。開始次元 = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)     #特徴量マップの要素数雨を削減
            z_lib = torch.tensor(transformer.fit_transform(z_lib))
            print(f"   完了。変換後の次元 = {z_lib.shape}.")
        except ValueError:
            print( "   ベクトルをプロジェクションできませんでした。`eps`を増やしてください。")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx+1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to("cuda")
            z_lib = z_lib.to("cuda")
            min_distances = min_distances.to("cuda")

        #ピクセルごとに他ピクセルとのユークリッド距離を計算
        for _ in tqdm(range(n-1), **self.get_tqdm_params()):
            distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True) 
            min_distances = torch.minimum(distances, min_distances)
            select_idx = torch.argmax(min_distances)
            last_item = z_lib[select_idx:select_idx+1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))

        return torch.stack(coreset_idx)


    #推論
    def predict(self, sample):
        feature_maps = self(sample)
        resized_maps = self._resize_feature_maps(feature_maps)      #特徴マップリサイズ 学習時同様
        patch = self._reshape_and_concatenate(resized_maps)         #特徴マップ変形
        s, s_map = self._compute_anomaly_scores(patch, feature_maps)
        return s, s_map, patch

    def _compute_anomaly_scores(self, patch, feature_maps):
        dist = torch.cdist(patch, self.patch_lib)                   #推論画像のpatchと学習時のpatch_libとの距離を計算
        min_val, min_idx = torch.min(dist, dim=1)                   #distの各行で最小値となる要素とそのインデックスを取得
        s_star, s_idx = torch.max(min_val), torch.argmax(min_val)   #min_valの最大値，そのインテックスを取得
        #画像全体の異常度を計算
        w = self._reweight(patch, min_idx, s_star, s_idx)
        s = w * s_star
        s_map = self._create_segmentation_map(min_val, feature_maps)
        return s, s_map     #画像全体の異常度，ピクセルごとの異常マップ

    def _reweight(self, patch, min_idx, s_star, s_idx):
        m_test = patch[s_idx].unsqueeze(0)                          
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)        #s_starの算出に関わったpatch内のピクセルの特徴量とpatch_lib内のピクセルの特徴量を取得
        w_dist = torch.cdist(m_star, self.patch_lib)                #m_starとpatch_libの各ピクセルとの距離を計算してw_distに
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)                #w_distが最も小さいk個のインデックスを取得
        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)   #得られたk個のピクセルとm_testの距離を計算，m_star_knn
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        return 1 - (torch.exp(s_star / D) / torch.sum(torch.exp(m_star_knn / D)))       #重みの計算.m_star_knnが小さいほど重みは0に近づき，m_star_knnが大きいほど重みは1に近づく

    #ピクセル単位の異常度マップとする
    def _create_segmentation_map(self, min_val, feature_maps):      #patch.libの中から最も距離の近いピクセルを拾ってきたのに，距離が遠かったら以上の可能性大
        s_map = min_val.view(1, 1, *feature_maps[0].shape[-2:])
        s_map = torch.nn.functional.interpolate(
            s_map, size=(SIZE, SIZE), mode='bilinear'
        )
        return s_map



    def standby(self):
        largest_fmap_size = torch.LongTensor([SIZE // 8, SIZE // 8])
        self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
        self.patch_lib = np.load("./npy_data/patch_lib.npy")
        self.patch_lib = torch.from_numpy(self.patch_lib.astype(np.float32)).clone()
    
    def get_tqdm_params(self):
        return {
	        "file" : sys.stdout,
	        "bar_format" : "   {l_bar}{bar:10}{r_bar}{bar:-10b}",
        }