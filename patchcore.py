from tqdm import tqdm
import torch
import timm
from dataset import SIZE
import numpy as np
# from detectron2.config import get_cfg
# from detectron2.modeling import build_model
# from detectron2 import model_zoo
# from detectron2.checkpoint import DetectionCheckpointer
import sys
from sklearn import random_projection
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from params import ARG
import math

class PatchCore(torch.nn.Module):
    def __init__(
            self, 
            f_coreset = 0.1, 
            backbone_name = "wide_resnet50_2", 
            coreset_eps = 0.50,         #もともと0.9
            pool_last = False, 
            pretrained = True
            ):
        
        super().__init__()

        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.backbone_name = backbone_name
        #2,3層の特徴を抽出
        self.out_indices = (2, 3)                                       
        # self.out_indices = (2, 3 ,4)                                       
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = self._initialize_feature_extractor(backbone_name, pretrained)    #ImageNetの場合
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # self.detectron2_model = self._detectron2_model()                                         #detectron2を用いる場合
        # self.detectron2_model = self.detectron2_model.to(self.device)
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
            out_indices=self.out_indices,
            features_only=True,
            pretrained=pretrained, 
        )
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        return feature_extractor

    # def _detectron2_model(self):
    #     cfg = get_cfg()
    #     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
    #     # cfg.merge_from_file(model_zoo.get_config_file("Base-RCNN-C4.yaml"))
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")
    #     # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Base-RCNN-C4.yaml")
    #     cfg.MODEL.DEVICE = "cpu"
    #     model = build_model(cfg)
    #     DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    #     model.eval()
    #     return model

    #ImageNetの場合
    def __call__(self, x: torch.Tensor):
        #勾配を計算しない　メモリ節約
        with torch.no_grad():
            #事前学習済みのモデルで特徴を抽出するモデル
            feature_maps = self.feature_extractor(x.to(self.device))        #入力画像から特徴マップを取得
            # print(feature_maps[0].shape)                                    #[1,512,16,16]
            # print(feature_maps[1].shape)                                    #[1,1024,8,8]
        feature_maps = [fmap.to("cpu") for fmap in feature_maps]            #特徴マップをCPUに戻す
        #マルチスケール特徴抽出＋グローバル特徴抽出
        if self.pool:
            #複数層の特徴マップ，プーリングされた特徴
            return feature_maps[:-1], self.pool(feature_maps[-1])           #最終層をプーリングして返す
        #こっちのルート
        else:
            return feature_maps
    
    #detectron2の場合
    # def __call__(self, x: torch.Tensor):
    #     #勾配を計算しない　メモリ節約
    #     feature_maps = []
    #     with torch.no_grad():
    #         #事前学習済みのモデルで特徴を抽出するモデル
    #         # feature_maps = self.feature_extractor(x.to(self.device))        #入力画像から特徴マップを取得
    #         # print(feature_maps[0].shape)                                    #[1,512,16,16]
    #         # print(feature_maps[1].shape)                                    #[1,1024,8,8]
    #         x = self.detectron2_model.backbone.stem(x.to(self.device))
    #         res2 = self.detectron2_model.backbone.res2(x)
    #         res3 = self.detectron2_model.backbone.res3(res2)
    #         res4 = self.detectron2_model.backbone.res4(res3)
    #         res5 = self.detectron2_model.backbone.res5(res4)
    #         # print(res2.shape)   #[1,512,32,32]
    #         # print(res3.shape)   #[1,512,16,16]
    #         # print(res4.shape)   #[1,1024,8,8]
    #         # print(res5.shape)   #[1,2048,8,8]
    #         feature_maps.append(res2)
    #         feature_maps.append(res3)
    #         # print(feature_maps[0].shape)
    #         # print(feature_maps[1].shape)
    #         # feature_maps = self.detectron2_model.backbone(x.to(self.device))
    #     feature_maps = [fmap.to("cpu") for fmap in feature_maps]            #特徴マップをCPUに戻す
    #     #マルチスケール特徴抽出＋グローバル特徴抽出
    #     if self.pool:
    #         #複数層の特徴マップ，プーリングされた特徴
    #         return feature_maps[:-1], self.pool(feature_maps[-1])           #最終層をプーリングして返す
    #     #こっちのルート
    #     else:
    #         return feature_maps

    def _self_attention(self, feature_maps):
        map2 = feature_maps[0]
        print(map2.shape)
        map3 = feature_maps[1]
        out_map2 = []
        out_map3 = []
        for i, map in enumerate(map2[0]):
            map = map.flatten()
            map_col = map.view(-1,1) #[256,1]
            map_row = map.view(1,-1) #[1,256]
            score2 = torch.matmul(map_col, map_row) / 256**0.5  #[256,256]
            attention2 = torch.nn.functional.softmax(score2)    #[256,256]
            out2 = torch.matmul(attention2, map)
            out2 = out2.view(16,16)
            out_map2.append(out2)
        output2 = torch.stack(out_map2, dim=0)
        output2 = output2.unsqueeze(0)
        print(output2.shape)
        for i, map in enumerate(map3[0]):
            map = map.flatten()
            map_col = map.view(-1,1)
            map_row = map.view(1,-1)
            score3 = torch.matmul(map_col, map_row) / 64**0.5
            attention3 = torch.nn.functional.softmax(score3)
            out3 = torch.matmul(attention3, map)
            out3 = out3.view(8,8)
            out_map3.append(out3)
        output3 = torch.stack(out_map3, dim=0)
        output3 = output3.unsqueeze(0)
        print(output3.shape)
        maps = [output2,output3]

        return maps

    def _position_encoding(self, feature_maps):
        d_model = 512
        map2 = feature_maps[0].squeeze(0)
        print(map2.shape)
        map3 = feature_maps[1].squeeze(0)
        pe = torch.zeros_like(map2)
        for y in range(16):
            for x in range(16):
                for i in range(0, d_model, 2):
                    pe[i+1,y,x] = math.sin(y / (10000 ** (2*i / d_model)))
                    pe[i+1,y,x] = math.cos(x / (10000 ** (2*i / d_model)))
        pe2 = map2 + pe
        pe2 = pe2.unsqueeze(0)
        
        pe = torch.zeros_like(map3)
        d_model = 1024
        for y in range(8):
            for x in range(8):
                for i in range(0, d_model, 2):
                    pe[i+1,y,x] = math.sin(y / (10000 ** (2*i / d_model)))
                    pe[i+1,y,x] = math.cos(x / (10000 ** (2*i / d_model)))
        pe3 = map3 + pe
        pe3 = pe.unsqueeze(0)
        print(pe3.shape, pe2.shape)
        out = [pe2, pe3]
        return out

    #学習    
    def fit(self, train_dl):
        self.patch_lib = self._create_patch_library(train_dl)
        # print(np.shape(self.patch_lib))                                                         #[502528,1536]
        self.reduce_patch_lib = self._reduce_patch_library()        #最終的なpatch_libは下の特徴量マップからピクセルの情報のみを削っている
        # print(f"reduce_patch_lib:{self.reduce_patch_lib.shape}\n{self.reduce_patch_lib}")   #[50252,1536]

    #特徴マップの抽出，patch_libに追加
    def _create_patch_library(self, train_dl):
        patch_lib = []
        for sample, _, _ in tqdm(train_dl, **self.get_tqdm_params()):
            # print(sample.shape)                                                    #[1,3,128,128]
            feature_maps = self(sample)                                             #__call__関数呼び出し 2,3層の特徴マップ
            feature_maps = self._position_encoding(feature_maps)                                         
            feature_maps = self._self_attention(feature_maps)
            # print(f"feature_maps[0]:{feature_maps[0].shape}\n{feature_maps}")      #[1,512,16,16]   
            # print(f"feature_maps[1]:{feature_maps[1].shape}\n{feature_maps}")      #[1,1024,8,8]
            resized_maps = self._resize_feature_maps(feature_maps)
            # print(f"resized_maps[0]:{resized_maps[0].shape}\n{resized_maps}")      #[1,512,16,16]
            # print(f"resized_maps[1]:{resized_maps[1].shape}\n{resized_maps}")      #[1,1024,16,16]
            patch = self._reshape_and_concatenate(resized_maps)                    #[1,1536,16,16]
            # print(f"resized_patch:{patch.shape}\n{patch}")                         #[256,1536]
            patch_lib.append(patch)        
        # print(np.shape(patch_lib))                                                  #[1963,256,1536]
        return torch.cat(patch_lib, 0)  #0次元で連結

    #リサイズ
    def _resize_feature_maps(self, feature_maps):
        if self.resize is None:
            largest_fmap_size = feature_maps[0].shape[-2:]     
            self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)     #すべての層の出力を最も大きな画像サイズに揃える，画像サイズの小さなより深い層の出力を引き伸ばしている
        return [self.resize(self.average(fmap)) for fmap in feature_maps]   

    def _reshape_and_concatenate(self, resized_maps):
        patch = torch.cat(resized_maps, 1)
        # print(f"patch:{patch.shape}\n{patch}")      #[1,1536,16,16]
        return patch.reshape(patch.shape[1], -1).T      #特徴マップの構造をピクセル単位でバラし，2次元構造へ変形

    def _reduce_patch_library(self):
        print(self.patch_lib.shape[0])  #
        if self.f_coreset < 1:
            self.coreset_idx = self._get_coreset_idx_randomp(   #Coresetが次元圧縮しても精度があんまり落ちないサンプリング手法らしい
                self.patch_lib,
                n=int(self.f_coreset * self.patch_lib.shape[0]),    #self.patch_lib.shape[0]=train_data(1963)*patch.shape[0](256)
                eps=self.coreset_eps,
            )
            self.patch_lib = self.patch_lib[self.coreset_idx]       #coreset_idxでピックアップされたピクセルを特徴量マップから取り出してpatch_libに保持
            x = self.patch_lib.to('cpu').detach().numpy().copy()
            np.save(f"/home/kby/mnt/hdd/coffee/PatchCore/npy_data/patch_lib_{ARG}.npy", x)        #モデルの保存先
            # np.save(f"../../PatchCore/npy_data/patch_lib_{ARG}.npy", x)        #モデルの保存先
            return x

    #スパース・ランダム射影による圧縮
    def _get_coreset_idx_randomp(self, z_lib, n = 1000, eps = 0.90, float16 = True, force_cpu = False):
        print(f"   ランダムプロジェクション。開始次元 = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)     #特徴量マップの要素数を削減
            z_lib = torch.tensor(transformer.fit_transform(z_lib))              #ここでメモリ大きすぎてクラッシュ
            z_lib = torch.tensor(z_lib, dtype=torch.float16 if float16 else torch.float32)
            print(f"   完了。変換後の次元 = {z_lib.shape}.")
            print(f"要素数: {z_lib.numel()} → メモリ目安: {z_lib.numel() * 4 / (1024**2):.2f} MB")
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
            min_distances[select_idx] = 0               #以降のイテレーションで再選択されないようにする
            coreset_idx.append(select_idx.to("cpu"))

        return torch.stack(coreset_idx)


    #推論
    def predict(self, sample):
        feature_maps = self(sample)
        feature_maps = self._position_encoding(feature_maps)  
        feature_maps = self._self_attention(feature_maps)           #SelfAttention
        resized_maps = self._resize_feature_maps(feature_maps)      #特徴マップリサイズ 学習時同様
        patch = self._reshape_and_concatenate(resized_maps)         #特徴マップ変形     [256,1536]
        s, s_map, min_val, m_test, m_star = self._compute_anomaly_scores(patch, feature_maps)
        return s, s_map, patch, min_val, m_test, m_star

    def _compute_anomaly_scores(self, patch, feature_maps):
        dist = torch.cdist(patch, self.patch_lib)                   #推論画像のpatchと学習時のpatch_libとの距離を計算 [256,50252]
        print(self.patch_lib.shape)
        print(dist.shape)
        min_val, min_idx = torch.min(dist, dim=1)                   #distの各行で最小値となる要素とそのインデックスを取得 minval.shape=[256]
        min_min_val, max_min_val = torch.min(min_val), torch.max(min_val)
        self.min_val = min_val
        
        # self.hist()
        s_star, s_idx = torch.max(min_val), torch.argmax(min_val)   #min_valの最大値，そのインテックスを取得
        #画像全体の異常度を計算
        w, m_test, m_star = self._reweight(patch, min_idx, s_star, s_idx)
        print(min_val.shape)
        # print(min_val.numpy())
        s = w * s_star
        # s_map = self._reweight_patch(patch, min_val, min_idx, s_star, s_idx)
        s_map = self._create_segmentation_map(min_val, feature_maps)
        # print(s_map)
        return s, s_map, min_val, m_test, m_star     #画像全体の異常度，ピクセルごとの異常マップ

    def _reweight(self, patch, min_idx, s_star, s_idx):
        m_test = patch[s_idx].unsqueeze(0)                          #s_starの算出に関わったpatch内のピクセルの特徴量
        print(patch.shape)  #[256,1536]
        print(m_test.shape) #[1,1536]                        
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)        #patch_lib内のピクセルの特徴量を取得
        w_dist = torch.cdist(m_star, self.patch_lib)                #m_starとpatch_libの各ピクセルとの距離を計算してw_distに
        print(w_dist.shape)
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)                #w_distが最も小さいk個のインデックスを取得
        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)   #得られたk個のピクセルとm_testの距離を計算，m_star_knn
        print(m_star_knn)
        print(s_star)
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        print(torch.sum(torch.exp(m_star_knn / D)))
        print(torch.exp(m_star_knn / D))
        w = 1 - (torch.exp(s_star / D) / torch.sum(torch.exp(m_star_knn / D)))
        return w , m_test, m_star      #重みの計算.m_star_knnが小さいほど重みは0に近づき，m_star_knnが大きいほど重みは1に近づく

    #ピクセル単位の異常度マップとする
    def _create_segmentation_map(self, min_val, feature_maps):      #patch.libの中から最も距離の近いピクセルを拾ってきたのに，距離が遠かったら以上の可能性大
        s_map = min_val.view(1, 1, *feature_maps[0].shape[-2:])     #[1,1,16,16]    
        # print(min_val)
        s_map = torch.nn.functional.interpolate(
            s_map, size=(SIZE, SIZE), mode='bilinear'
        )
        return s_map

    def _reweight_patch(self, patch, min_val, min_idx, s_star, s_idx):
        s_map = []
        for i, val in enumerate(min_val):
            m_test = patch[s_idx].unsqueeze(0)
            m_star = self.patch_lib[min_idx[i]].unsqueeze(0)
            w_dist = torch.cdist(m_star, self.patch_lib)
            _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)
            m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
            D = torch.sqrt(torch.tensor(patch.shape[1]))
            w = (1 - (torch.exp(s_star / D) / torch.sum(torch.exp(m_star_knn / D))))
            s_map.append(w * val)
        return s_map

    def standby(self):
        largest_fmap_size = torch.LongTensor([SIZE // 8, SIZE // 8])        #特徴マップの大きさに応じて指定
        self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
        self.patch_lib = np.load(f"/home/kby/mnt/hdd/coffee/PatchCore/npy_data/patch_lib_{ARG}.npy")
        # self.patch_lib = np.load(f"../../PatchCore/npy_data/patch_lib_{ARG}.npy")
        self.patch_lib = torch.from_numpy(self.patch_lib.astype(np.float32)).clone()
    
    def get_tqdm_params(self):
        return {
	        "file" : sys.stdout,
	        "bar_format" : "   {l_bar}{bar:10}{r_bar}{bar:-10b}",
        }
