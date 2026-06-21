# Imports and configs
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.base import clone
from lightgbm import LGBMClassifier
import random
from tqdm import tqdm
from koolbox import Trainer
import numpy as np
import pandas as pd
import itertools
import warnings
import optuna
import joblib
from joblib import Parallel, delayed
import glob
import gc
import json
from collections import defaultdict
import polars as pl
import os
from datetime import datetime
import shutil
# ============================================================================
# 初始化与配置
# ============================================================================
class CFG:
    train_path = "/kaggle/input/MABe-mouse-behavior-detection/train.csv"
    test_path = "/kaggle/input/MABe-mouse-behavior-detection/test.csv"
    train_annotation_path = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation"
    train_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/train_tracking"
    test_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/test_tracking"

    # train_path = "./MABe-mouse-behavior-detection/train.csv"
    # test_path = "./MABe-mouse-behavior-detection/test.csv"
    # train_annotation_path = "./MABe-mouse-behavior-detection/train_annotation"
    # train_tracking_path = "./MABe-mouse-behavior-detection/train_tracking"
    # test_tracking_path = "./MABe-mouse-behavior-detection/test_tracking"

    model_path = "/kaggle/input"
    model_name = "fold555"

    SEED = 3407

    # mode = "validate"
    mode = "submit"

    n_splits = 5
    cv = StratifiedGroupKFold(n_splits)

    # model = XGBClassifier(
    #     verbosity=0,
    #     random_state=42,
    #     n_estimators=250,
    #     learning_rate=0.08,
    #     max_depth=6,
    #     min_child_weight=5,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     tree_method='gpu_hist',  # 使用GPU加速
    #     device='cuda:0',
    # )


    model = LGBMClassifier(
        verbosity=-1,             # 静默模式
        random_state=42,
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,              # 限制深度
        num_leaves=31,            # 关键参数: LightGBM 是 leaf-wise，配合 max_depth=6，建议 31 (2^5) 到 63 (2^6-1)
        min_child_weight=10,       # 对应 min_sum_hessian_in_leaf，控制叶子节点最小权重和
        subsample=0.8,
        subsample_freq=1,         # 关键参数: LightGBM 需要设置 freq > 0 才能启用 subsample (bagging)
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        # device='gpu',             # 启用 GPU
        # gpu_device_id=1,          # 指定 GPU ID
        n_jobs=-1                 # 并行数
    )



def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
set_global_seeds(CFG.SEED)

# Global constants
# 需要丢弃的身体部位（冗余或噪声较大的关键点）
drop_body_parts = [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright',
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright',
    'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint']


META_NUM_COLS = [
    "frames_per_second",
    "pix_per_cm_approx",
    "video_width",
    "video_height",
    "arena_width_cm",
    "arena_height_cm",
    "n_mice",
]

META_CAT_COLS = [
    "lab_id",
    "arena_shape",
    "arena_type",
    "tracking_method",
]


META_CAT_ENCODING = "onehot"  # {"onehot", "label"} 


META_CAT_CATEGORIES = {}   # col -> List[str]
META_CAT_KNOWN_SET = {}    # col -> Set[str] (fast membership)
META_CAT_SPECIAL_UNK = "__UNK__"
META_CAT_SPECIAL_NA = "__NA__"



META_CAT_ENCODERS = {}
META_CAT_UNK = {}


def meta_to_features(meta: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=meta.index)

    # 数值型
    for col in META_NUM_COLS:
        if col in meta.columns:
            feats[col] = pd.to_numeric(meta[col], errors="coerce").astype(np.float32)

    # === New: derived numeric meta features (robust + domain-general) ===
    eps = np.float32(1e-6)

    # fps transforms
    if "frames_per_second" in feats.columns:
        fps = feats["frames_per_second"].astype(np.float32)
        fps_pos = fps.clip(lower=0)
        feats["fps_log1p"] = np.log1p(fps_pos).astype(np.float32)
        feats["sec_per_frame"] = (1.0 / (fps_pos + eps)).astype(np.float32)

    # pixel <-> cm scale transforms
    if "pix_per_cm_approx" in feats.columns:
        ppc = feats["pix_per_cm_approx"].astype(np.float32)
        ppc_pos = ppc.clip(lower=0)
        feats["cm_per_pix_approx"] = (1.0 / (ppc_pos + eps)).astype(np.float32)
        feats["log_pix_per_cm"] = np.log1p(ppc_pos).astype(np.float32)

    # video geometry (pixels)
    if "video_width" in feats.columns and "video_height" in feats.columns:
        vw = feats["video_width"].astype(np.float32)
        vh = feats["video_height"].astype(np.float32)
        feats["video_aspect"] = (vw / (vh + eps)).astype(np.float32)
        area_v = (vw * vh).astype(np.float32)
        feats["video_area_px"] = area_v
        feats["video_area_log1p"] = np.log1p(area_v.clip(lower=0)).astype(np.float32)
        feats["video_diag_px"] = np.sqrt((vw * vw + vh * vh).clip(lower=0)).astype(np.float32)

    # arena geometry (cm)
    if "arena_width_cm" in feats.columns and "arena_height_cm" in feats.columns:
        aw = feats["arena_width_cm"].astype(np.float32)
        ah = feats["arena_height_cm"].astype(np.float32)
        aw_pos = aw.clip(lower=0)
        ah_pos = ah.clip(lower=0)

        feats["arena_aspect"] = (aw_pos / (ah_pos + eps)).astype(np.float32)
        area_a = (aw_pos * ah_pos).astype(np.float32)
        feats["arena_area_cm2"] = area_a
        feats["arena_area_log1p"] = np.log1p(area_a).astype(np.float32)
        feats["arena_perimeter_cm"] = (2.0 * (aw_pos + ah_pos)).astype(np.float32)
        feats["arena_diag_cm"] = np.sqrt((aw_pos * aw_pos + ah_pos * ah_pos).clip(lower=0)).astype(np.float32)

    # density / per-mouse normalization
    if "n_mice" in feats.columns:
        nm = feats["n_mice"].astype(np.float32).clip(lower=0)
        feats["n_mice_log1p"] = np.log1p(nm).astype(np.float32)
        feats["n_mice_sq"] = (nm * nm).astype(np.float32)

        if "arena_area_cm2" in feats.columns:
            feats["mice_density"] = (nm / (feats["arena_area_cm2"] + eps)).astype(np.float32)
            feats["arena_area_per_mouse"] = (feats["arena_area_cm2"] / (nm + eps)).astype(np.float32)
    # === End new code ===

    # 类别型
    if META_CAT_ENCODING == "onehot":
        for col in META_CAT_COLS:
            if col not in meta.columns:
                continue
            if col not in META_CAT_CATEGORIES:
                # vocab 未构建（理论上 main() 会用 train 构建），这里保守跳过
                continue

            known = META_CAT_CATEGORIES[col]
            known_set = META_CAT_KNOWN_SET[col]

            # 统一转字符串；缺失 -> __NA__；未知 -> __UNK__
            s = meta[col].astype("string")
            s = s.fillna(META_CAT_SPECIAL_NA)
            s = s.where(s.isin(known_set), META_CAT_SPECIAL_UNK)

            # 固定 categories，确保 get_dummies 输出列集合稳定（即便某些类别本批次没出现）
            cat = pd.Categorical(
                s,
                categories=known + [META_CAT_SPECIAL_UNK, META_CAT_SPECIAL_NA],
                ordered=False,
            )

            dummies = pd.get_dummies(
                cat,
                prefix=col,
                prefix_sep="=",
                dtype=np.uint8,  # 省内存；下游 concat 后再统一 float32 也行
            )

            feats = pd.concat([feats, dummies], axis=1)

    else:
    
        for col in META_CAT_COLS:
            if col in meta.columns and col in META_CAT_ENCODERS:
                enc = META_CAT_ENCODERS[col]
                unk = META_CAT_UNK[col]
                feats[col + "_idx"] = meta[col].map(lambda v: enc.get(v, unk)).astype(np.int32)

    return feats





# ============================================================================
# Missing-value handling (short-gap fill) + missingness-as-signal features
# ============================================================================

MISSING_MAX_GAP_SEC = 0.25  # 只填补 <= 0.25s 的连续缺失
MISSING_EPS = 1e-6


def _fill_small_gaps(df: pd.DataFrame, fps: float, max_gap_sec: float = MISSING_MAX_GAP_SEC) -> pd.DataFrame:
    """只填补短缺失（<= max_gap_sec 秒），长缺失保留 NaN。"""
    try:
        fps = float(fps) if fps is not None and not pd.isna(fps) else 30.0
    except Exception:
        fps = 30.0
    limit = max(1, int(round(max_gap_sec * fps)))

    out = df.copy()
    # 内部空洞插值（不外推）
    out = out.interpolate(method="linear", axis=0, limit=limit, limit_area="inside")
    # 极短边界缺失（仍受 limit 限制）
    out = out.ffill(limit=limit).bfill(limit=limit)
    return out


def _part_ok(mouse_df: pd.DataFrame) -> pd.DataFrame:
    """
    mouse_df columns: MultiIndex (bodypart, x/y)
    return: frames x bodypart (bool), True 表示该部位 x/y 都非空
    """
    # pandas: groupby(axis=1) deprecated -> use transpose
    return mouse_df.notna().T.groupby(level=0).all().T


def _nose_proxy_ok(part_ok: pd.DataFrame) -> pd.Series:
    idx = part_ok.index
    if "nose" in part_ok.columns:
        return part_ok["nose"]
    if "head" in part_ok.columns:
        return part_ok["head"]
    if "ear_left" in part_ok.columns and "ear_right" in part_ok.columns:
        return part_ok["ear_left"] & part_ok["ear_right"]
    return pd.Series(False, index=idx)


def _center_proxy_ok(part_ok: pd.DataFrame) -> pd.Series:
    idx = part_ok.index
    if "body_center" in part_ok.columns:
        return part_ok["body_center"]
    if "neck" in part_ok.columns:
        return part_ok["neck"]
    if "nose" in part_ok.columns and "tail_base" in part_ok.columns:
        return part_ok["nose"] & part_ok["tail_base"]
    if "head" in part_ok.columns and "tail_base" in part_ok.columns:
        return part_ok["head"] & part_ok["tail_base"]
    if "ear_left" in part_ok.columns and "ear_right" in part_ok.columns:
        return part_ok["ear_left"] & part_ok["ear_right"]
    return pd.Series(False, index=idx)


def _streak_len(mask: pd.Series) -> pd.Series:
    """
    mask=True 表示处于“缺失状态”，返回当前连续缺失长度（否则为0）
    """
    m = mask.fillna(False).astype(bool)
    grp = (~m).cumsum()
    out = m.groupby(grp).cumcount() + 1
    out = out.where(m, 0)
    return out


def add_missingness_features_single(X: pd.DataFrame, single_mouse: pd.DataFrame, fps: float, section: int) -> pd.DataFrame:
    part_ok = _part_ok(single_mouse)  # frames x bodypart
    ok_cnt = part_ok.sum(axis=1).astype(np.float32)
    total = float(part_ok.shape[1] if part_ok.shape[1] > 0 else 1)
    miss_ratio = (1.0 - ok_cnt / (total + MISSING_EPS)).astype(np.float32)

    nose_ok = _nose_proxy_ok(part_ok).astype(np.float32)
    center_ok = _center_proxy_ok(part_ok).astype(np.float32)

    X["kp_ok_cnt"] = ok_cnt
    X["kp_missing_ratio"] = miss_ratio
    X["nose_proxy_ok"] = nose_ok
    X["center_proxy_ok"] = center_ok

    # 连续缺失段长度（遮挡/打斗/追逐常见）
    X["nose_missing_streak"] = _streak_len(nose_ok < 0.5).astype(np.float32)
    X["center_missing_streak"] = _streak_len(center_ok < 0.5).astype(np.float32)

    # 多尺度 rolling（轻量、对 domain shift 很稳）
    window = [20, 40, 60, 80] if section == 9 else [15, 30, 60, 120]
    for w in window:
        ws = _scale(w, fps)
        roll = dict(min_periods=max(1, ws // 5), center=True)
        X[f"kp_missing_m{w}"] = miss_ratio.rolling(ws, **roll).mean().astype(np.float32)
        X[f"nose_ok_m{w}"] = nose_ok.rolling(ws, **roll).mean().astype(np.float32)
        X[f"center_ok_m{w}"] = center_ok.rolling(ws, **roll).mean().astype(np.float32)

    return X


def add_missingness_features_pair(X: pd.DataFrame, mouse_pair: pd.DataFrame, fps: float) -> pd.DataFrame:
    part_ok_A = _part_ok(mouse_pair["A"])
    part_ok_B = _part_ok(mouse_pair["B"])

    totalA = float(part_ok_A.shape[1] if part_ok_A.shape[1] > 0 else 1)
    totalB = float(part_ok_B.shape[1] if part_ok_B.shape[1] > 0 else 1)

    A_ok_cnt = part_ok_A.sum(axis=1).astype(np.float32)
    B_ok_cnt = part_ok_B.sum(axis=1).astype(np.float32)
    A_miss = (1.0 - A_ok_cnt / (totalA + MISSING_EPS)).astype(np.float32)
    B_miss = (1.0 - B_ok_cnt / (totalB + MISSING_EPS)).astype(np.float32)

    X["A_kp_missing_ratio"] = A_miss
    X["B_kp_missing_ratio"] = B_miss
    X["AB_kp_missing_ratio_mean"] = (0.5 * (A_miss + B_miss)).astype(np.float32)
    X["AB_kp_missing_ratio_diff"] = (A_miss - B_miss).astype(np.float32)

    A_nose_ok = _nose_proxy_ok(part_ok_A).astype(np.float32)
    B_nose_ok = _nose_proxy_ok(part_ok_B).astype(np.float32)
    A_center_ok = _center_proxy_ok(part_ok_A).astype(np.float32)
    B_center_ok = _center_proxy_ok(part_ok_B).astype(np.float32)

    X["A_nose_proxy_ok"] = A_nose_ok
    X["B_nose_proxy_ok"] = B_nose_ok
    X["A_center_proxy_ok"] = A_center_ok
    X["B_center_proxy_ok"] = B_center_ok
    X["AB_both_center_ok"] = (A_center_ok * B_center_ok).astype(np.float32)

    X["A_nose_missing_streak"] = _streak_len(A_nose_ok < 0.5).astype(np.float32)
    X["B_nose_missing_streak"] = _streak_len(B_nose_ok < 0.5).astype(np.float32)

    for w in [15, 30, 60, 120]:
        ws = _scale(w, fps)
        roll = dict(min_periods=max(1, ws // 5), center=True)
        X[f"AB_miss_m{w}"] = X["AB_kp_missing_ratio_mean"].rolling(ws, **roll).mean().astype(np.float32)

    return X



# ============================================================================
# Creating solution data
# ============================================================================
def create_solution_df(dataset):
    """
    创建验证用的标准答案DataFrame

    输入:
        dataset: pd.DataFrame - 训练数据集的元信息

    输出:
        pd.DataFrame - 包含所有标注数据的DataFrame，列包括：
            - lab_id: 实验室ID
            - video_id: 视频ID
            - agent_id: 执行动作的老鼠ID（格式：'mouse1', 'mouse2'等）
            - target_id: 目标老鼠ID或'self'
            - action: 行为类型
            - start_frame, stop_frame: 行为的起止帧
            - behaviors_labeled: 该视频标注的行为列表

    作用:
        整合分散的标注文件，生成一个标准化的 “标准答案” 数据集，
        用于验证模式下计算模型性能指标
    """
    solution = []
    #从dataset中逐行提取lab_id和video_id
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):

        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):  #跳过lab_id以MABe22开头的视频
            continue

        video_id = row['video_id']
        path = f"{CFG.train_annotation_path}/{lab_id}/{video_id}.parquet"  #找到该轮lab_id和video_id对应数据
        try:
            annot = pd.read_parquet(path)
        except FileNotFoundError:
            continue

        # 为标注数据添加元信息，包括lab_id、video_id、behaviors_labeled（从元信息中获取）
        annot['lab_id'] = lab_id
        annot['video_id'] = video_id
        annot['behaviors_labeled'] = row['behaviors_labeled']
        #target_id和agent_id转换为mouse1、mouse2等格式
        annot['target_id'] = np.where(annot.target_id != annot.agent_id, annot['target_id'].apply(lambda s: f"mouse{s}"), 'self') 
        annot['agent_id'] = annot['agent_id'].apply(lambda s: f"mouse{s}")
        solution.append(annot) #将每个标注数据添加到solution列表中

    solution = pd.concat(solution) #将solution列表中的所有标注数据合并成一个DataFrame

    return solution #返回solution

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    """
    生成器函数：逐个生成单只老鼠或老鼠对的数据

    输入:
        dataset: pd.DataFrame - 数据集元信息
        traintest: str - 'train'或'test'，指定数据类型
        traintest_directory: str, optional - tracking数据目录路径
        generate_single: bool - 是否生成单只老鼠数据
        generate_pair: bool - 是否生成老鼠对数据

    输出 (yield):
        对于训练数据:
            ('single', data, meta, label) 或 ('pair', data, meta, label)
            - data: pd.DataFrame - 老鼠的坐标数据
            - meta: pd.DataFrame - 元数据（video_id, agent_id, target_id, video_frame）
            - label: pd.DataFrame - 标签数据

        对于测试数据:
            ('single', data, meta, actions) 或 ('pair', data, meta, actions)
            - data: pd.DataFrame - 老鼠的坐标数据
            - meta: pd.DataFrame - 元数据
            - actions: np.array - 需要预测的行为列表

    作用:
        从tracking文件中读取老鼠的关键点坐标数据，
        并根据需要生成单只老鼠或老鼠对的数据及对应标签
    """
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
        # traintest_directory = f"dataset/MABe-mouse-behavior-detection/{traintest}_tracking"
    # 逐行读取视频信息，提取lab_id和video_id
    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        if lab_id.startswith('MABe22') or type(row.behaviors_labeled) != str:  #跳过lab_id以MABe22开头的视频或behaviors_labeled不是字符串的视频
            continue

        scalar_meta = {}
        for col in META_NUM_COLS + META_CAT_COLS:
            if col in dataset.columns:
                scalar_meta[col] = row[col]


        video_id = row.video_id
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"  #找到该轮lab_id和video_id对应数据
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:  #如果该视频有5个以上的身体部位，则丢弃bodypart中包含drop_body_parts的行
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])

        

        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid /= row.pix_per_cm_approx

        # === Best: fill only short gaps (keeps long occlusions as NaN) ===
        fps_row = row.frames_per_second if "frames_per_second" in dataset.columns else 30.0
        pvid = _fill_small_gaps(pvid, fps=fps_row, max_gap_sec=MISSING_MAX_GAP_SEC)
        # === End best code ===


        # 将视频元信息中的行为描述字符串解析并转换为结构化的 DataFrame
        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])

        # 训练数据时，读取标注数据
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))  #找到该轮lab_id和video_id对应标注数据
            except FileNotFoundError:
                continue  #如果标注数据不存在，跳过该视频

        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")  # 从视频的所有行为中筛选出单鼠行为
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):  # 遍历所有单鼠行为
                try:
                    mouse_id = int(mouse_id_str[-1])  # 提取老鼠ID
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)  #提取该老鼠的所有自身行为类型
                    single_mouse = pvid.loc[:, mouse_id]
                    assert len(single_mouse) == len(pvid)
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index,
                        **scalar_meta,   # <-- 新增: 注入视频级 meta
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label  #类型+特征+元数据+标签
                    else:
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions  #类型+特征+元数据+要预测的行为类型
                except KeyError:
                    pass

        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2): # int8
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    assert len(mouse_pair) == len(pvid)
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index,
                        **scalar_meta,   # <-- 新增: 注入视频级 meta
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions

# ============================================================================
# Transforming coordinates
# ============================================================================
def safe_rolling(series, window, func, min_periods=None):
    """
    安全的滚动窗口计算函数，避免窗口过小导致的计算失败

    输入:
        series: pd.Series - 需要进行滚动计算的时间序列数据
        window: int - 滚动窗口大小（帧数）
        func: callable - 应用于窗口的函数
        min_periods: int, optional - 最小有效数据点数，默认为窗口大小的1/4

    输出:
        pd.Series - 滚动计算后的结果序列

    作用:
        对时间序列数据进行滚动窗口计算，自动处理边界情况
    """
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def _scale(n_frames_at_30fps, fps, ref=30.0):
    """
    根据实际帧率缩放窗口大小

    输入:
        n_frames_at_30fps: int - 在30fps下的帧数
        fps: float - 实际视频帧率
        ref: float - 参考帧率，默认30.0

    输出:
        int - 缩放后的帧数（至少为1）

    作用:
        将基于30fps设计的窗口大小转换为适应实际帧率的窗口大小
        例如：如果实际fps=60，则窗口大小会翻倍
    """
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    """
    根据实际帧率缩放窗口大小（保留正负符号）

    输入:
        n_frames_at_30fps: int - 在30fps下的帧数（可以为负数）
        fps: float - 实际视频帧率
        ref: float - 参考帧率，默认30.0

    输出:
        int - 缩放后的帧数（保留符号，至少为±1）

    作用:
        类似_scale，但保留符号，用于时间偏移量的缩放
        例如：-10帧在60fps下会变成-20帧
    """
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    """
    从元数据中提取视频帧率

    输入:
        meta_df: pd.DataFrame - 包含视频元数据的DataFrame
        fallback_lookup: dict - video_id到fps的映射字典（备用）
        default_fps: float - 默认帧率，默认30.0

    输出:
        float - 视频的帧率

    作用:
        按优先级获取视频帧率：
        1. 从meta_df的frames_per_second列
        2. 从fallback_lookup字典
        3. 使用默认值30.0
    """
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

# ============================================================================
# extract features
# ============================================================================
# 单鼠特征
# 曲率和转向相关的特征
def add_curvature_features(X, center_x, center_y, fps, section):
    new_features = {}
    
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)

    window = [25, 50, 75] if section == 9 else [15, 30, 60, 120]
    for w in window:
        ws = _scale(w, fps)
        new_features[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 5)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    window = [30] if section == 9 else [15, 30, 60, 120]
    for w in window:
        ws = _scale(w, fps)
        new_features[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 5)).sum()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

# 多尺度速度特征
def add_multiscale_features(X, center_x, center_y, fps, section):
    new_features = {}
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [20, 40, 60, 80] if section == 9 else [15, 30, 60, 120]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            new_features[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            new_features[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()
            new_features[f'sp_q25_{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).quantile(0.25)
            # 为了计算 IQR，我们需要先计算 q75
            q75 = speed.rolling(ws, min_periods=max(1, ws // 4)).quantile(0.75)
            new_features[f'sp_q75_{scale}'] = q75
            if f'sp_q25_{scale}' in new_features:
                new_features[f'sp_iqr{scale}'] = q75 - new_features[f'sp_q25_{scale}']

    # 计算 sp_ratio 需要先有 sp_m
    if len(scales) >= 2:
        k1 = f'sp_m{scales[0]}'
        k2 = f'sp_m{scales[-1]}'
        # 如果在本轮计算中生成了这两个特征，或者它们已经在 X 中（虽然这个函数通常是第一次生成它们）
        # 我们优先从 new_features 获取
        val1 = new_features.get(k1, X.get(k1))
        val2 = new_features.get(k2, X.get(k2))
        
        if val1 is not None and val2 is not None:
            new_features['sp_ratio'] = val1 / (val2 + 1e-6)

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

# 运动状态特征
def add_state_features(X, center_x, center_y, fps, section):
    new_features = {}
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        # FIX: speed_ma 已经是 cm/s（上面乘过 fps），分箱阈值也必须是 cm/s 的常数，不能再乘 fps
        bins_cms = [-np.inf, 0.5, 2.0, 5.0, np.inf]  # cm/s
        speed_states = pd.cut(speed_ma, bins=bins_cms, labels=[0, 1, 2, 3]).astype(float)

        window = [20, 40, 60, 80] if section == 9 else [15, 30, 60, 120]
        for w in window:
            ws = _scale(w, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    new_features[f's{state}_{w}'] = (
                        (speed_states == state).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 5)).mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                new_features[f'trans_{w}'] = state_changes.rolling(ws, min_periods=max(1, ws // 5)).sum()
    except Exception:
        pass

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

# 长时程特征
def add_longrange_features(X, center_x, center_y, fps):
    new_features = {}
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            new_features[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            new_features[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    for span in [30, 60, 120]:
        s = _scale(span, fps)
        new_features[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        new_features[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            new_features[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_posture_stability_features(X, single_mouse, available_body_parts, fps):
    new_features = {}
    if 'ear_left' in available_body_parts and 'ear_right' in available_body_parts:
        ear_left = single_mouse['ear_left']
        ear_right = single_mouse['ear_right']

        # 耳朵对称性
        ear_mid_x = (ear_left['x'] + ear_right['x']) / 2
        ear_mid_y = (ear_left['y'] + ear_right['y']) / 2

        # 耳朵中点稳定性
        ear_mid_std_x = ear_mid_x.rolling(_scale(30, fps),
                                          min_periods=_scale(5, fps)).std()
        ear_mid_std_y = ear_mid_y.rolling(_scale(30, fps),
                                          min_periods=_scale(5, fps)).std()
        new_features['ear_mid_std'] = np.sqrt(ear_mid_std_x ** 2 + ear_mid_std_y ** 2)

    # 身体部位相对位置稳定性
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    if nose is not None and 'tail_base' in available_body_parts:
        nose_tail_vec_x = nose['x'] - single_mouse['tail_base']['x']
        nose_tail_vec_y = nose['y'] - single_mouse['tail_base']['y']
        nose_tail_length = np.sqrt(nose_tail_vec_x ** 2 + nose_tail_vec_y ** 2)

        new_features['nose_tail_length_std'] = nose_tail_length.rolling(
            _scale(30, fps), min_periods=_scale(5, fps)).std()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X


def add_head_elevation_features(X, single_mouse, available_body_parts, fps, section):
    if section <= 3 or section == 8:
        return X
    
    new_features = {}
    center = _get_substitute_body_center(single_mouse, available_body_parts)
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    if center is not None and nose is not None and 'tail_base' in available_body_parts:
        head_elevation = nose['y'] - center['y']  # 头部相对身体中心的高度
        body_angle_vertical = (nose['y'] - single_mouse['tail_base']['y'])  # 身体垂直度
        for w in [10, 20, 30]:
            ws = _scale(w, fps)
            new_features[f'head_elev_mean_{w}'] = head_elevation.rolling(ws).mean()
            new_features[f'head_elev_std_{w}'] = head_elevation.rolling(ws).std()
            new_features[f'body_vert_{w}'] = body_angle_vertical.rolling(ws).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X


def add_body_width_features(X, single_mouse, available_body_parts, fps, section):
    if section == 3:
        return X
    
    new_features = {}
    if 'lateral_left' in available_body_parts and 'lateral_right' in available_body_parts:
        body_width_x = np.abs(single_mouse['lateral_left']['x'] - single_mouse['lateral_right']['x'])
        body_width_y = np.abs(single_mouse['lateral_left']['y'] - single_mouse['lateral_right']['y'])
        body_width = np.sqrt(body_width_x**2 + body_width_y**2)
        for w in [15, 30, 60, 120]:
            ws = _scale(w, fps)
            new_features[f'body_width_mean_{w}'] = body_width.rolling(ws).mean()
            new_features[f'body_width_std_{w}'] = body_width.rolling(ws).std()
            new_features[f'body_width_change_{w}'] = body_width.diff().rolling(ws).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_pose_shape_features(X, single_mouse, available_body_parts, fps, section):
    new_features = {}
    center = _get_substitute_body_center(single_mouse, available_body_parts)
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    # 躯干-头部-尾部三角形特征
    if center is not None and nose is not None and 'tail_base' in available_body_parts:
        nose_x = nose['x']
        nose_y = nose['y']
        center_x = center['x']
        center_y = center['y']
        tail_x = single_mouse['tail_base']['x']
        tail_y = single_mouse['tail_base']['y']

        # 三角形面积（使用叉积公式）
        triangle_area = pd.Series(0.5 * np.abs(
            nose_x * (center_y - tail_y) +
            center_x * (tail_y - nose_y) +
            tail_x * (nose_y - center_y)
        ), index=single_mouse.index)
        new_features['tri_area'] = triangle_area

        # 三角形边长
        nose_center_dist = pd.Series(np.sqrt((nose_x - center_x)**2 + (nose_y - center_y)**2), index=single_mouse.index)
        center_tail_dist = pd.Series(np.sqrt((center_x - tail_x)**2 + (center_y - tail_y)**2), index=single_mouse.index)
        nose_tail_dist = pd.Series(np.sqrt((nose_x - tail_x)**2 + (nose_y - tail_y)**2), index=single_mouse.index)

        # 边长比例（前半身 vs 后半身）
        new_features['tri_front_back_ratio'] = nose_center_dist / (center_tail_dist + 1e-6)

        # 三角形的"紧凑度"：面积 / 周长^2（类似圆形度）
        perimeter = nose_center_dist + center_tail_dist + nose_tail_dist
        new_features['tri_compactness'] = triangle_area / (perimeter**2 + 1e-6)

        window = [15, 30] if section == 9 else [15, 30, 60, 120]
        # 多尺度统计
        for w in window:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 4), center=True)
            new_features[f'tri_area_m{w}'] = triangle_area.rolling(ws, **roll_params).mean()
            new_features[f'tri_area_s{w}'] = triangle_area.rolling(ws, **roll_params).std()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_head_body_decoupled_features(X, single_mouse, available_body_parts, fps, section):
    """
    添加头部与身体解耦的运动特征
    (优化：批量添加列以避免 DataFrame 碎片化)
    """
    new_features = {} # 使用字典收集新特征

    center = _get_substitute_body_center(single_mouse, available_body_parts)
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    if center is not None and nose is not None:
        # 1. 头部相对身体的速度
        nose_x = nose['x']
        nose_y = nose['y']
        center_x = center['x']
        center_y = center['y']

        # 头部相对于体心的位置
        rel_nose_x = nose_x - center_x
        rel_nose_y = nose_y - center_y

        # 头部相对速度（相对位置的变化率）
        rel_nose_vx = rel_nose_x.diff() * fps
        rel_nose_vy = rel_nose_y.diff() * fps
        head_rel_speed = pd.Series(np.sqrt(rel_nose_vx**2 + rel_nose_vy**2), index=single_mouse.index)

        new_features['head_rel_speed'] = head_rel_speed

        # 体心速度
        center_vx = center_x.diff() * fps
        center_vy = center_y.diff() * fps
        body_speed = pd.Series(np.sqrt(center_vx**2 + center_vy**2), index=single_mouse.index)

        # 头部速度 / 体心速度比值
        new_features['head_body_speed_ratio'] = head_rel_speed / (body_speed + 1e-6)

        window = [15, 30, 60] if section == 9 else [15, 30, 60, 120]
        # 多尺度统计
        for w in window:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 4), center=True)
            new_features[f'head_rel_sp_m{w}'] = head_rel_speed.rolling(ws, **roll_params).mean()
            new_features[f'head_rel_sp_s{w}'] = head_rel_speed.rolling(ws, **roll_params).std()

        # 2. 头部转动速度（方向变化率）
        # 头部方向向量（体心指向鼻子）
        head_dir_x = nose_x - center_x
        head_dir_y = nose_y - center_y

        # 头部朝向角度
        head_angle = pd.Series(np.arctan2(head_dir_y, head_dir_x), index=single_mouse.index)

        # 角度变化（注意处理-π到π的跳变）
        angle_diff = head_angle.diff()
        angle_diff = pd.Series(np.where(angle_diff > np.pi, angle_diff - 2*np.pi, angle_diff), index=single_mouse.index)
        angle_diff = pd.Series(np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff), index=single_mouse.index)

        # 头部转动速度（弧度/秒）
        head_turn_speed = pd.Series(np.abs(angle_diff) * fps, index=single_mouse.index)
        new_features['head_turn_speed'] = head_turn_speed

        window = [15, 30] if section == 9 else [15, 30, 60, 120]
        # 多尺度统计
        for w in window:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 4), center=True)
            new_features[f'head_turn_m{w}'] = head_turn_speed.rolling(ws, **roll_params).mean()
            new_features[f'head_turn_s{w}'] = head_turn_speed.rolling(ws, **roll_params).std()
            # 头部方向变化的总和（累积转动）
            new_features[f'head_turn_sum{w}'] = head_turn_speed.rolling(ws, **roll_params).sum()

    # 3. 耳朵与鼻子/体心的几何关系
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']) and center is not None and nose is not None:
        ear_left_x = single_mouse['ear_left']['x']
        ear_left_y = single_mouse['ear_left']['y']
        ear_right_x = single_mouse['ear_right']['x']
        ear_right_y = single_mouse['ear_right']['y']
        center_x = center['x']
        center_y = center['y']
        nose_x = nose['x']
        nose_y = nose['y']

        # 耳朵中点
        ear_mid_x = (ear_left_x + ear_right_x) / 2
        ear_mid_y = (ear_left_y + ear_right_y) / 2

        # 耳朵中点到体心的距离
        ear_center_dist = pd.Series(np.sqrt((ear_mid_x - center_x)**2 + (ear_mid_y - center_y)**2), index=single_mouse.index)
        new_features['ear_center_dist'] = ear_center_dist

        # 耳朵中点到鼻子的距离
        ear_nose_dist = pd.Series(np.sqrt((ear_mid_x - nose_x)**2 + (ear_mid_y - nose_y)**2), index=single_mouse.index)
        new_features['ear_nose_dist'] = ear_nose_dist

        # 耳朵-鼻子-体心形成的角度
        vec_nose_ear_x = ear_mid_x - nose_x
        vec_nose_ear_y = ear_mid_y - nose_y
        vec_nose_center_x = center_x - nose_x
        vec_nose_center_y = center_y - nose_y

        dot_product = vec_nose_ear_x * vec_nose_center_x + vec_nose_ear_y * vec_nose_center_y
        norm_product = pd.Series(np.sqrt(vec_nose_ear_x**2 + vec_nose_ear_y**2) *
                       np.sqrt(vec_nose_center_x**2 + vec_nose_center_y**2) + 1e-6, index=single_mouse.index)
        new_features['ear_nose_center_angle'] = dot_product / norm_product

        window = [30, 60] if section == 9 else [15, 30, 60, 120]
        # 多尺度统计
        for w in window:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 4), center=True)
            new_features[f'ear_center_m{w}'] = ear_center_dist.rolling(ws, **roll_params).mean()
            new_features[f'ear_center_s{w}'] = ear_center_dist.rolling(ws, **roll_params).std()

    # 最后一次性合并所有新特征
    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X


def add_body_axis_motion_features(X, single_mouse, available_body_parts, fps, section):
    new_features = {}  # 收集新特征

    center = _get_substitute_body_center(single_mouse, available_body_parts)
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    # 如果缺少必要的锚点，跳过
    if nose is None or center is None or 'tail_base' not in available_body_parts:
        return X

    # 1. 计算身体朝向向量（尾部 -> 头部）
    ori_x = nose['x'] - single_mouse['tail_base']['x']
    ori_y = nose['y'] - single_mouse['tail_base']['y']
    ori_norm = np.sqrt(ori_x ** 2 + ori_y ** 2) + 1e-6

    # 归一化朝向向量
    ori_x_unit = ori_x / ori_norm
    ori_y_unit = ori_y / ori_norm

    # 2. 计算运动中心的速度向量
    vx = center['x'].diff() * fps  # cm/s
    vy = center['y'].diff() * fps  # cm/s

    # 3. 将速度分解到体轴坐标系
    # v_forward: 沿身体朝向的速度分量（正=前进，负=后退）
    v_forward = vx * ori_x_unit + vy * ori_y_unit
    # v_lateral: 垂直于身体朝向的速度分量（正=向右，负=向左）
    # 法向量：(-ori_y_unit, ori_x_unit) 指向右侧
    v_lateral = vx * (-ori_y_unit) + vy * ori_x_unit

    # 将结果转换为 Series 并设置索引
    v_forward = pd.Series(v_forward.values, index=single_mouse.index)
    v_lateral = pd.Series(v_lateral.values, index=single_mouse.index)

    # 4. 基础特征
    new_features['v_forward'] = v_forward
    new_features['v_lateral'] = v_lateral
    new_features['v_forward_abs'] = np.abs(v_forward)
    new_features['v_lateral_abs'] = np.abs(v_lateral)

    # 侧向 vs 前向速度比
    lat_over_fwd = np.abs(v_lateral) / (np.abs(v_forward) + 1e-6)
    new_features['lat_over_fwd'] = lat_over_fwd

    window = [15, 30, 60] if section == 9 else [15, 30, 60, 120]
    # 5. 多尺度统计
    for w in window:
        ws = _scale(w, fps)
        roll_params = dict(min_periods=max(1, ws // 4), center=True)

        # 前向速度统计
        new_features[f'v_fwd_m{w}'] = v_forward.rolling(ws, **roll_params).mean()
        new_features[f'v_fwd_s{w}'] = v_forward.rolling(ws, **roll_params).std()
        new_features[f'v_fwd_abs_m{w}'] = np.abs(v_forward).rolling(ws, **roll_params).mean()

        # 侧向速度统计
        new_features[f'v_lat_m{w}'] = v_lateral.rolling(ws, **roll_params).mean()
        new_features[f'v_lat_s{w}'] = v_lateral.rolling(ws, **roll_params).std()
        new_features[f'v_lat_abs_m{w}'] = np.abs(v_lateral).rolling(ws, **roll_params).mean()

        # 侧向/前向比值统计
        new_features[f'lat_fwd_ratio_m{w}'] = lat_over_fwd.rolling(ws, **roll_params).mean()

    # 6. 行为模式指标
    # 前进时间占比（v_forward > 阈值）
    fwd_threshold = 1.0  # cm/s
    is_forward = (v_forward > fwd_threshold).astype(float)
    is_backward = (v_forward < -fwd_threshold).astype(float)
    is_lateral = (np.abs(v_lateral) > np.abs(v_forward)).astype(float)

    window = [30, 60] if section == 9 else [15, 30, 60, 120]
    for w in window:
        ws = _scale(w, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)

        # 前进/后退/侧移时间占比
        new_features[f'forward_ratio_{w}'] = is_forward.rolling(ws, **roll_params).mean()
        new_features[f'backward_ratio_{w}'] = is_backward.rolling(ws, **roll_params).mean()
        new_features[f'lateral_dom_ratio_{w}'] = is_lateral.rolling(ws, **roll_params).mean()

    # 7. 滑移角（身体朝向与运动方向的偏差）
    # 运动方向角
    theta_move = np.arctan2(vy, vx)
    # 身体朝向角
    theta_body = np.arctan2(ori_y, ori_x)

    # 滑移角（-π 到 π）
    slip_angle = theta_move - theta_body
    # 归一化到 -π 到 π
    slip_angle = np.arctan2(np.sin(slip_angle), np.cos(slip_angle))
    slip_angle = pd.Series(slip_angle.values, index=single_mouse.index)

    new_features['slip_angle'] = slip_angle
    new_features['slip_angle_abs'] = np.abs(slip_angle)

    window = [15, 30, 60] if section == 9 else [15, 30, 60, 120]
    # 滑移角统计
    for w in window:
        ws = _scale(w, fps)
        roll_params = dict(min_periods=max(1, ws // 4), center=True)

        new_features[f'slip_m{w}'] = slip_angle.rolling(ws, **roll_params).mean()
        new_features[f'slip_s{w}'] = slip_angle.rolling(ws, **roll_params).std()
        new_features[f'slip_abs_m{w}'] = np.abs(slip_angle).rolling(ws, **roll_params).mean()

    # 滑移角对齐程度（身体朝向与运动方向一致的时间占比）
    align_threshold = np.pi / 6  # 30度
    is_aligned = (np.abs(slip_angle) < align_threshold).astype(float)

    window = [30, 60] if section == 9 else [15, 30, 60, 120]
    for w in window:
        ws = _scale(w, fps)
        new_features[f'slip_align_ratio_{w}'] = is_aligned.rolling(
            ws, min_periods=max(1, ws // 5), center=True
        ).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def add_high_freq_micromotion_features(X, single_mouse, available_body_parts, fps, section):
    new_features = {} # 收集新特征

    # 获取前端锚点（nose代理）和运动中心（body_center代理）
    nose = _get_substitute_nose(single_mouse, available_body_parts)
    center = _get_substitute_body_center(single_mouse, available_body_parts)

    # 1. 高频头部微运动特征
    if nose is not None:
        nose_x = nose['x']
        nose_y = nose['y']

        # 头部瞬时速度（帧间位移）
        head_vx = nose_x.diff() * fps
        head_vy = nose_y.diff() * fps
        head_speed = pd.Series(np.sqrt(head_vx ** 2 + head_vy ** 2), index=single_mouse.index)

        # --- 1.1 短窗口高频头部运动统计 ---
        # 使用非常短的窗口捕捉高频微运动
        for w in [3, 5, 7]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)

            # 头部速度的短窗口统计
            new_features[f'head_hf_m{w}'] = head_speed.rolling(ws, **roll_params).mean()
            new_features[f'head_hf_s{w}'] = head_speed.rolling(ws, **roll_params).std()
            new_features[f'head_hf_max{w}'] = head_speed.rolling(ws, **roll_params).max()

        # --- 1.2 头部加速度（二阶微分，捕捉颤动/抖动） ---
        head_ax = head_vx.diff() * fps
        head_ay = head_vy.diff() * fps
        head_accel = pd.Series(np.sqrt(head_ax ** 2 + head_ay ** 2), index=single_mouse.index)
        new_features['head_accel'] = head_accel

        for w in [5, 10]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)
            new_features[f'head_accel_m{w}'] = head_accel.rolling(ws, **roll_params).mean()
            new_features[f'head_accel_s{w}'] = head_accel.rolling(ws, **roll_params).std()

        # --- 1.3 头部抖动指数（jitter index）---
        # 连续帧之间速度方向的变化（检测快速来回抖动）
        head_dir = pd.Series(np.arctan2(head_vy, head_vx + 1e-8), index=single_mouse.index)
        head_dir_change = head_dir.diff()
        # 处理角度跳变
        head_dir_change = pd.Series(
            np.where(head_dir_change > np.pi, head_dir_change - 2 * np.pi, head_dir_change),
            index=single_mouse.index
        )
        head_dir_change = pd.Series(
            np.where(head_dir_change < -np.pi, head_dir_change + 2 * np.pi, head_dir_change),
            index=single_mouse.index
        )
        head_jitter = pd.Series(np.abs(head_dir_change), index=single_mouse.index)
        new_features['head_jitter'] = head_jitter

        for w in [5, 10, 15]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)
            new_features[f'head_jitter_m{w}'] = head_jitter.rolling(ws, **roll_params).mean()
            # 方向变化总和（累积抖动量）
            new_features[f'head_jitter_sum{w}'] = head_jitter.rolling(ws, **roll_params).sum()

    # 2. 尾巴高频微运动特征
    if 'tail_base' in available_body_parts:
        tail_x = single_mouse['tail_base']['x']
        tail_y = single_mouse['tail_base']['y']

        # 尾巴瞬时速度
        tail_vx = tail_x.diff() * fps
        tail_vy = tail_y.diff() * fps
        tail_speed = pd.Series(np.sqrt(tail_vx ** 2 + tail_vy ** 2), index=single_mouse.index)

        # --- 2.1 短窗口尾巴高频运动统计 ---
        for w in [3, 5, 7]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)
            new_features[f'tail_hf_m{w}'] = tail_speed.rolling(ws, **roll_params).mean()
            new_features[f'tail_hf_s{w}'] = tail_speed.rolling(ws, **roll_params).std()

        # --- 2.2 尾巴加速度 ---
        tail_ax = tail_vx.diff() * fps
        tail_ay = tail_vy.diff() * fps
        tail_accel = pd.Series(np.sqrt(tail_ax ** 2 + tail_ay ** 2), index=single_mouse.index)
        new_features['tail_accel'] = tail_accel

        for w in [5, 10]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)
            new_features[f'tail_accel_m{w}'] = tail_accel.rolling(ws, **roll_params).mean()

        # --- 2.3 尾巴抖动指数 ---
        tail_dir = pd.Series(np.arctan2(tail_vy, tail_vx + 1e-8), index=single_mouse.index)
        tail_dir_change = tail_dir.diff()
        tail_dir_change = pd.Series(
            np.where(tail_dir_change > np.pi, tail_dir_change - 2 * np.pi, tail_dir_change),
            index=single_mouse.index
        )
        tail_dir_change = pd.Series(
            np.where(tail_dir_change < -np.pi, tail_dir_change + 2 * np.pi, tail_dir_change),
            index=single_mouse.index
        )
        tail_jitter = pd.Series(np.abs(tail_dir_change), index=single_mouse.index)
        new_features['tail_jitter'] = tail_jitter

        for w in [5, 10]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 2), center=True)
            new_features[f'tail_jitter_m{w}'] = tail_jitter.rolling(ws, **roll_params).mean()

        # --- 2.4 尾巴相对于体心的局部运动 ---
        if center is not None:
            center_x = center['x']
            center_y = center['y']
            body_vx = center_x.diff() * fps
            body_vy = center_y.diff() * fps

            rel_tail_vx = tail_vx - body_vx
            rel_tail_vy = tail_vy - body_vy
            rel_tail_speed = pd.Series(np.sqrt(rel_tail_vx ** 2 + rel_tail_vy ** 2), index=single_mouse.index)
            new_features['tail_rel_speed'] = rel_tail_speed

            for w in [10, 20]:
                ws = _scale(w, fps)
                roll_params = dict(min_periods=max(1, ws // 3), center=True)
                new_features[f'tail_rel_m{w}'] = rel_tail_speed.rolling(ws, **roll_params).mean()

            # 静止时的尾巴活动（如果已计算is_body_still）
            if 'is_body_still' in X.columns:
                is_body_still = X['is_body_still']
                still_tail_activity = is_body_still * rel_tail_speed
                new_features['still_tail_activity'] = still_tail_activity

                for w in [10, 20]:
                    ws = _scale(w, fps)
                    roll_params = dict(min_periods=max(1, ws // 3), center=True)
                    new_features[f'still_tail_act_m{w}'] = still_tail_activity.rolling(ws, **roll_params).mean()

    # 4. 头尾协调/独立运动特征
    if nose is not None and 'tail_base' in available_body_parts:
        nose_x = nose['x']
        nose_y = nose['y']
        tail_x = single_mouse['tail_base']['x']
        tail_y = single_mouse['tail_base']['y']

        head_vx = nose_x.diff() * fps
        head_vy = nose_y.diff() * fps
        tail_vx = tail_x.diff() * fps
        tail_vy = tail_y.diff() * fps

        head_speed = pd.Series(np.sqrt(head_vx ** 2 + head_vy ** 2), index=single_mouse.index)
        tail_speed = pd.Series(np.sqrt(tail_vx ** 2 + tail_vy ** 2), index=single_mouse.index)

        # --- 4.1 头尾速度比 ---
        head_tail_speed_ratio = head_speed / (tail_speed + 1e-6)
        new_features['head_tail_speed_ratio'] = head_tail_speed_ratio

        for w in [10, 20]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 3), center=True)
            new_features[f'head_tail_ratio_m{w}'] = head_tail_speed_ratio.rolling(ws, **roll_params).mean()

        # --- 4.2 头尾运动方向一致性 ---
        # 点积归一化：1表示同向运动，-1表示反向运动，0表示垂直
        head_speed_safe = head_speed + 1e-6
        tail_speed_safe = tail_speed + 1e-6
        head_tail_dir_dot = (head_vx * tail_vx + head_vy * tail_vy) / (head_speed_safe * tail_speed_safe)
        new_features['head_tail_dir_consistency'] = head_tail_dir_dot

        for w in [10, 20, 30]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 3), center=True)
            new_features[f'head_tail_dir_m{w}'] = head_tail_dir_dot.rolling(ws, **roll_params).mean()

        # --- 4.3 头尾独立运动指数 ---
        # 低一致性 + 高速度 = 独立运动（可能是不同行为阶段）
        head_tail_independence = (1 - head_tail_dir_dot.abs()) * (head_speed + tail_speed)
        new_features['head_tail_independence'] = head_tail_independence

        for w in [15, 30]:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 3), center=True)
            new_features[f'head_tail_indep_m{w}'] = head_tail_independence.rolling(ws, **roll_params).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def add_arena_spatial_features(X, single_mouse, available_body_parts, fps, section, video_id):
    global arena_data
    # 获取场地信息
    try:
        arena_info = arena_data.loc[video_id]
        arena_width = arena_info['arena_width_cm']
        arena_height = arena_info['arena_height_cm']
        arena_shape = arena_info['arena_shape']
    except (KeyError, TypeError):
        # 如果找不到场地信息，跳过
        return X

    # 检查场地尺寸是否有效
    if pd.isna(arena_width) or pd.isna(arena_height) or arena_width <= 0 or arena_height <= 0:
        return X

    # 判断场地形状：圆形 vs 矩形（非圆形都按矩形处理）
    is_circular = False
    if pd.notna(arena_shape):
        shape_lower = str(arena_shape).lower()
        if 'circle' in shape_lower or 'circular' in shape_lower or 'round' in shape_lower:
            is_circular = True

    # 获取运动中心代理点
    center = _get_substitute_body_center(single_mouse, available_body_parts)
    if center is None:
        return X
    center_x = center['x']
    center_y = center['y']

    # ============================================
    # 使用数据范围估算 Arena 边界位置
    # ============================================
    # 使用整个视频序列的坐标范围估算 arena 边界
    x_min = center_x.min()
    x_max = center_x.max()
    y_min = center_y.min()
    y_max = center_y.max()

    # Arena 中心（基于实际数据范围）
    arena_center_x = (x_min + x_max) / 2
    arena_center_y = (y_min + y_max) / 2

    # 定义边界区域宽度（距离墙 15% 的区域视为边界区域）
    border_ratio = 0.15
    border_width = min(arena_width, arena_height) * border_ratio

    if is_circular:
        # 圆形场地的特征计算
        radius = min(arena_width, arena_height) / 2

        # 到场地中心的距离
        dist_to_center = pd.Series(
            np.sqrt((center_x - arena_center_x) ** 2 + (center_y - arena_center_y) ** 2),
            index=single_mouse.index
        )
        X['dist_to_arena_center'] = dist_to_center

        # 归一化到中心的距离（0=中心，1=边界）
        X['dist_to_center_norm'] = dist_to_center / (radius + 1e-6)

        # 到边界（圆周）的距离
        dist_to_nearest_wall = (radius - dist_to_center).clip(lower=0)
        X['dist_to_nearest_wall'] = dist_to_nearest_wall

        # 归一化到墙距离（0=贴墙，1=中心）
        X['dist_to_wall_norm'] = dist_to_nearest_wall / (radius + 1e-6)

        # 空间区域二值指示特征
        wall_threshold = radius * border_ratio  # 靠墙阈值
        central_radius = radius * 0.3  # 中央 30% 区域

        is_near_wall = (dist_to_nearest_wall < wall_threshold).astype(float)
        is_in_center = (dist_to_center < central_radius).astype(float)

        X['is_near_wall'] = is_near_wall
        X['is_in_center'] = is_in_center
        # 圆形场地没有角落
        X['is_in_corner'] = pd.Series(0.0, index=single_mouse.index)

    else:
        # 矩形场地的特征计算
        # 到四面墙的距离（基于数据范围估算边界）
        dist_to_left = center_x - x_min
        dist_to_right = x_max - center_x
        dist_to_bottom = center_y - y_min
        dist_to_top = y_max - center_y

        X['dist_to_left_wall'] = dist_to_left
        X['dist_to_right_wall'] = dist_to_right
        X['dist_to_bottom_wall'] = dist_to_bottom
        X['dist_to_top_wall'] = dist_to_top

        # 到最近墙的距离
        dist_to_nearest_wall = pd.concat(
            [dist_to_left, dist_to_right, dist_to_bottom, dist_to_top], axis=1
        ).min(axis=1)
        X['dist_to_nearest_wall'] = dist_to_nearest_wall

        # 到最近墙的归一化距离（0=贴墙，1=中心）
        max_dist_to_wall = min(arena_width, arena_height) / 2
        X['dist_to_wall_norm'] = dist_to_nearest_wall / (max_dist_to_wall + 1e-6)

        # 到场地中心的距离
        dist_to_center = pd.Series(
            np.sqrt((center_x - arena_center_x) ** 2 + (center_y - arena_center_y) ** 2),
            index=single_mouse.index
        )
        X['dist_to_arena_center'] = dist_to_center

        # 归一化到中心的距离（0=中心，1=角落）
        max_dist_to_center = np.sqrt((arena_width / 2) ** 2 + (arena_height / 2) ** 2)
        X['dist_to_center_norm'] = dist_to_center / (max_dist_to_center + 1e-6)

        # 到四个角落的距离
        dist_to_corner_bl = pd.Series(
            np.sqrt((center_x - x_min) ** 2 + (center_y - y_min) ** 2),
            index=single_mouse.index
        )
        dist_to_corner_br = pd.Series(
            np.sqrt((center_x - x_max) ** 2 + (center_y - y_min) ** 2),
            index=single_mouse.index
        )
        dist_to_corner_tl = pd.Series(
            np.sqrt((center_x - x_min) ** 2 + (center_y - y_max) ** 2),
            index=single_mouse.index
        )
        dist_to_corner_tr = pd.Series(
            np.sqrt((center_x - x_max) ** 2 + (center_y - y_max) ** 2),
            index=single_mouse.index
        )

        # 到最近角落的距离
        dist_to_nearest_corner = pd.concat(
            [dist_to_corner_bl, dist_to_corner_br, dist_to_corner_tl, dist_to_corner_tr],
            axis=1
        ).min(axis=1)
        X['dist_to_nearest_corner'] = dist_to_nearest_corner

        # 空间区域二值指示特征
        is_near_wall = (dist_to_nearest_wall < border_width).astype(float)
        X['is_near_wall'] = is_near_wall

        # 是否在中央区域
        central_radius = min(arena_width, arena_height) * 0.3  # 中央 30% 区域
        is_in_center = (dist_to_center < central_radius).astype(float)
        X['is_in_center'] = is_in_center

        # 是否在角落区域
        corner_radius = min(arena_width, arena_height) * 0.2  # 角落 20% 区域
        is_in_corner = (dist_to_nearest_corner < corner_radius).astype(float)
        X['is_in_corner'] = is_in_corner

    # 6. 头部朝向与墙面的关系
    nose = _get_substitute_nose(single_mouse, available_body_parts)

    if nose is not None:
        nose_x = nose['x']
        nose_y = nose['y']

        # 头部朝向向量（体心指向头部）
        head_dir_x = nose_x - center_x
        head_dir_y = nose_y - center_y
        head_dir_norm = np.sqrt(head_dir_x ** 2 + head_dir_y ** 2) + 1e-6

        # 归一化朝向
        head_dir_x_unit = head_dir_x / head_dir_norm
        head_dir_y_unit = head_dir_y / head_dir_norm

        if is_circular:
            # 圆形场地：法线方向指向圆心
            # 从老鼠位置指向 arena 中心的单位向量
            to_center_x = arena_center_x - center_x
            to_center_y = arena_center_y - center_y
            to_center_norm = np.sqrt(to_center_x ** 2 + to_center_y ** 2) + 1e-6
            to_center_x_unit = to_center_x / to_center_norm
            to_center_y_unit = to_center_y / to_center_norm

            # 面向墙 = 朝向与指向圆心方向相反（负的点积）
            facing_nearest_wall = -(head_dir_x_unit * to_center_x_unit + head_dir_y_unit * to_center_y_unit)
            X['facing_nearest_wall'] = facing_nearest_wall
        else:
            # 矩形场地：到各墙的方向向量
            # 左墙：(-1, 0)，右墙：(1, 0)，下墙：(0, -1)，上墙：(0, 1)
            # 计算头部朝向与最近墙方向的点积
            wall_dirs = pd.DataFrame({
                'left': -head_dir_x_unit,  # 朝左
                'right': head_dir_x_unit,  # 朝右
                'bottom': -head_dir_y_unit,  # 朝下
                'top': head_dir_y_unit  # 朝上
            }, index=single_mouse.index)

            wall_dists = pd.DataFrame({
                'left': X['dist_to_left_wall'],
                'right': X['dist_to_right_wall'],
                'bottom': X['dist_to_bottom_wall'],
                'top': X['dist_to_top_wall']
            }, index=single_mouse.index)

            # 找到最近墙的索引
            # nearest_wall_idx = wall_dists.idxmin(axis=1) # 原代码会触发警告

            # 修复: 仅对非全NA的行计算idxmin，避免FutureWarning
            nearest_wall_idx = pd.Series(index=wall_dists.index, dtype='object')
            valid_rows = wall_dists.notna().any(axis=1)
            if valid_rows.any():
                nearest_wall_idx.loc[valid_rows] = wall_dists.loc[valid_rows].idxmin(axis=1)

            # 计算是否面向最近的墙
            facing_nearest_wall = pd.Series(index=single_mouse.index, dtype=float)
            for wall_name in ['left', 'right', 'bottom', 'top']:
                mask = (nearest_wall_idx == wall_name)
                facing_nearest_wall.loc[mask] = wall_dirs.loc[mask, wall_name]

            X['facing_nearest_wall'] = facing_nearest_wall

        window = [15, 30] if section == 9 else [15, 30, 60, 120]
        # 多尺度统计
        for w in window:
            ws = _scale(w, fps)
            roll_params = dict(min_periods=max(1, ws // 4), center=True)
            X[f'facing_wall_m{w}'] = X['facing_nearest_wall'].rolling(ws, **roll_params).mean()

    return X

# 获取替代的body_center和nose
def _get_substitute_body_center(mouse_data, avail_parts):
    # 1. 有 body_center 就用 body_center
    if 'body_center' in avail_parts:
        return mouse_data['body_center']

    # 对于没有body_center的(section7、8、9)
    # 2. 使用neck
    if 'neck' in avail_parts:
        return mouse_data['neck']
    # 3. 使用 nose + tail_base 中点
    if 'nose' in avail_parts and 'tail_base' in avail_parts:
        return (mouse_data['nose'] + mouse_data['tail_base']) / 2

    # 4. 使用 head + tail_base 中点
    if 'head' in avail_parts and 'tail_base' in avail_parts:
        return (mouse_data['head'] + mouse_data['tail_base']) / 2

    # 5. 使用耳朵中点(实际用不上)
    if 'ear_left' in avail_parts and 'ear_right' in avail_parts:
        return (mouse_data['ear_left'] + mouse_data['ear_right']) / 2

    return None

def _get_substitute_nose(mouse_data, avail_parts):
    # 1. 有 nose 就用 nose
    if 'nose' in avail_parts:
        return mouse_data['nose']

    # 对于没有nose的(section7)
    # 2. 使用 head 替代
    if 'head' in avail_parts:
        return mouse_data['head']

    # 3. 使用耳朵中点(实际用不上)
    if 'ear_left' in avail_parts and 'ear_right' in avail_parts:
        return (mouse_data['ear_left'] + mouse_data['ear_right']) / 2

    return None

# 双鼠交互特征
def add_ear_features(X, mouse_pair, avail_A, avail_B, fps):
    new_features = {}
    lag = _scale(10, fps)
    ear_types = ['left', 'right']
    for ear_type in ear_types:
        ear_col = f'ear_{ear_type}'
        if ear_col in avail_A and ear_col in avail_B:
            shA = mouse_pair['A'][ear_col].shift(lag)
            shB = mouse_pair['B'][ear_col].shift(lag)
            
            new_features[f'sp_A_{ear_type}'] = np.square(mouse_pair['A'][ear_col] - shA).sum(axis=1, skipna=False)
            new_features[f'sp_AB_{ear_type}'] = np.square(mouse_pair['A'][ear_col] - shB).sum(axis=1, skipna=False)
            new_features[f'sp_B_{ear_type}'] = np.square(mouse_pair['B'][ear_col] - shB).sum(axis=1, skipna=False)

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_nose_features(X, mouse_pair, avail_A, avail_B, fps):
    new_features = {}
    if 'nose' in avail_A and 'nose' in avail_B:
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            shA_n = mouse_pair['A']['nose'].shift(l)
            shB_n = mouse_pair['B']['nose'].shift(l)
            past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
            new_features[f'appr_{lag}'] = cur - past

        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x']) ** 2 +
                     (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y']) ** 2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            new_features[f'nn_lg{lag}'] = nn.shift(l)
            new_features[f'nn_ch{lag}'] = nn - nn.shift(l)
            is_cl = (nn < 10.0).astype(float)
            new_features[f'cl_ps{lag}'] = is_cl.rolling(l, min_periods=1).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_body_with_substitute_center_features(X, mouse_pair, avail_A, avail_B, fps):
    # 获取运动中心代理点
    center_A = _get_substitute_body_center(mouse_pair['A'], avail_A)
    center_B = _get_substitute_body_center(mouse_pair['B'], avail_B)

    # 如果任一老鼠没有可用的运动中心，跳过
    if center_A is None or center_B is None:
        return X

    new_features = {}
    
    # 相对位置
    rel_x = center_A['x'] - center_B['x']
    rel_y = center_A['y'] - center_B['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    A_vx = center_A['x'].diff()
    A_vy = center_A['y'].diff()
    B_vx = center_B['x'].diff()
    B_vy = center_B['y'].diff()
    # A、B的速度
    A_speed = np.sqrt(A_vx ** 2 + A_vy ** 2)
    B_speed = np.sqrt(B_vx ** 2 + B_vy ** 2)

    # 1. 相对速度分解：沿连线方向 vs 垂直方向
    # A沿A->B方向的速度分量
    A_vel_along = (A_vx * rel_x + A_vy * rel_y) / (rel_dist + 1e-6)
    # A垂直于A->B方向的速度分量
    A_vel_perp = (A_vx * (-rel_y) + A_vy * rel_x) / (rel_dist + 1e-6)

    # B沿B->A方向的速度分量（注意方向相反）
    B_vel_along = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (rel_dist + 1e-6)
    B_vel_perp = (B_vx * rel_y + B_vy * (-rel_x)) / (rel_dist + 1e-6)

    new_features['A_vel_along'] = A_vel_along
    new_features['A_vel_perp'] = A_vel_perp
    new_features['B_vel_along'] = B_vel_along
    new_features['B_vel_perp'] = B_vel_perp

    # 2. 追逐指标
    # A追B：A沿向B方向移动 + 距离在缩小
    dist_change = rel_dist.diff()
    A_chasing = ((A_vel_along > 0) & (dist_change < 0)).astype(float)
    B_chasing = ((B_vel_along > 0) & (dist_change > 0)).astype(float)

    new_features['A_chasing'] = A_chasing
    new_features['B_chasing'] = B_chasing

    # 追逐强度（速度 * 接近率）
    new_features['A_chase_intensity'] = A_vel_along * (-dist_change) / (A_speed + 1e-6)
    new_features['B_chase_intensity'] = B_vel_along * dist_change / (B_speed + 1e-6)

    # 3. 逃跑指标
    # B逃离A：B沿远离A方向移动 + A在接近
    A_escaping = ((A_vel_along < 0) & (B_vel_along < 0)).astype(float)
    B_escaping = ((B_vel_along < 0) & (A_vel_along > 0)).astype(float)

    new_features['A_escaping'] = A_escaping
    new_features['B_escaping'] = B_escaping

    # 4. 侧向躲避（垂直分量占主导）
    A_sidestepping = (np.abs(A_vel_perp) > np.abs(A_vel_along)).astype(float)
    B_sidestepping = (np.abs(B_vel_perp) > np.abs(B_vel_along)).astype(float)

    new_features['A_sidestepping'] = A_sidestepping
    new_features['B_sidestepping'] = B_sidestepping

    # 5. 多尺度统计
    for window in [15, 30, 60]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)

        # 追逐时间占比
        new_features[f'A_chasing_p{window}'] = A_chasing.rolling(ws, **roll_params).mean()
        new_features[f'B_chasing_p{window}'] = B_chasing.rolling(ws, **roll_params).mean()

        # 逃跑时间占比
        new_features[f'A_escaping_p{window}'] = A_escaping.rolling(ws, **roll_params).mean()
        new_features[f'B_escaping_p{window}'] = B_escaping.rolling(ws, **roll_params).mean()

        # 侧向移动占比
        new_features[f'A_sidestep_p{window}'] = A_sidestepping.rolling(ws, **roll_params).mean()
        new_features[f'B_sidestep_p{window}'] = B_sidestepping.rolling(ws, **roll_params).mean()

        # 追逐强度统计
        # 注意：这里需要先从new_features获取intensity
        new_features[f'A_chase_int_m{window}'] = new_features['A_chase_intensity'].rolling(ws, **roll_params).mean()
        new_features[f'B_chase_int_m{window}'] = new_features['B_chase_intensity'].rolling(ws, **roll_params).mean()

    # 6. 追逐方向一致性（持续追逐 vs 来回拉锯）
    for window in [30, 60]:
        ws = _scale(window, fps)
        # 方向一致性：同号的比例
        new_features[f'A_chase_consist_{window}'] = (A_vel_along > 0).astype(float).rolling(
            ws, min_periods=max(1, ws // 5), center=True
        ).mean()

        new_features[f'B_chase_consist_{window}'] = (B_vel_along > 0).astype(float).rolling(
            ws, min_periods=max(1, ws // 5), center=True
        ).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X


def add_body_without_substitute_center_features(X, mouse_pair, avail_A, avail_B, fps):
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X
    
    new_features = {} # 收集新特征
    
    # 通用变量
    center_A = mouse_pair['A']['body_center']
    center_B = mouse_pair['B']['body_center']

    # 相对位置
    rel_x = center_A['x'] - center_B['x']
    rel_y = center_A['y'] - center_B['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    # 相对速度
    rel_vx = rel_x.diff() * fps
    rel_vy = rel_y.diff() * fps
    rel_speed = np.sqrt(rel_vx ** 2 + rel_vy ** 2)

    # 相对加速度
    rel_ax = rel_vx.diff() * fps
    rel_ay = rel_vy.diff() * fps
    rel_accel = np.sqrt(rel_ax ** 2 + rel_ay ** 2)
    new_features['rel_accel'] = rel_accel

    # 相对速度方向角度
    rel_angle = pd.Series(np.arctan2(rel_vy, rel_vx), index=rel_vx.index)
    # 相对速度方向变化率（转向速度）
    angle_diff = rel_angle.diff()
    # 处理角度跳变（-π到π）
    angle_diff = pd.Series(np.where(angle_diff > np.pi, angle_diff - 2 * np.pi, angle_diff), index=rel_angle.index)
    angle_diff = pd.Series(np.where(angle_diff < -np.pi, angle_diff + 2 * np.pi, angle_diff), index=rel_angle.index)
    rel_turn_rate = np.abs(angle_diff) * fps
    new_features['rel_turn_rate'] = rel_turn_rate

    A_vx = center_A['x'].diff()
    A_vy = center_A['y'].diff()
    B_vx = center_B['x'].diff()
    B_vy = center_B['y'].diff()
    # A、B的速度
    A_speed = np.sqrt(A_vx ** 2 + A_vy ** 2)
    B_speed = np.sqrt(B_vx ** 2 + B_vy ** 2)

    # 运动方向角度
    A_angle = pd.Series(np.arctan2(A_vy, A_vx), index=A_vx.index)
    B_angle = pd.Series(np.arctan2(B_vy, B_vx), index=B_vx.index)

    # 方向差（处理角度跳变）
    angle_diff_2 = A_angle - B_angle
    angle_diff_2 = pd.Series(np.where(angle_diff_2 > np.pi, angle_diff_2 - 2 * np.pi, angle_diff_2), index=angle_diff_2.index)
    angle_diff_2 = pd.Series(np.where(angle_diff_2 < -np.pi, angle_diff_2 + 2 * np.pi, angle_diff_2), index=angle_diff_2.index)

    # 开始提取不同特征
    new_features['v_cls'] = (rel_dist < 5.0).astype(float)
    new_features['cls'] = ((rel_dist >= 5.0) & (rel_dist < 15.0)).astype(float)
    new_features['med'] = ((rel_dist >= 15.0) & (rel_dist < 30.0)).astype(float)
    new_features['far'] = (rel_dist >= 30.0).astype(float)

    cd_full = np.square(center_A - center_B).sum(axis=1, skipna=False)
    coord = A_vx * B_vx + A_vy * B_vy
    for w in [5, 15, 30, 60]:
        ws = _scale(w, fps)
        roll = dict(min_periods=1, center=True)
        new_features[f'd_m{w}'] = cd_full.rolling(ws, **roll).mean()
        new_features[f'd_s{w}'] = cd_full.rolling(ws, **roll).std()
        new_features[f'd_mn{w}'] = cd_full.rolling(ws, **roll).min()
        new_features[f'd_mx{w}'] = cd_full.rolling(ws, **roll).max()

        d_var = cd_full.rolling(ws, **roll).var()
        new_features[f'int{w}'] = 1 / (1 + d_var)

        new_features[f'co_m{w}'] = coord.rolling(ws, **roll).mean()
        new_features[f'co_s{w}'] = coord.rolling(ws, **roll).std()
    w = _scale(30, fps)
    new_features['int_con'] = cd_full.rolling(w, min_periods=1, center=True).std() / \
                   (cd_full.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    val = (A_vx * B_vx + A_vy * B_vy) / (np.sqrt(A_vx ** 2 + A_vy ** 2) * np.sqrt(B_vx ** 2 + B_vy ** 2) + 1e-6)
    for off in [-30, -20, -10, 0, 10, 20, 30]:
        o = _scale_signed(off, fps)
        new_features[f'va_{off}'] = val.shift(-o)

    '''
    原add_interaction_features函数内的特征
    - A_ld*, B_ld*: 领先/跟随指标（谁在追谁）
    - chase_*: 追逐行为强度
    - sp_cor*: 速度相关性（同步运动程度）
    '''
    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx ** 2 + A_vy ** 2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx ** 2 + B_vy ** 2) * rel_dist + 1e-6)

    for window in [30, 60]:
        ws = _scale(window, fps)
        new_features[f'A_ld{window}'] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        new_features[f'B_ld{window}'] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    approach = -rel_dist.diff()
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    new_features[f'chase_{w}'] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = np.sqrt(A_vx ** 2 + A_vy ** 2)
        B_sp = np.sqrt(B_vx ** 2 + B_vy ** 2)
        new_features[f'sp_cor{window}'] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    '''
    原add_advanced_interaction_dynamics函数内的特征
    '''
    def simple_dtw_distance(seq1, seq2):
        """简化的DTW距离计算"""
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0:
            return 0
        # 使用欧氏距离作为基础
        return np.mean(np.abs(seq1[:min(n, m)] - seq2[:min(n, m)]))

    # 计算速度序列的DTW-like距离
    # 注意：这里需要逐行计算，难以向量化，可能仍会比较慢
    # 但我们可以尽量减少 DataFrame 操作
    window_size = _scale(30, fps)
    if len(A_vx) > window_size:
        A_speed = np.sqrt(A_vx ** 2 + A_vy ** 2)
        B_speed = np.sqrt(B_vx ** 2 + B_vy ** 2)
        
        # 使用 numpy 数组加速
        A_speed_vals = A_speed.values
        B_speed_vals = B_speed.values
        dtw_distances = np.full(len(A_speed), np.nan)

        for i in range(window_size, len(A_speed)):
            window_A = A_speed_vals[i - window_size:i]
            window_B = B_speed_vals[i - window_size:i]
            dtw_distances[i] = simple_dtw_distance(window_A, window_B)
            
        new_features['speed_dtw_distance'] = pd.Series(dtw_distances, index=X.index)

    # 2. 领导-跟随关系的动态变化
    # 领导力指标（基于速度方向和相对位置的相关性）
    A_leadership = (A_vx * rel_x + A_vy * rel_y) / (rel_dist + 1e-6)
    B_leadership = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (rel_dist + 1e-6)
    leadership_asymmetry = A_leadership - B_leadership
    new_features['leadership_asymmetry'] = leadership_asymmetry
    # 3. 交互势能（基于距离和速度）
    # 类似物理中的势能概念：距离越近，交互势能越高
    interaction_potential = 1 / (rel_dist + 1e-6)
    new_features['interaction_potential'] = interaction_potential
    # 4. 逃避/接近行为的量化
    approach_rate = -rel_dist.diff() * fps  # 正表示接近，负表示远离
    new_features['approach_rate'] = approach_rate
    # 逃避行为的检测（突然的远离）
    sudden_escape = (approach_rate < -20).astype(float)  # 阈值可调整
    new_features['sudden_escape'] = sudden_escape

    '''
    原add_relative_trajectory_features函数内的特征
    '''
    # 相对路径曲率（相对速度方向的变化率 / 相对速度）
    rel_curvature = rel_turn_rate / (rel_speed + 1e-6)
    new_features['rel_curvature'] = rel_curvature

    # 多尺度统计
    for window in [15, 30, 60]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)
        new_features[f'rel_turn_rate_m{window}'] = rel_turn_rate.rolling(ws, **roll_params).mean()
        new_features[f'rel_curvature_m{window}'] = rel_curvature.rolling(ws, **roll_params).mean()
        new_features[f'rel_accel_m{window}'] = rel_accel.rolling(ws, **roll_params).mean()
        new_features[f'rel_speed_m{window}'] = rel_speed.rolling(ws, **roll_params).mean()
        new_features[f'rel_speed_s{window}'] = rel_speed.rolling(ws, **roll_params).std()

    '''
    原add_motion_pattern_similarity_features函数内的特征
    '''
    # 方向相似性（余弦相似度）
    direction_similarity = np.cos(angle_diff_2)
    new_features['direction_similarity'] = direction_similarity

    # 速度相似性（归一化的速度差）
    speed_similarity = 1 - np.abs(A_speed - B_speed) / (A_speed + B_speed + 1e-6)
    new_features['speed_similarity'] = speed_similarity

    # 多尺度统计：滚动窗口内的相关性
    for window in [60, 120]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 6), center=True)

        # 方向相似性均值
        new_features[f'direction_sim_m{window}'] = direction_similarity.rolling(ws, **roll_params).mean()

        # 速度相似性均值
        new_features[f'speed_sim_m{window}'] = speed_similarity.rolling(ws, **roll_params).mean()

    '''
    原add_symmetric_asymmetric_features函数内的特征
    '''
    # 1. 速度差和速度比
    speed_diff = A_speed - B_speed
    speed_ratio = A_speed / (B_speed + 1e-6)

    new_features['speed_diff'] = speed_diff
    new_features['speed_ratio'] = speed_ratio

    for window in [15, 30, 60]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)
        new_features[f'speed_diff_m{window}'] = speed_diff.rolling(ws, **roll_params).mean()
        new_features[f'speed_diff_s{window}'] = speed_diff.rolling(ws, **roll_params).std()
        new_features[f'speed_ratio_m{window}'] = speed_ratio.rolling(ws, **roll_params).mean()

    # 2. 朝向差（如果已有朝向特征）
    if 'A_face_B' in X.columns and 'B_face_A' in X.columns:
        # 朝向不对称性：A面向B但B不面向A，或反之
        facing_asymmetry = X['A_face_B'] - X['B_face_A']
        new_features['facing_asymmetry'] = facing_asymmetry

        for window in [15, 30]:
            ws = _scale(window, fps)
            new_features[f'facing_asym_m{window}'] = facing_asymmetry.rolling(
                ws, min_periods=max(1, ws // 5), center=True
            ).mean()

    # 3. 活动半径差（路径长度）
    for window in [30, 60]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)

        # A的路径长度
        A_path_length = A_speed.rolling(ws, **roll_params).sum()
        # B的路径长度
        B_path_length = B_speed.rolling(ws, **roll_params).sum()

        # 路径长度差
        path_diff = A_path_length - B_path_length
        new_features[f'path_diff_{window}'] = path_diff

        # 路径长度比
        new_features[f'path_ratio_{window}'] = A_path_length / (B_path_length + 1e-6)

    # 4. 加速度差
    A_ax = A_vx.diff() * fps
    A_ay = A_vy.diff() * fps
    B_ax = B_vx.diff() * fps
    B_ay = B_vy.diff() * fps

    A_accel = np.sqrt(A_ax**2 + A_ay**2)
    B_accel = np.sqrt(B_ax**2 + B_ay**2)

    accel_diff = A_accel - B_accel
    new_features['accel_diff'] = accel_diff

    for window in [15, 30]:
        ws = _scale(window, fps)
        new_features[f'accel_diff_m{window}'] = accel_diff.rolling(
            ws, min_periods=max(1, ws // 5), center=True
        ).mean()
        
    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
        
    return X

def add_tail_features(X, mouse_pair, avail_A, avail_B, fps):
    if 'tail_base' not in avail_A or 'tail_base' not in avail_B:
        return X
    
    new_features = {}
    tail_A = mouse_pair['A']['tail_base']
    tail_B = mouse_pair['B']['tail_base']

    # 尾部相对位置
    tail_rel_x = tail_B['x'] - tail_A['x']
    tail_rel_y = tail_B['y'] - tail_A['y']
    tail_rel_dist = np.sqrt(tail_rel_x ** 2 + tail_rel_y ** 2)
    new_features['tail_rel_dist'] = tail_rel_dist

    # 尾部相对速度
    tail_rel_vx = tail_rel_x.diff() * fps
    tail_rel_vy = tail_rel_y.diff() * fps
    tail_rel_speed = np.sqrt(tail_rel_vx ** 2 + tail_rel_vy ** 2)
    new_features['tail_rel_speed'] = tail_rel_speed

    # 尾部接近/远离速度
    tail_approach_rate = -tail_rel_dist.diff() * fps
    new_features['tail_approach_rate'] = tail_approach_rate

    # 如果同时有nose，计算尾部-头部相对位置
    if 'nose' in avail_A and 'nose' in avail_B:
        nose_A = mouse_pair['A']['nose']
        nose_B = mouse_pair['B']['nose']

        # A的尾部到B的头部距离
        new_features['tailA_noseB_dist'] = np.sqrt((tail_A['x'] - nose_B['x']) ** 2 + (tail_A['y'] - nose_B['y']) ** 2)

        # B的尾部到A的头部距离
        new_features['tailB_noseA_dist'] = np.sqrt((tail_B['x'] - nose_A['x']) ** 2 + (tail_B['y'] - nose_A['y']) ** 2)

    # 多尺度统计
    for window in [15, 30, 60]:
        ws = _scale(window, fps)
        roll_params = dict(min_periods=max(1, ws // 5), center=True)
        new_features[f'tail_rel_dist_m{window}'] = tail_rel_dist.rolling(ws, **roll_params).mean()
        new_features[f'tail_rel_speed_m{window}'] = tail_rel_speed.rolling(ws, **roll_params).mean()
        new_features[f'tail_approach_rate_m{window}'] = tail_approach_rate.rolling(ws, **roll_params).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

def add_nose_tail_body_features(X, mouse_pair, avail_A, avail_B, fps):
    # 检查是否有body_center
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    new_features = {}
    
    # 体心距离
    center_A = mouse_pair['A']['body_center']
    center_B = mouse_pair['B']['body_center']

    vec_AB = center_B - center_A  # A -> B
    vec_BA = center_A - center_B  # B -> A

    rel_x = center_A['x'] - center_B['x']
    rel_y = center_A['y'] - center_B['y']
    rel_dist = np.sqrt(rel_x ** 2 + rel_y ** 2)

    # 如果有nose和tail_base，计算朝向相关的重叠特征
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        # 身体朝向
        ori_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        ori_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        dot_A = ori_A['x'] * vec_AB['x'] + ori_A['y'] * vec_AB['y']
        dot_B = ori_B['x'] * vec_BA['x'] + ori_B['y'] * vec_BA['y']
        norm_A = (np.sqrt(ori_A['x'] ** 2 + ori_A['y'] ** 2) *
                  np.sqrt(vec_AB['x'] ** 2 + vec_AB['y'] ** 2) + 1e-6)
        norm_B = (np.sqrt(ori_B['x'] ** 2 + ori_B['y'] ** 2) *
                  np.sqrt(vec_BA['x'] ** 2 + vec_BA['y'] ** 2) + 1e-6)

        new_features['A_face_B'] = dot_A / norm_A
        new_features['B_face_A'] = dot_B / norm_B

        # 在 B 的身体坐标系中表达 A 的位置：前后(front) + 左右(side)
        ori_B_norm = np.sqrt(ori_B['x'] ** 2 + ori_B['y'] ** 2) + 1e-6
        front_BA = (vec_BA['x'] * ori_B['x'] + vec_BA['y'] * ori_B['y']) / ori_B_norm
        side_BA = (vec_BA['x'] * (-ori_B['y']) + vec_BA['y'] * ori_B['x']) / ori_B_norm

        new_features['A_front_of_B'] = front_BA
        new_features['A_side_of_B'] = side_BA

        # 在 A 的身体坐标系中表达 B 的位置：前后(front) + 左右(side)
        ori_A_norm = np.sqrt(ori_A['x'] ** 2 + ori_A['y'] ** 2) + 1e-6
        front_AB = (vec_AB['x'] * ori_A['x'] + vec_AB['y'] * ori_A['y']) / ori_A_norm
        side_AB = (vec_AB['x'] * (-ori_A['y']) + vec_AB['y'] * ori_A['x']) / ori_A_norm

        new_features['B_front_of_A'] = front_AB
        new_features['B_side_of_A'] = side_AB

        # 朝向对齐度（余弦相似度）
        ori_alignment = (ori_A['x'] * ori_B['x'] + ori_A['y'] * ori_B['y']) / (
            np.sqrt(ori_A['x']**2 + ori_A['y']**2) * np.sqrt(ori_B['x']**2 + ori_B['y']**2) + 1e-6
        )
        new_features['body_ori_alignment'] = ori_alignment

        # 重叠概率：距离很近 + 朝向对齐
        overlap_score = (1 / (rel_dist + 1.0)) * np.abs(ori_alignment)
        new_features['body_overlap_score'] = overlap_score

        # 重叠状态（二值）
        overlap_binary = ((rel_dist < 8.0) & (np.abs(ori_alignment) > 0.7)).astype(float)
        new_features['body_overlap_binary'] = overlap_binary

        # 滚动窗口统计
        for window in [15, 30, 60]:
            ws = _scale(window, fps)
            roll_params = dict(min_periods=max(1, ws // 5), center=True)
            new_features[f'overlap_score_m{window}'] = overlap_score.rolling(ws, **roll_params).mean()
            new_features[f'overlap_binary_p{window}'] = overlap_binary.rolling(ws, **roll_params).mean()

        # 垂直重叠特征（A在B上方或下方）
        # A在B正上方：front接近0，side接近0，距离很近
        vertical_overlap = ((np.abs(front_BA) < 3.0) &
                           (np.abs(side_BA) < 3.0) &
                           (rel_dist < 8.0)).astype(float)
        new_features['vertical_overlap'] = vertical_overlap

        for window in [15, 30]:
            ws = _scale(window, fps)
            new_features[f'vertical_overlap_p{window}'] = vertical_overlap.rolling(
                ws, min_periods=max(1, ws // 5), center=True
            ).mean()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X

# 双鼠交互高级特征：接触类型+角色分化
def _get_front_anchor(mouse_df, avail_parts, mouse_name=''):
    """
    获取老鼠的前端锚点（用于接触检测）

    优先级：nose > head > 耳朵中点

    参数
    ------
    mouse_df : pd.DataFrame
        单只老鼠的关键点数据
    avail_parts : Index
        可用的身体部位列表

    返回
    ------
    anchor : pd.DataFrame or None
        前端锚点的坐标 (x, y)
    anchor_name : str or None
        锚点名称
    """
    # 优先级1: nose
    if 'nose' in avail_parts:
        return mouse_df['nose'], 'nose'

    # 优先级2: head
    if 'head' in avail_parts:
        return mouse_df['head'], 'head'

    # 优先级3: 耳朵中点
    if 'ear_left' in avail_parts and 'ear_right' in avail_parts:
        ear_mid = pd.DataFrame({
            'x': (mouse_df['ear_left']['x'] + mouse_df['ear_right']['x']) / 2,
            'y': (mouse_df['ear_left']['y'] + mouse_df['ear_right']['y']) / 2,
        }, index=mouse_df.index)
        return ear_mid, 'ear_mid'

    # 无可用锚点
    return None, None

def add_contact_semantic_features(X, mouse_pair, avail_A, avail_B, fps):
    new_features = {}
    
    # 获取A和B的前端锚点
    anchor_A, anchorA_name = _get_front_anchor(mouse_pair['A'], avail_A, 'Mouse A')
    anchor_B, anchorB_name = _get_front_anchor(mouse_pair['B'], avail_B, 'Mouse B')

    # 1. A前端锚点到B各部位的最小距离
    if anchor_A is not None:
        distances_to_B = {}
        for part in avail_B:
            if part in mouse_pair['B'].columns.get_level_values(0):
                part_B = mouse_pair['B'][part]
                dist = np.sqrt((anchor_A['x'] - part_B['x'])**2 + (anchor_A['y'] - part_B['y'])**2)
                distances_to_B[part] = dist

        if distances_to_B:
            # A前端锚点到B任意部位的最小距离
            min_dist_A_front_to_B = pd.concat(distances_to_B.values(), axis=1).min(axis=1)
            new_features['A_front_to_B_min'] = min_dist_A_front_to_B

            # 特定部位的距离（如果存在）
            if 'nose' in distances_to_B:
                new_features['A_front_to_B_nose'] = distances_to_B['nose']
            if 'head' in distances_to_B:
                new_features['A_front_to_B_head'] = distances_to_B['head']
            if 'body_center' in distances_to_B:
                new_features['A_front_to_B_center'] = distances_to_B['body_center']
            if 'tail_base' in distances_to_B:
                new_features['A_front_to_B_tail'] = distances_to_B['tail_base']

            # 接触阈值特征（多个阈值）
            for threshold in [3.0, 5.0, 8.0]:  # cm
                contact_mask = (min_dist_A_front_to_B < threshold).astype(float)
                new_features[f'A_front_contact_{int(threshold)}'] = contact_mask

                # 滚动窗口内的接触占比
                for window in [15, 30, 60]:
                    ws = _scale(window, fps)
                    new_features[f'A_front_contact_{int(threshold)}_p{window}'] = contact_mask.rolling(
                        ws, min_periods=max(1, ws // 5), center=True
                    ).mean()

    # 2. B前端锚点到A各部位的最小距离（对称特征）
    if anchor_B is not None:
        distances_to_A = {}
        for part in avail_A:
            if part in mouse_pair['A'].columns.get_level_values(0):
                part_A = mouse_pair['A'][part]
                dist = np.sqrt((anchor_B['x'] - part_A['x'])**2 + (anchor_B['y'] - part_A['y'])**2)
                distances_to_A[part] = dist

        if distances_to_A:
            min_dist_B_front_to_A = pd.concat(distances_to_A.values(), axis=1).min(axis=1)
            new_features['B_front_to_A_min'] = min_dist_B_front_to_A

            if 'nose' in distances_to_A:
                new_features['B_front_to_A_nose'] = distances_to_A['nose']
            if 'head' in distances_to_A:
                new_features['B_front_to_A_head'] = distances_to_A['head']
            if 'body_center' in distances_to_A:
                new_features['B_front_to_A_center'] = distances_to_A['body_center']
            if 'tail_base' in distances_to_A:
                new_features['B_front_to_A_tail'] = distances_to_A['tail_base']

            for threshold in [3.0, 5.0, 8.0]:
                contact_mask = (min_dist_B_front_to_A < threshold).astype(float)
                new_features[f'B_front_contact_{int(threshold)}'] = contact_mask

                for window in [15, 30, 60]:
                    ws = _scale(window, fps)
                    new_features[f'B_front_contact_{int(threshold)}_p{window}'] = contact_mask.rolling(
                        ws, min_periods=max(1, ws // 5), center=True
                    ).mean()

    # 3. 接触持续时间特征
    # 注意：需要检查 min_dist 特征是否已在本轮计算中生成
    A_min_dist = new_features.get('A_front_to_B_min', X.get('A_front_to_B_min'))
    if A_min_dist is not None:
        contact_binary = (A_min_dist < 5.0).astype(int)
        for window in [30, 60]:
            ws = _scale(window, fps)
            new_features[f'A_contact_duration_{window}'] = contact_binary.rolling(
                ws, min_periods=1, center=True
            ).sum()

    B_min_dist = new_features.get('B_front_to_A_min', X.get('B_front_to_A_min'))
    if B_min_dist is not None:
        contact_binary = (B_min_dist < 5.0).astype(int)
        for window in [30, 60]:
            ws = _scale(window, fps)
            new_features[f'B_contact_duration_{window}'] = contact_binary.rolling(
                ws, min_periods=1, center=True
            ).sum()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    return X


def transform_single(single_mouse, body_parts_tracked, fps, section, video_id=None):
    available_body_parts = single_mouse.columns.get_level_values(0)
    
    # 初始化特征字典
    features = {}

    # 1. 身体部位间距离
    for p1, p2 in itertools.combinations(body_parts_tracked, 2):
        if p1 in available_body_parts and p2 in available_body_parts:
            features[f"{p1}+{p2}"] = np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)

    # 先创建初始 DataFrame
    X = pd.DataFrame(features, index=single_mouse.index)
    # Reindex using list comprehension to ensure order (optional, but keeps consistency)
    cols = [f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)]
    X = X.reindex(columns=cols, copy=False)
    

    # === Best: missingness-as-signal features ===
    X = add_missingness_features_single(X, single_mouse, fps, section)
    # === End best code ===


    # 重新开始收集后续特征
    new_features = {}

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        new_features['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    center = _get_substitute_body_center(single_mouse, available_body_parts)
    nose = _get_substitute_nose(single_mouse, available_body_parts)

    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        lag = _scale(10, fps)
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(lag)
        new_features['sp_lf'] = np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False)
        new_features['sp_rt'] = np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False)
        new_features['sp_lf2'] = np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False)
        new_features['sp_rt2'] = np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False)

    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 +
                        (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        for off in [-30, -20, -10, 10, 20, 30]:
            o = _scale_signed(off, fps)
            new_features[f'ear_o{off}'] = ear_d.shift(-o)
        w = _scale(30, fps)
        new_features['ear_con'] = ear_d.rolling(w, min_periods=1, center=True).std() / \
                       (ear_d.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    if 'tail_base' in available_body_parts and center is not None and nose is not None:
        v1 = nose - center
        v2 = single_mouse['tail_base'] - center
        new_features['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

    if 'tail_base' in available_body_parts and nose is not None:
        nt_dist = np.sqrt((nose['x'] - single_mouse['tail_base']['x'])**2 +
                          (nose['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            new_features[f'nt_lg{lag}'] = nt_dist.shift(l)
            new_features[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l)

    if center is not None:
        cx = center['x']
        cy = center['y']

        # === Best: winsorize once, then cheap rolling max-min (avoid rolling quantile cost) ===
        cx_clip = cx.clip(lower=cx.quantile(0.01), upper=cx.quantile(0.99))
        cy_clip = cy.clip(lower=cy.quantile(0.01), upper=cy.quantile(0.99))
        # === End best code ===

        window = [5, 15, 30, 60] if section == 9 else [15, 30, 60, 120]
        for w in window:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)

            new_features[f'cx_m{w}'] = cx.rolling(ws, **roll).mean()
            new_features[f'cy_m{w}'] = cy.rolling(ws, **roll).mean()
            new_features[f'cx_s{w}'] = cx.rolling(ws, **roll).std()
            new_features[f'cy_s{w}'] = cy.rolling(ws, **roll).std()

            # FIX: robust range (after winsorize)
            new_features[f'x_rng{w}'] = cx_clip.rolling(ws, **roll).max() - cx_clip.rolling(ws, **roll).min()
            new_features[f'y_rng{w}'] = cy_clip.rolling(ws, **roll).max() - cy_clip.rolling(ws, **roll).min()
            
            
            
            new_features[f'disp{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).sum()**2 +
                                     cy.diff().rolling(ws, min_periods=1).sum()**2)
            new_features[f'act{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).var() +
                                   cy.diff().rolling(ws, min_periods=1).var())

    # 一次性合并目前的特征
    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    # 调用子函数（每个子函数现在都会返回一个新的合并后的 DataFrame）
    if center is not None:
        X = add_curvature_features(X, cx, cy, fps, section)
        X = add_multiscale_features(X, cx, cy, fps, section)
        X = add_state_features(X, cx, cy, fps, section)
        X = add_longrange_features(X, cx, cy, fps)

    X = add_posture_stability_features(X, single_mouse, available_body_parts, fps)
    X = add_head_elevation_features(X, single_mouse, available_body_parts, fps, section)
    X = add_body_width_features(X, single_mouse, available_body_parts, fps, section)
    X = add_pose_shape_features(X, single_mouse, available_body_parts, fps, section)
    X = add_head_body_decoupled_features(X, single_mouse, available_body_parts, fps, section)
    X = add_body_axis_motion_features(X, single_mouse, available_body_parts, fps, section)
    X = add_high_freq_micromotion_features(X, single_mouse, available_body_parts, fps, section)
    if video_id is not None:
        X = add_arena_spatial_features(X, single_mouse, available_body_parts, fps, section, video_id)

    return X.astype(np.float32, copy=False)

def transform_pair(mouse_pair, body_parts_tracked, fps):
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)
    
    features = {}
    for p1, p2 in itertools.product(body_parts_tracked, repeat=2):
        if p1 in avail_A and p2 in avail_B:
            features[f"{p1}+{p2}"] = np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)

    X = pd.DataFrame(features, index=mouse_pair.index)
    cols = [f"{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)]
    X = X.reindex(columns=cols, copy=False)


    # === Best: missingness-as-signal features for A/B ===
    X = add_missingness_features_pair(X, mouse_pair, fps)
    # === End best code ===


    new_features = {}
    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        new_features['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)
        
    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    # 添加双鼠交互特征
    X = add_nose_tail_body_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_nose_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_tail_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_ear_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_body_with_substitute_center_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_body_without_substitute_center_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_contact_semantic_features(X, mouse_pair, avail_A, avail_B, fps)

    return X.astype(np.float32, copy=False)

# 评估函数
class HostVisibleError(Exception):
    pass

# 训练和提交
def robustify(submission, dataset, traintest, traintest_directory=None):
    """
    对提交结果进行鲁棒性处理和验证

    输入:
        submission: pd.DataFrame - 预测结果，包含列：
            video_id, agent_id, target_id, action, start_frame, stop_frame
        dataset: pd.DataFrame - 数据集元信息
        traintest: str - 'train'或'test'
        traintest_directory: str, optional - tracking数据目录

    输出:
        pd.DataFrame - 处理后的提交结果

    作用:
        1. 删除无效的预测（start_frame >= stop_frame）
        2. 删除重叠的预测区间
        3. 为没有预测的视频填充默认预测
        确保提交格式符合比赛要求
    """
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
        # traintest_directory = f"dataset/MABe-mouse-behavior-detection/{traintest}_tracking"

    old_submission = submission.copy()
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != len(old_submission):
        print("ERROR: Dropped frames with start >= stop")

    old_submission = submission.copy()
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop_frame:
                mask[i] = False
            else:
                last_stop_frame = row['stop_frame']
        group_list.append(group[mask])

    submission = pd.concat(group_list)

    if len(submission) != len(old_submission):
        print("ERROR: Dropped duplicate frames")

    s_list = []
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):
            continue

        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        if type(row.behaviors_labeled) != str:
            continue

        print(f"Video {video_id} has no predictions.")

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        vid_behaviors = json.loads(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])

        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1

        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])
        print("ERROR: Filled empty videos")

    submission = submission.reset_index(drop=True)

    return submission

def predict_multiclass(pred, meta, thresholds):
    """
    将多个二分类预测结果转换为多分类预测区间

    输入:
        pred: pd.DataFrame - 预测概率矩阵，每列对应一个行为类别
        meta: pd.DataFrame - 元数据，包含video_id, agent_id, target_id, video_frame
        thresholds: dict - 每个行为的阈值字典

    输出:
        pd.DataFrame - 预测结果，包含列：
            video_id, agent_id, target_id, action, start_frame, stop_frame

    作用:
        1. 对每一帧选择概率最高的行为
        2. 应用阈值过滤低置信度预测
        3. 将连续的相同预测合并为时间区间
        4. 处理视频边界情况
    """
    ama = np.argmax(pred.values, axis=1)
    max_proba = pred.max(axis=1).values

    threshold_array = np.array([thresholds.get(col, 0.27) for col in pred.columns])
    action_thresholds = threshold_array[ama]

    ama = np.where(max_proba >= action_thresholds, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)

    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]

    mask = ama_changes.values >= 0
    mask[-1] = False

    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })

    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame

    return submission_part

def tune_threshold(oof_action, y_action):
    """
    使用Optuna优化二分类阈值以最大化F1分数

    输入:
        oof_action: np.array - 交叉验证的预测概率
        y_action: np.array - 真实标签（0或1）

    输出:
        float - 最优阈值（0到1之间）

    作用:
        通过网格搜索找到使F1分数最大的概率阈值，
        用于将概率预测转换为二分类结果
    """
    def objective(trial):
        threshold = trial.suggest_float("threshold", 0, 1, step=0.01)
        return f1_score(y_action, (oof_action >= threshold), zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=-1)
    return study.best_params["threshold"]

def cross_validate_classifier(X, label, meta, body_parts_tracked_str, section):
    """
    对分类器进行交叉验证训练

    输入:
        X: pd.DataFrame - 特征矩阵
        label: pd.DataFrame - 标签矩阵，每列对应一个行为类别
        meta: pd.DataFrame - 元数据，包含video_id等分组信息
        body_parts_tracked_str: str - 身体部位配置的字符串表示
        section: int - 配置编号，用于保存模型

    输出:
        tuple: (submission_list, f1_list, thresholds)
            - submission_list: list - 预测结果列表
            - f1_list: list - 每个行为的F1分数列表
            - thresholds: dict - 每个行为的最优阈值

    作用:
        1. 对每个行为类别分别训练二分类模型
        2. 使用StratifiedGroupKFold进行交叉验证
        3. 优化每个行为的预测阈值
        4. 保存训练好的模型和阈值
        5. 生成out-of-fold预测结果
    """
    oof = pd.DataFrame(index=meta.video_frame)

    f1_list = []
    submission_list = []
    thresholds = {}

    for action in label.columns:
        action_mask = ~ label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)
        X_action = X[action_mask]
        groups_action = meta.video_id[action_mask]

        if len(np.unique(groups_action)) < CFG.n_splits:
            continue

        if not (y_action == 0).all():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)

                    trainer = Trainer(
                        estimator=clone(CFG.model),
                        cv=CFG.cv,
                        cv_args={"groups": groups_action},
                        metric=f1_score,
                        task="binary",
                        verbose=False,
                        save=True,
                        save_path=f"{CFG.model_name}/{section}/{action}"
                    )

                    trainer.fit(X_action, y_action)
                    oof_action = trainer.oof_preds

                    threshold = tune_threshold(oof_action, y_action)
                    thresholds[action] = threshold

                    f1 = f1_score(y_action, (oof_action >= threshold), zero_division=0)
                    f1_list.append((body_parts_tracked_str, action, f1))

                    joblib.dump(oof_action, f"{CFG.model_name}/{section}/{action}/oof_pred_probs.pkl")
                    joblib.dump(threshold, f"{CFG.model_name}/{section}/{action}/threshold.pkl")

                    log_print(f"\tF1: {f1:.4f} (threshold: {threshold:.2f}) Section: {section} Action: {action}")

                    

            except Exception as e:
                oof_action = np.zeros(len(y_action))
                log_print(f"\tF1: 0.0000 (0.00) Section: {section} Action: {action}")

        else:
            oof_action = np.zeros(len(y_action))
            log_print(f"\tF1: 0.0000 (0.00) Section: {section} Action: {action}")

        oof_column = np.zeros(len(label))
        oof_column[action_mask] = oof_action
        oof[action] = oof_column

        

    submission_part = predict_multiclass(oof, meta, thresholds)
    submission_list.append(submission_part)

    return submission_list, f1_list, thresholds

def submit(body_parts_tracked_str, switch_tr, section, thresholds):
    """
    对测试集进行预测并生成提交结果

    输入:
        body_parts_tracked_str: str - 身体部位配置的JSON字符串
        switch_tr: str - 'single'或'pair'，指定预测类型
        section: int - 配置编号，用于加载对应的模型
        thresholds: dict - 预测阈值字典

    输出:
        list - 预测结果列表，每个元素是一个DataFrame

    作用:
        1. 加载训练好的模型
        2. 对测试集数据进行特征提取
        3. 使用模型进行预测
        4. 应用阈值并转换为提交格式
    """
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(
        test_subset,
        'test',
        traintest_directory=CFG.test_tracking_path,
        generate_single=(switch_tr == 'single'),
        generate_pair=(switch_tr == 'pair')
    )

    fps_lookup = (
        test_subset[['video_id', 'frames_per_second']]
        .drop_duplicates('video_id')
        .set_index('video_id')['frames_per_second']
        .to_dict()
    )

    submission_list = []
    features_printed = True  # Add flag to print features only once
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)
            video_id_i = meta_te['video_id'].iloc[0]

            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked, fps_i, section, video_id=video_id_i)
            else:
                X_te = transform_pair(data_te, body_parts_tracked, fps_i)
            

            # === New: attach meta-based features to X_te ===
            X_meta_te = meta_to_features(meta_te)
            X_meta_te.index = X_te.index
            X_te = pd.concat([X_te, X_meta_te], axis=1)
            # === End new code ===

            if not features_printed:
                print(f"\n[DEBUG] Submit Mode ({switch_te}) - Feature Count: {len(X_te.columns)}")
                print(f"[DEBUG] Submit Mode ({switch_te}) - Features: {list(X_te.columns)}")
                features_printed = True

            pred = pd.DataFrame(index=meta_te.video_frame)
            for action in actions_te:
                files = glob.glob(f"{CFG.model_path}/{CFG.model_name}/{section}/{action}/*_trainer_*.pkl")
                if len(files) == 1:
                    trainer = joblib.load(files[0])
                    pred[action] = trainer.predict(X_te)

                    

            

            if pred.shape[1] != 0:
                submission_part = predict_multiclass(pred, meta_te, thresholds)
                submission_list.append(submission_part)

        except KeyError:
            del data_te
            

    return submission_list


# ============================================================================
# === Logging setup (只在 validate 模式下执行) ===
# ============================================================================
if CFG.mode == "validate":
    # Setup logging - create log directory with timestamp
    log_dir = os.path.join(".", "log", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Copy current script to log directory
    __file__ = "./run.py"
    shutil.copy(__file__, os.path.join(log_dir, "run.py"))
    
    # Open log file for writing
    log_file = open(os.path.join(log_dir, "training.log"), "w")
    
    def log_print(message):
        """Print and log specific messages"""
        print(message)
        log_file.write(message + "\n")
        log_file.flush()
else:
    def log_print(message):
        """In submit mode, just print"""
        print(message)


def process_single_wrapper(data_i, meta_i, body_parts_tracked, fps_lookup, section, arena_data_ref):
    """
    单鼠特征提取的并行包装函数
    显式传递 arena_data 以确保在工作进程中可用
    """
    global arena_data
    arena_data = arena_data_ref
    
    fps_i = _fps_from_meta(meta_i, fps_lookup, default_fps=30.0)
    video_id_i = meta_i['video_id'].iloc[0]
    return transform_single(data_i, body_parts_tracked, fps_i, section, video_id=video_id_i).astype(np.float32)

def process_pair_wrapper(data_i, meta_i, body_parts_tracked, fps_lookup):
    """双鼠特征提取的并行包装函数"""
    fps_i = _fps_from_meta(meta_i, fps_lookup, default_fps=30.0)
    return transform_pair(data_i, body_parts_tracked, fps_i).astype(np.float32)



def main():

    global test, train, solution, arena_data

    # Set optuna and warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore')

    # Logging startup information (only in validate mode)
    if CFG.mode == "validate":
        log_print("="*80)
        log_print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Model: {CFG.model_name}")
        log_print(f"Mode: {CFG.mode}")
        log_print(f"Number of splits: {CFG.n_splits}")
        log_print(f"Log directory: {log_dir}")
        log_print("="*80 + "\n")

    # Data loading and preprocessing
    log_print("Loading data...")
    train = pd.read_csv(CFG.train_path)
    train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
    train_without_mabe22 = train.query("~lab_id.str.startswith('MABe22_')")  # 因为 MABe22_开头的在train_annotation中没有对应的标签

    test = pd.read_csv(CFG.test_path)
    test['n_mice'] = 4 - test[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)




    for col in META_CAT_COLS:
        if col in train.columns:
            # 强制转 str，避免 mixed types 导致 vocab 不稳定
            values = sorted(train[col].dropna().astype(str).unique().tolist())
            META_CAT_CATEGORIES[col] = values
            META_CAT_KNOWN_SET[col] = set(values)

            if META_CAT_ENCODING == "label":
                enc = {v: i for i, v in enumerate(values)}
                META_CAT_ENCODERS[col] = enc
                META_CAT_UNK[col] = len(enc)


    body_parts_tracked_list = list(np.unique(train.body_parts_tracked))  # 提取训练数据所有身体部位追踪配置

    arena_data = pd.concat([
        train[['video_id', 'arena_width_cm', 'arena_height_cm', 'arena_shape']],
        test[['video_id', 'arena_width_cm', 'arena_height_cm', 'arena_shape']]
    ], ignore_index=True).drop_duplicates(subset='video_id').set_index('video_id')

    # Creating solution data for validation mode
    solution = None
    if CFG.mode == 'validate':
        log_print("Creating solution dataframe...")
        solution = create_solution_df(train_without_mabe22)

    # 初始化行为识别的预测阈值
    if CFG.mode == "validate":
        thresholds = {
            "single": {},
            "pair": {}
        }
    else:
        thresholds = joblib.load(f"{CFG.model_path}/{CFG.model_name}/thresholds.pkl")

    # 存储各行为的F1分数和预测结果（包括单鼠和双鼠都存在这里面）
    f1_list = []
    submission_list = []

    # Main processing loop
    log_print(f"\nProcessing {len(body_parts_tracked_list) - 1} body part configurations...\n")
    # 遍历不同身体部位追踪配置，为每种配置筛选数据并按 "单鼠行为" 和 "双鼠交互行为" 分类收集信息，为后续特征提取和模型训练做准备
    for section in range(1, len(body_parts_tracked_list)):
        body_parts_tracked_str = body_parts_tracked_list[section]

        if CFG.mode == 'validate':
            try:
                body_parts_tracked = json.loads(body_parts_tracked_str)
                log_print(f"\n{'='*80}")
                log_print(datetime.now().strftime("%H:%M:%S"))
                log_print(f"Section {section}/{len(body_parts_tracked_list) - 1}: {body_parts_tracked}")
                log_print('='*80)

                if len(body_parts_tracked) > 5:
                    body_parts_tracked = [b for b in body_parts_tracked if
                                          b not in drop_body_parts]  # 过滤掉 drop_body_parts 中定义的冗余部位，减少特征维度，简化模型。

                train_subset = train[train.body_parts_tracked == body_parts_tracked_str]  # 从总训练数据中筛选出使用当前身体部位配置的视频数据

                # 构建帧率查询表。提取每个视频的帧率，去重后转换为字典（video_id 为键，帧率为值），方便后续快速查询视频帧率
                _fps_lookup = (
                    train_subset[['video_id', 'frames_per_second']]
                    .drop_duplicates('video_id')
                    .set_index('video_id')['frames_per_second']
                    .to_dict()
                )

                single_mouse_list = []
                single_mouse_label_list = []
                single_mouse_meta_list = []

                mouse_pair_list = []
                mouse_pair_label_list = []
                mouse_pair_meta_list = []

                for switch, data, meta, label in generate_mouse_data(train_subset, 'train', traintest_directory=CFG.train_tracking_path, generate_single=True, generate_pair=True):
                    # 单鼠行为：添加到单鼠列表
                    if switch == 'single':
                        single_mouse_list.append(data)
                        single_mouse_meta_list.append(meta)
                        single_mouse_label_list.append(label)
                    # 双鼠交互：添加到双鼠列表
                    else:
                        mouse_pair_list.append(data)
                        mouse_pair_meta_list.append(meta)
                        mouse_pair_label_list.append(label)

                

                # 将分散的单鼠行为数据转换为统一的特征矩阵、标签矩阵和元数据
                if len(single_mouse_list) > 0:
                    # Parallel feature extraction for single mice
                    single_feats_parts = Parallel(n_jobs=-1, verbose=1)(
                        delayed(process_single_wrapper)(
                            data_i, meta_i, body_parts_tracked, _fps_lookup, section, arena_data
                        )
                        for data_i, meta_i in zip(single_mouse_list, single_mouse_meta_list)
                    )
                    
                    # 拼接 single_feats_parts，其中存储了每个单鼠视频的特征数据（如运动特征、身体部位位置特征等），拼接后形成训练集
                    X_tr = pd.concat(single_feats_parts, axis=0, ignore_index=True)
                    # 拼接 single_mouse_label_list ，形成统一的标签矩阵
                    single_mouse_label = pd.concat(single_mouse_label_list, axis=0, ignore_index=True)
                    # 拼接 single_mouse_meta_list，形成统一的元数据框。元数据包含视频 ID、帧号、小鼠 ID 等信息，用于关联特征与原始视频上下文
                    single_mouse_meta = pd.concat(single_mouse_meta_list, axis=0, ignore_index=True)

                    # === New: attach meta-based features to X_tr ===
                    X_meta = meta_to_features(single_mouse_meta)
                    X_tr = pd.concat(
                        [X_tr.reset_index(drop=True), X_meta.reset_index(drop=True)],
                        axis=1
                    )
                    # === End new code ===

                    log_print(f"\n[DEBUG] Validate Mode (Single) - Feature Count: {len(X_tr.columns)}")
                    log_print(f"[DEBUG] Validate Mode (Single) - Features: {list(X_tr.columns)}")

                    # 交叉验证训练（单鼠和双鼠分开各训练模型）
                    temp_submission_list, temp_f1_list, temp_thresholds = cross_validate_classifier(X_tr,
                                                                                                    single_mouse_label,
                                                                                                    single_mouse_meta,
                                                                                                    body_parts_tracked_str,
                                                                                                    section)

                    # 检查全局阈值字典 thresholds 中 "单鼠行为（single）" 是否包含当前配置编号（section）的键，
                    # 若不存在则初始化空字典，再将当前配置下的行为阈值（temp_thresholds）存入，统一管理不同配置的阈值。
                    if f"{section}" not in thresholds["single"].keys():
                        thresholds["single"][f"{section}"] = {}
                    for k, v in temp_thresholds.items():
                        thresholds["single"][f"{section}"][k] = v

                    f1_list.extend(temp_f1_list)
                    submission_list.extend(temp_submission_list)

                    

                # 将分散的双鼠交互原始数据转换为模型可直接使用的结构化特征矩阵、标签和元数据
                if len(mouse_pair_list) > 0:
                    # Parallel feature extraction for mouse pairs
                    pair_feats_parts = Parallel(n_jobs=-1, verbose=1)(
                        delayed(process_pair_wrapper)(
                            data_i, meta_i, body_parts_tracked, _fps_lookup
                        )
                        for data_i, meta_i in zip(mouse_pair_list, mouse_pair_meta_list)
                    )

                    X_tr = pd.concat(pair_feats_parts, axis=0, ignore_index=True)  # 前面单鼠定义的X_tr已经过del显式删除
                    mouse_pair_label = pd.concat(mouse_pair_label_list, axis=0, ignore_index=True)
                    mouse_pair_meta = pd.concat(mouse_pair_meta_list, axis=0, ignore_index=True)

                    # === New: attach meta-based features to X_tr ===
                    X_meta = meta_to_features(mouse_pair_meta)
                    X_tr = pd.concat(
                        [X_tr.reset_index(drop=True), X_meta.reset_index(drop=True)],
                        axis=1
                    )
                    # === End new code ===

                    log_print(f"\n[DEBUG] Validate Mode (Pair) - Feature Count: {len(X_tr.columns)}")
                    log_print(f"[DEBUG] Validate Mode (Pair) - Features: {list(X_tr.columns)}")           

                    # 交叉验证训练（单鼠和双鼠分开各训练模型）
                    temp_submission_list, temp_f1_list, temp_thresholds = cross_validate_classifier(X_tr,
                                                                                                    mouse_pair_label,
                                                                                                    mouse_pair_meta,
                                                                                                    body_parts_tracked_str,
                                                                                                    section)

                    # 检查全局阈值字典 thresholds 中 "双鼠交互（pair）" 部分是否包含当前配置编号（section）的键，
                    # 若不存在则初始化空字典，再将当前配置下的行为阈值（temp_thresholds）存入，统一管理不同配置的阈值（方便后续提交模式加载使用）。
                    if f"{section}" not in thresholds["pair"].keys():
                        thresholds["pair"][f"{section}"] = {}
                    for k, v in temp_thresholds.items():
                        thresholds["pair"][f"{section}"][k] = v

                    f1_list.extend(temp_f1_list)
                    submission_list.extend(temp_submission_list)

                    
            except Exception as e:
                log_print(f"\tError in section {section}: {e}")
            log_print("")

        else:
            try:
                body_parts_tracked = json.loads(body_parts_tracked_str)
                log_print(f"\n{'='*80}")
                log_print(f"Section {section}/{len(body_parts_tracked_list) - 1}: {body_parts_tracked}")
                log_print('='*80)

                # 检查 section 是否存在于 thresholds 中，避免 KeyError
                if f"{section}" in thresholds["single"]:
                    temp_submission_list_single = submit(body_parts_tracked_str, 'single', section,
                                                         thresholds["single"][f"{section}"])
                    submission_list.extend(temp_submission_list_single)

                if f"{section}" in thresholds["pair"]:
                    temp_submission_list_pair = submit(body_parts_tracked_str, 'pair', section,
                                                       thresholds["pair"][f"{section}"])
                    submission_list.extend(temp_submission_list_pair)

                
            except Exception as e:
                log_print(f"\tError in section {section}: {e}")
            log_print("")

    # 验证模式的后处理和输出
    if CFG.mode == 'validate':
        log_print("\n" + "="*80)
        log_print("Validation mode - generating metrics...")
        log_print("="*80)
        
        # 将之前训练过程中收集的所有预测结果（submission_list，包含单鼠和双鼠行为的交叉验证预测）
        # 拼接成一个完整的submission数据框，统一存储所有行为的预测时间区间（video_id、agent_id、start_frame等）
        submission = pd.concat(submission_list)
        submission_robust = robustify(submission, train, 'train', traintest_directory=CFG.train_tracking_path)  # 对原始预测结果进行后处理
        # log_print(f"Competition metric: {score(solution, submission_robust, ''):.4f}")

        f1_df = pd.DataFrame(f1_list, columns=['body_parts_tracked_str', 'action',
                                               'binary F1 score'])  # 收集的所有 F1 分数列表（f1_list）转换为 DataFrame
        log_print(f"\nMean F1:            {f1_df['binary F1 score'].mean():.4f}")

        # 将thresholds字典（包含单鼠 / 双鼠各行为的最优阈值，通过tune_threshold函数优化得到）保存到指定路径
        # 这些阈值在后续 "提交模式" 中会被用于将预测概率转换为二分类结果
        joblib.dump(thresholds, f"{CFG.model_name}/thresholds.pkl")
        #  F1 分数 DataFrame（f1_df）保存，留存评估结果，便于后续分析不同配置 / 行为的性能，或用于模型对比。
        joblib.dump(f1_df, f"{CFG.model_name}/scores.pkl")
        log_print(f"Saved thresholds and scores to {CFG.model_name}/")
        
        # === 输出程序内存峰值 (Linux Optimized) ===
        import resource
        # resource.RUSAGE_SELF 获取当前进程的资源使用情况
        # ru_maxrss 在 Linux 下的单位是 KB (Kilobytes)
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        log_print(f"\n[System] Peak Memory Usage: {peak_kb / 1024 / 1024:.2f} GB")
        
        log_print(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print("="*80)
        
        # Close log file
        log_file.close()
        print(f"\nLog saved to: {log_dir}")

    # 提交模式
    elif CFG.mode == 'submit':
        print("\n" + "="*80)
        print("Submit mode - generating submission file...")
        print("="*80)
        # 若submission_list（之前收集的所有测试集预测结果列表）非空，
        # 通过pd.concat将列表中所有分散的预测结果 DataFrame 拼接成一个完整的submission数据框，存储所有预测的时间区间信息。
        if len(submission_list) > 0:
            submission = pd.concat(submission_list)
        # 若submission_list为空（可能因模型未生成任何预测），则创建一个默认 DataFrame 作为 "占位符"，含必要字段和示例数据。为了避免提交文件为空
        else:
            submission = pd.DataFrame(
                dict(
                    video_id=438887472,
                    agent_id='mouse1',
                    target_id='self',
                    action='rear',
                    start_frame=278,
                    stop_frame=500
                ), index=[44])

        submission_robust = robustify(submission, test, 'test', traintest_directory=CFG.test_tracking_path)
        submission_robust.index.name = 'row_id'
        submission_robust.to_csv('submission.csv')
        print("Submission file saved to submission.csv")
        print(f"\nFirst 5 rows:\n{submission.head(5)}")

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()