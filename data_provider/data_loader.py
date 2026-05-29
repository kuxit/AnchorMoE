import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import Dataset


class Normalizer:
    """Feature normalizer for UEA time-series data."""

    def __init__(self, norm_type="standardization", mean=None, std=None, min_val=None, max_val=None):
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)
        if self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)
        if self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / (grouped.transform("std") + np.finfo(float).eps)
        if self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (grouped.transform("max") - min_vals + np.finfo(float).eps)
        raise ValueError(f"Unknown norm_type: {self.norm_type}")


def _subsample_if_needed(y, limit=256, factor=2):
    if len(y) > limit:
        return y[::factor].reset_index(drop=True) if hasattr(y, "reset_index") else y[::factor]
    return y


class UEAloader(Dataset):
    """UEA multivariate time-series classification loader.

    Expected directory:
        root_path/
            DatasetName_TRAIN.ts
            DatasetName_TEST.ts
    """

    def __init__(self, root_path, flag="TRAIN", normalizer=None, norm_type="standardization", normalize=True):
        self.root_path = root_path
        self.all_df, self.labels_df = self._load_all(root_path, flag=flag)
        self.all_IDs = self.all_df.index.unique()
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df
        self.normalizer = normalizer if normalizer is not None else Normalizer(norm_type=norm_type)
        if normalize:
            self.feature_df = self.normalizer.normalize(self.feature_df)

    def _load_all(self, root_path, flag):
        data_paths = glob.glob(os.path.join(root_path, "*"))
        if flag is not None:
            data_paths = [p for p in data_paths if re.search(flag, p)]
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith(".ts")]
        if not input_paths:
            raise FileNotFoundError(f"No .ts file found in {root_path!r} with flag {flag!r}")
        return self._load_single(input_paths[0])

    def _load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(
            filepath,
            return_separate_X_and_y=True,
            replace_missing_vals_with="NaN",
        )
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int64)

        elemwise = getattr(df, "map", None) or df.applymap
        lengths = elemwise(lambda x: len(x)).values
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))
        if np.sum(horiz_diffs) > 0:
            df = elemwise(_subsample_if_needed)
            elemwise = getattr(df, "map", None) or df.applymap
            lengths = elemwise(lambda x: len(x)).values

        target_len = int(np.max(lengths[:, 0]))
        self.max_seq_len = target_len

        def pad_truncate(x):
            arr = np.asarray(x, dtype=float)
            if arr.shape[0] == target_len:
                return arr
            if arr.shape[0] > target_len:
                return arr[:target_len]
            out = np.full((target_len,), np.nan, dtype=float)
            out[: arr.shape[0]] = arr
            return out

        rows = []
        for row in range(df.shape[0]):
            sample = {col: pad_truncate(df.at[row, col]) for col in df.columns}
            sample_df = pd.DataFrame(sample)
            sample_df.index = pd.Index([row] * target_len)
            rows.append(sample_df)

        all_df = pd.concat(rows, axis=0)
        groups = []
        for _, group in all_df.groupby(all_df.index):
            groups.append(group.interpolate(limit_direction="both", axis=0).ffill().bfill())
        all_df = pd.concat(groups, axis=0)
        return all_df, labels_df

    def __getitem__(self, index):
        x = torch.from_numpy(self.feature_df.loc[self.all_IDs[index]].values.copy()).float()
        y = torch.from_numpy(self.labels_df.loc[self.all_IDs[index]].values.copy()).long()
        return x, y

    def __len__(self):
        return len(self.all_IDs)


_cached_masks = {}


def padding_mask(lengths, max_len=None):
    key = (tuple(lengths.tolist()), max_len)
    if key in _cached_masks:
        return _cached_masks[key]
    max_len = int(max_len or lengths.max())
    mask = torch.arange(0, max_len).expand(lengths.numel(), -1).lt(lengths.unsqueeze(1))
    _cached_masks[key] = mask
    return mask


def collate_fn(data, max_len=None):
    features, labels = zip(*data)
    lengths = [x.shape[0] for x in features]
    max_len = int(max_len or max(lengths))
    x = torch.zeros(len(features), max_len, features[0].shape[-1])
    for i, feat in enumerate(features):
        end = min(lengths[i], max_len)
        x[i, :end] = feat[:end]
    y = torch.stack(labels, dim=0)
    mask = padding_mask(torch.tensor(lengths, dtype=torch.int64), max_len=max_len)
    return x, y, mask
