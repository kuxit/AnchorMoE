import pandas as pd
from torch.utils.data import DataLoader

from data_provider.data_loader import Normalizer, UEAloader, collate_fn


def data_provider(args, flag):
    if args.data != "UEA":
        raise ValueError("The open-source release only includes UEA classification.")

    norm_scope = str(getattr(args, "uea_norm_scope", "train")).lower()
    norm_type = str(getattr(args, "uea_norm_type", "standardization")).lower()
    normalizer = None

    if norm_scope == "all":
        normalizer = getattr(args, "_uea_all_normalizer", None)
        if normalizer is None:
            train_raw = UEAloader(args.root_path, flag="TRAIN", norm_type=norm_type, normalize=False)
            test_raw = UEAloader(args.root_path, flag="TEST", norm_type=norm_type, normalize=False)
            normalizer = Normalizer(norm_type=norm_type)
            normalizer.normalize(pd.concat([train_raw.feature_df, test_raw.feature_df], axis=0))
            setattr(args, "_uea_all_normalizer", normalizer)
    elif norm_scope == "train" and str(flag).upper() != "TRAIN":
        normalizer = getattr(args, "_uea_train_normalizer", None)

    data_set = UEAloader(
        args.root_path,
        flag=flag,
        normalizer=normalizer,
        norm_type=norm_type,
    )
    if norm_scope == "train" and str(flag).upper() == "TRAIN":
        setattr(args, "_uea_train_normalizer", data_set.normalizer)

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=(str(flag).upper() == "TRAIN"),
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=lambda batch: collate_fn(batch, max_len=args.seq_len),
    )
    return data_set, data_loader
