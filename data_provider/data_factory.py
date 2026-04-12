from data_provider.data_loader import UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader


from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

try:
    from data_provider.pattern_injection import PatternInjectionLoader
except ImportError:
    PatternInjectionLoader = None

try:
    from data_provider.synthetic_importance import SyntheticImportanceDataset
except ImportError:
    SyntheticImportanceDataset = None

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'FordA': UEAloader,
}
if PatternInjectionLoader is not None:
    data_dict['PatternInjection'] = PatternInjectionLoader
if SyntheticImportanceDataset is not None:
    data_dict['SyntheticImportance'] = SyntheticImportanceDataset


def data_provider(args, flag):
    Data = data_dict[args.data]
    embed = getattr(args, 'embed', 'timeF')
    timeenc = 0 if embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = getattr(args, 'freq', 'h')

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        if args.data == 'PatternInjection' and PatternInjectionLoader is not None:
            data_set = PatternInjectionLoader(
                root_path=args.root_path,
                flag=flag,
                seq_len=getattr(args, 'seq_len', 200),
                motif_len=getattr(args, 'pattern_motif_len', 20),
                n_train=getattr(args, 'pattern_n_train', 800),
                n_test=getattr(args, 'pattern_n_test', 400),
                seed=getattr(args, 'seed', 42),
            )
        elif args.data == 'SyntheticImportance' and SyntheticImportanceDataset is not None:
            data_set = SyntheticImportanceDataset(
                root_path=getattr(args, 'root_path', ''),
                flag=flag,
                seq_len=getattr(args, 'seq_len', 100),
                n_channels=getattr(args, 'synthetic_n_channels', 4),
                n_classes=getattr(args, 'synthetic_n_classes', 2),
                n_important_segments=getattr(args, 'synthetic_n_segments', 2),
                segment_len=getattr(args, 'synthetic_segment_len', 15),
                n_train=getattr(args, 'synthetic_n_train', 400),
                n_test=getattr(args, 'synthetic_n_test', 200),
                noise_sigma=getattr(args, 'synthetic_noise_sigma', 0.35),
                signal_amplitude=getattr(args, 'synthetic_signal_amplitude', 1.2),
                seed=getattr(args, 'seed', 42),
            )
        else:
            data_set = Data(
                root_path=args.root_path,
                flag=flag,
            )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader


# def data_provider(args, flag, bin_edges=None):
#     Data = data_dict[args.data]
#     timeenc = 0 if args.embed != 'timeF' else 1

#     # 兼容大小写的flag检查
#     flag_lower = flag.lower() if flag else ''
#     if flag_lower in ['test', 'val', 'vali']:
#         shuffle_flag = False
#         drop_last = True
#         if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
#             batch_size = args.batch_size
#         else:
#             batch_size = 1  # bsz=1 for evaluation
#         freq = args.freq
#     else:
#         shuffle_flag = True
#         drop_last = True
#         batch_size = args.batch_size  # bsz for train and valid
#         freq = args.freq


#     if args.task_name == 'classification':
#         drop_last = False
#         data_set = Data(
#             root_path=args.root_path,
#             flag=flag,
#         )

#         data_loader = DataLoader(
#             data_set,
#             batch_size=batch_size,
#             shuffle=shuffle_flag,
#             num_workers=args.num_workers,
#             drop_last=drop_last,
#             collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
#         )
#         return data_set, data_loader
