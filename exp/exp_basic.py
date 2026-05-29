import os

import torch

from models.AnchorMoE import Model as AnchorMoE


class ExpBasic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {"AnchorMoE": AnchorMoE, "AnchorMoE_CR": AnchorMoE}
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        if self.args.model not in self.model_dict:
            raise ValueError(f"Unknown model {self.args.model!r}. Use AnchorMoE.")
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
