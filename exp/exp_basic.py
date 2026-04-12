import os
import torch

from models import gwk, VectorOsc
from models.AnchorMoE import Model as AnchorMoE
from models.newmodel import Model as newmodel
from models.newmodel_ablation import create_ablation_model
from models.othermodel.patchTST.PatchTST import Model as PatchTST
from models.othermodel.TimesNet.TimesNet import Model as TimesNet


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "gwk": gwk,
            "VecGWK": gwk,
            "VectorOsc": VectorOsc,
            "newmodel": newmodel,
            "AnchorMoE": AnchorMoE,
            "PatchTST": PatchTST,
            "TimesNet": TimesNet,
            "newmodel_ablation": None,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        if self.args.model == "newmodel_ablation":
            ablation_type = getattr(self.args, "ablation_type", "full")
            model = create_ablation_model(self.args, ablation_type).float()
        elif self.args.model in self.model_dict:
            model_module = self.model_dict[self.args.model]
            cls = getattr(model_module, "Model", model_module)
            model = cls(self.args).float()
        else:
            raise NotImplementedError(
                f"Model {self.args.model} not implemented in exp_basic.py"
            )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        elif self.args.use_gpu and self.args.gpu_type == "mps":
            device = torch.device("mps")
            print("Use GPU: mps")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
