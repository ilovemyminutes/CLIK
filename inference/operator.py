from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import requests
import torch
from networks import CLIK
from preprocessing import TextPreprocessor
from utils.data_utils import txt_input_to_device


class TaskOperator:
    def __init__(
        self,
        model: CLIK,
        img_transforms: A.Compose,
        txt_preprocessor: TextPreprocessor = None,
        device: torch.device = None,
    ):
        self.model = model
        self.img_transforms = img_transforms
        self.txt_preprocessor = txt_preprocessor
        self.device = device if device is not None else torch.device("cpu")

        self.model.to(self.device)
        if self.model.training:
            self.model.eval()
        assert (
            not self.model.training
        ), "model is on train mode. please activate '.eval()'"

    def __call__(self, plan_attrs, prod_paths, topk: int = 1):
        return self.infer(plan_attrs, prod_paths, topk)

    def infer(
        self,
        plan_attrs: Dict[str, str],
        prod_paths: Union[List[int], List[str], List[np.array]],
        topk: int = 1,
    ) -> Tuple[int, str]:
        batch = self.collate(plan_attrs, prod_paths)
        logits = self.model.predict(batch)
        _, indices = torch.topk(logits.squeeze(), k=topk)
        output = [prod_paths[i] for i in indices.tolist()]
        return output

    def collate(
        self, plan_attrs: Dict[str, str], prod_paths: Union[List[str], List[np.array]]
    ):
        if not isinstance(prod_paths, list):
            prod_paths = [prod_paths]

        # collate batch
        if plan_attrs.get("date", None) is None:
            plan_attrs["date"] = self.today()

        plc = txt_input_to_device(
            self.txt_preprocessor(**plan_attrs, expand_dim=True), device=self.device
        )
        pri = self.read_transform_imgs(prod_paths).to(self.device)
        batch = dict(contexts=plc, instances=pri)
        return batch

    def read_transform_imgs(
        self, prod_paths: Union[List[str], List[np.array]]
    ) -> torch.Tensor:
        if isinstance(prod_paths[0], str):
            imgs = []
            for path in prod_paths:
                if path.startswith("http"):  # load by request
                    img_arr = np.array(
                        bytearray(requests.get(path).content), dtype=np.uint8
                    )
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                else:  # load from local
                    img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(self.img_transforms(image=img)["image"])

        elif isinstance(prod_paths[0], np.ndarray):
            imgs = [self.img_transforms(image=img)["image"] for img in prod_paths]

        imgs = torch.stack(imgs).to(self.device)
        return imgs

    @staticmethod
    def today() -> str:
        t = datetime.today() + timedelta(hours=9)  # korean time
        t = t.strftime("%Y-%m-%d")
        return t
