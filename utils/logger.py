import os
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


class Logger:
    def __init__(self, start_epoch: int = 0, start_step: int = 0):
        self.resume_from(start_epoch, start_step)
        self.names: Optional[List[str]] = None
        self.logs: Optional[Dict[str, List[float]]] = None
        self.__logs_per_step: Optional[Dict[str, List[float]]] = None
        self.__logs_per_epoch: Optional[Dict[str, List[float]]] = None

        self.start_epoch: Optional[int] = None
        self.start_step: Optional[int] = None
        self.last_step: Optional[int] = None

    def register(self, names: list) -> None:
        self.names = names
        self.logs = {n: [] for n in names}
        self.__logs_per_step = {n: [] for n in names}
        self.__logs_per_epoch = {n: [] for n in names}

    def record(self, logs: dict):
        """매 step마다 log를 기록"""
        for n in self.names:
            v = logs.get(n, np.nan)
            self.logs[n].append(v)

            if n == "step" and v is not np.nan:
                self.last_step = v

    def update(self):
        """지금까지의 log를 바탕으로 통계량 집계"""
        for n in self.names:
            self.__logs_per_step[n].extend(self.logs[n])
            if n == "step":
                continue
            elif n == "epoch":
                self.__logs_per_epoch[n].append(int(np.nanmean(self.logs[n])))
            elif isinstance(self.logs[n][0], list):
                num_hits, tot_length = 0, 0
                for step_log in self.logs[n]:
                    if step_log is not np.nan:
                        num_hits += sum(step_log)
                        tot_length += len(step_log)
                self.__logs_per_epoch[n].append(num_hits / tot_length)
            else:
                self.__logs_per_epoch[n].append(np.nanmean(self.logs[n]))

        self.initialize_logs()

    def save(self, save_dir: str):
        """save_dir에 log 파일을 저장"""
        # epoch 단위 logs
        logs_per_epoch_to_save = pd.DataFrame(self.logs_per_epoch)

        # step 단위 logs
        logs_per_step_to_save = {
            k: v for k, v in self.__logs_per_step.items() if not isinstance(v[0], list)
        }
        logs_per_step_to_save = pd.DataFrame(logs_per_step_to_save)
        logs_per_epoch_to_save.to_csv(
            os.path.join(save_dir, "logs_per_epoch.csv"), index=False
        )
        logs_per_step_to_save.to_csv(
            os.path.join(save_dir, "logs_per_step.csv"), index=False
        )

    def return_last_logs(self, by: str = "epoch"):
        """가장 최근의 log 데이터를 return"""
        if by == "epoch":
            output = {
                n: self.__logs_per_epoch[n][-1] for n in self.names if n != "step"
            }
            output["step"] = self.last_step
        else:
            output = {n: self.__logs_per_step[n][-1] for n in self.names}
            output["step"] = self.last_step
        return output

    def return_logs(self, by: str = "epoch"):
        """현재까지 축적된 log 데이터를 return"""
        if by == "epoch":
            return self.__logs_per_epoch
        elif by == "step":
            return self.__logs_per_step

    def initialize_logs(self):
        self.logs = {n: [] for n in self.names}

    def resume_from(self, epoch: int, step: int):
        self.start_epoch = epoch
        self.start_step = step
        self.last_step = step

    @property
    def logs_per_epoch(self):
        output = {k: v for k, v in self.__logs_per_epoch.items() if k != "step"}
        return output

    @property
    def logs_per_step(self):
        return self.__logs_per_step
