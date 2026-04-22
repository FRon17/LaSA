# 负责定义训练作业的主要代码，包含训练作业的配置、训练流程的管理和训练过程的执行等功能。它从配置文件中加载训练流程，并依次执行每个训练过程，支持多种不同类型的训练过程，如VAE、Slider、LoRA Hack等。
import json
import os

from jobs import BaseJob
from toolkit.kohya_model_util import load_models_from_stable_diffusion_checkpoint
from collections import OrderedDict
from typing import List
from jobs.process import BaseExtractProcess, TrainFineTuneProcess
from datetime import datetime

# 作用：将配置文件中的字符串标识映射到具体的训练过程类名
# 关注点：当你需要添加新的训练类型时，需要在这里注册映射关系
process_dict = {
    'vae': 'TrainVAEProcess',
    'slider': 'TrainSliderProcess',
    'slider_old': 'TrainSliderProcessOld',
    'lora_hack': 'TrainLoRAHack',
    'rescale_sd': 'TrainSDRescaleProcess',
    'esrgan': 'TrainESRGANProcess',
    'reference': 'TrainReferenceProcess',
}


class TrainJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.training_folder = self.get_conf('training_folder', required=True) # 训练文件夹路径，必填项
        self.is_v2 = self.get_conf('is_v2', False) # 是否使用Stable Diffusion 2.x版本，默认为False
        self.device = self.get_conf('device', 'cpu') # 训练设备，默认为'cpu'，可以设置为'cuda'或其他设备

        # self.gradient_accumulation_steps = self.get_conf('gradient_accumulation_steps', 1)
        # self.mixed_precision = self.get_conf('mixed_precision', False)  # fp16

        self.log_dir = self.get_conf('log_dir', None) # 日志目录 用于TensorBoard等可视化工具

        # loads the processes from the config
        self.load_processes(process_dict) # 加载并实例化训练过程列表

    def run(self):
        super().run()
        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()
