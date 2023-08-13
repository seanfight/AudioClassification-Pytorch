import numpy as np
from torch.utils.data import Dataset
from macls.data_utils.audio import AudioSegment
from macls.data_utils.augmentor.augmentation import AugmentationPipeline
from macls.utils.logger import setup_logger
logger = setup_logger(__name__)
# 重写customdataset类，构建数据集
class CustomDataset(Dataset):
    # init返回空
    def __init__(self,
                 data_list_path,
                 do_vad,
                 max_duriation,
                 min_duriation,
                 augmentation_config = '{}',
                 mode='train',
                 sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20) -> None:
        super().__init__()
        self.do_vad = do_vad
        self.max_duriation = max_duriation
        self.min_duriation = min_duriation
        self.mode = mode
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        # 音量增强的配置文件
        self._augmentation_config = AugmentationPipeline(augmentation_config=augmentation_config)
