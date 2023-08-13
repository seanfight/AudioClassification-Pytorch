import numpy as np
# import dataset
from torch.utils.data import Dataset

# 导入声音处理函数
from macls.data_utils.audio import AudioSegment
from macls.data_utils.augmentor.augmentation import AugmentationPipeline
from macls.utils.logger import setup_logger

# 重写customdataset类，构建数据集
class CustomDataset(Dataset):
    # 初始化
    def __init__(self,
                 data_list_path,
                 do_vad=True,
                 max_duriatin=3,
                 min_duriation=0.5,
                 augmentation_config='{}',
                 mode='train',
                 sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20):
        """音频加载器
        data_list_path:包含音频路径和标签的数据
        do_vad:是否对音频进行语音活动检测 vad来剪裁静音部分
        max_duriation:最长的音频长度，超过剪裁
        min_duriation:最短的音频长度，过滤掉
        augmentation_config:用于指定音频增器的配置
        mode:数据集模式，在训练模式下，数据可能需要预处理
        sample_rate:采样率
        use_dB_normalization:是否对音频进行音量归一化处理
        target_dB:音量归一化大小
        
        """
        super(CustomDataset,self).__init__()
        self.do_
        


    # 返回数据集参数
    def __getitem__(self,idx):
        pass

    # 返回数据集长度
    def __len__(self):
        return len(self.lines)