import argparse
# 使用functools中的偏函数partial，用于扩展函数的功能
import functools

from macls.trainer import MAClsTrainer

from macls.utils.utils import add_arguments, print_arguments
parser = argparse.ArgumentParser(description=__doc__)
# 需要被扩展的函数，add_arguments,parser的add_argument在函数内执行
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('configs',          str,    'configs/ecapa_tdnn.yml',      '配置文件')
add_arg("local_rank",       int,    0,                             '多卡训练需要的参数')
# add_arg("use_gpu",          bool,   False,                          '是否使用GPU')
add_arg("use_gpu",          bool,   True,                          '是否使用GPU训练')
add_arg('augment_conf_path',str,    'configs/augmentation.json',   '数据增强的配置文件，为json格式')
add_arg('save_model_path',  str,    'models/',                  '模型保存的路径')
add_arg('resume_model',     str,    None,                       '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model', str,    None,                       '预训练模型的路径，当为None则不使用预训练模型')
# 解析参数到args
args = parser.parse_args()
# 打印额外参数
print_arguments(args=args)

# 获取训练器 test
trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)

# trainer.train(save_model_path=args.save_model_path,
#               resume_model=args.resume_model,
#               pretrained_model=args.pretrained_model,
#               augment_conf_path=args.augment_conf_path)
# trainer.evaluate(save_matrix_path='docs/images')
trainer.export()
print("---")