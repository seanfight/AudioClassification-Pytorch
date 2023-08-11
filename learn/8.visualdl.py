# add_scalar接口介绍
# add_scalar(tag,value,step,walltime=None)
# 参数介绍
# tag   string  记录指标的标志，无%
# value float   要记录的数据
# step  int     记录的步数
# walltime   int    时间戳
# tag的用法，/前为父，/后为子tag

from visualdl import LogWriter

if __name__ == '__main__':
    value = [i/1000 for i in range(1000)]
    # 初始化一个记录器
    with LogWriter(logdir="learn/train") as writer:
        for step in range(1000):
            # 像记录器添加一个tag为acc的数据
            writer.add_scalar(tag="acc",step=step,value=value[step])
            writer.add_scalar(tag="loss",step=step,value=value[step]-2)
    
# visualdl --logdir ./train --port 8080
# 然后打开http://localhost:8080/这个网址







