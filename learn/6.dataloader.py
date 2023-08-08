# dataloader调用customdataset的__getitem__()的方法
# 组合成patch，dataset+sample，并在数据集上提供单线程和
# num_works的可迭代对象，在dataloader的多个参数，
# 参数解释如下：
# epoch：所有数据样本输入一个epoch
# 这里颇为重要的是“Epoch、Iteration
# 和Batchsize”之间的关系：
# 1）Epoch表示所有训练样本都已输入到模型中，
# 记为一个Epoch；
# 2）Iteration表示一批样本输入到模型中，
# 记为一个Iteration；
# 3）Batchsize表示批大小，决定一个Epoch中有多少个Iteration。
# 当样本数可以被Batchsize整除时，三者成立关系，即全体样本分成Batchsize分批次输入模型，每批次记为一次Iteration。
# 若样本总数80个，当Batchsize=8时，可以知道“1 Epoch = 10 Iteration”。
# 若样本总数87个，当Batchsize-8时，可以知道：1）若“drop_last=True”，则“1 Epoch = 10
# Iteration”；2）若“drop_last=False”，则“1 Epoch = 11 Iteration”，其最后一个Iteration时样本个数为7，小于既定Batchsize。
#  epoch：所有的训练样本输入到模型中称为一个epoch； 
#  2. iteration：一批样本输入到模型中，成为一个Iteration;
#  3. batchszie：批大小，决定一个epoch有多少个Iteration；
#  4. 迭代次数（iteration）=样本总数（epoch）/批尺寸（batchszie）
#  5. dataset (Dataset) – 决定数据从哪读取或者从何读取；
#  6. batch_size (python:int, optional) – 批尺寸(每次训练样本个数,默认为１）
#  7. shuffle (bool, optional) –每一个 epoch是否为乱序 (default: False)；
#  8. num_workers (python:int, optional) – 是否多进程读取数据（默认为０);
#  9. drop_last (bool, optional) – 当样本数不能被batchsize整除时，最后一批数据是否舍弃（default: False)
#  10. pin_memory（bool, optional) - 如果为True会将数据放置到GPU上去（默认为false）

