def t1(param1,*param2):
    print(param1)
    print(type(param2))
    # print(*param2)
t1(1,2,3,4)
# 输出
# 1
# (2, 3, 4) 以元组的方式出现

def t2(param1,**param2):
    print(param1)
    print(param2)
# t2(1,2,3,4)
# Exception has occurred: TypeError
# t2() takes 1 positional argument but 4 were given
#   File "/home/AudioClassification-Pytorch/learn/1.关于参数*和**.py", line 13, in <module>
#     t2(1,2,3,4)
# TypeError: t2() takes 1 positional argument but 4 were given

# t2(1,a=2,b=3,c=4)
# 1
# {'a': 2, 'b': 3, 'c': 4}

# 一个*还可以解压参数列表
def t3(param1,param2):
    print(param1,param2)
a = [1,2]
# t3(*a)
# 1 2

# 还可以同时使用
def t4(a=0,b=1,*args,**kwargs):
    print(a)
    print(b)
    print(args)
    print(kwargs)

# t4(1,2,4,5,6,c=1,d=3)
# 1
# 2
# (4, 5, 6)
# {'c': 1, 'd': 3}