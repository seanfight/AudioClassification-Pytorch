#partial,主要是扩展函数参数的功能
# /ˈpɑːʃ(ə)l/
# 美
# /ˈpɑːrʃ(ə)l/
# 全球发音
# 简明 牛津 新牛津 韦氏 柯林斯 例句 百科
# adj.
# 部分的，不完全的；偏袒的，不公平的；偏爱的
# n.
# （乐）分音，泛音
import functools

def f1(*parma):
    print(sum(parma))
f1_add = functools.partial(f1,100)
f1_add_2 = functools.partial(f1,90)
# f1_add(1,2,3)
# 106
# f1_add_2(1,2,3)

# 类fun = functools.partial(func,*args,**kwargs)
# func:需要被扩展的函数
# args：需要被固定的位置参数
# kwarg：需要被固定的关键字参数
def add(*args,**kwargs):
    for i in args:
        print(i)
    for k,v in kwargs.items():
        print("{0}:{1}".format(k,v))
add(1,2,3,a=5,b=8)
add_partial = functools.partial(add,12,a=122,b=122)
# add_partial(1,23,3,c=1,a=2)
# 1
# 2
# 3
# a:5
# b:8
# 12
# 1
# 23
# 3
# a:2
# b:122
# c:1
# ba = functools.partial(int,base = 2)
# print(ba("1010"))