from setuptools import setup

setup(
    name = "streampy",
    version = "0.0.1",
    description = "Java 8 Stream API/Collectors/Function Interface/Optional 在 Python 3.6+ 的实现",
    author = "17097231932",
    author_email = "17097231932@163.com",
    url = "https://github.com/17097231932/Stream.py",
    py_modules=['stream'],
    long_description = """# Stream.py
Java 8 Stream API/Collectors/Function Interface/Optional 在 Python 3.6+ 的实现

## Stream.py 是什么？

是一个处理数据的工具，它使用声明式的API，简单易读

## 简介

Java 8 Stream API/Collectors/Function Interface/Optional 在 Python 3.6+ 的实现

Stream是一个在某些数据上的抽象视图. 比如, Stream可以是一个list或者

文件中的几行或者其他任意的一个元素序列的视图. Stream API提供可以顺序

表现或者并行表现的操作总和. 开发者需要明白一点, Stream 是一种更高阶的

抽象概念, 而不是一种数据结构. Stream 不会储存数据, Stream 天生就很懒, 

只有在被使用到时才会执行计算.它允许我们产生无限的数据流(stream of data).


在我看来，Stream.py 的代码更好主要有以下几点原因: 



1. Stream.py代码能够清晰地表达开发者对数据过滤, 排序等操作的意图. 



2. 通过使用Stream API格式的更高抽象, 开发者表达他们所想要的是

什么而不是怎么去得到这些结果. 



3. Stream API为数据处理提供一种统一的语言, 使得开发者在谈论数

据处理时有共同的词汇. 当两个开发者讨论filter函数时, 你都会

明白他们都是在进行一个数据过滤操作. 



4. 开发者不再需要为实现数据处理而写的各种样板代码, 也不再需要

为loop代码或者临时集合来储存数据的冗余代码, Stream API会

处理这一切. 



5. Stream不会修改潜在的集合, 它是非交换的. 



注意: 不支持并行

      因为在CPU密集的情况下, Python GIL只允许一个线程执行, 

      所以并行意义不大, 而且如果在多线程情况下, 数据共享和

      内存占用得不偿失, 所以不支持并行及其相关函数.



提示: 和 Java 8 Stream API 的不同之处:

    1. reduce 只是一个中间操作

    2. Collectors 中的内容只是一个函数, 而非实例

    3. 由于 Python 是动态类型, 所以函数式接口靠实例化指定类型

    4. LongStream 是 IntStream

 

用法
````python

print(["Apple", "Banana", "Blackberry", "Coconut", "Avocado", "Cherry", "Apricots"]
    .stream()
    .collect(Collectors.groupingBy(lambda s: s[0], Collectors.toList()))
)    # {'A': ['Apple', 'Avocado', 'Apricots'], 'B': ['Banana', 'Blackberry'], 'C': ['Coconut', 'Cherry']}

print(["APPL:Apple", "MSFT:Microsoft"]
    .stream()
    .collect(Collectors.toMap(lambda s: s[:s.find(':')], lambda s: s[s.find(':') + 1:]))
)    # {'APPL': 'Apple', 'MSFT': 'Microsoft'}

print(["Orange", "apple", "Banana"]
    .stream()
    .sorted(lambda x, y: ord(x[0]) > ord(y[0]))
    .collect(Collectors.toList())
)   # ['Banana', 'Orange', 'apple']

print(["A", "B", "A", "C", "B", "D"]
    .stream()
    .distinct()
    .collect(Collectors.toList())
)    # ['B', 'A', 'C', 'D']

print(("A", "B", "C", "D", "E", "F")
    .stream()
    .skip(2)
    .limit(3)
    .collect(Collectors.toList())
)    # ['C', 'D', 'E']

````


CLI

````
$ stream -h

选项: stream [-h] [-f FILE] [-i IMPORTS] command

stream.py: Java 8 Stream API 在 Python 3.6+ 的实现

参数:
  command               在输入流中执行的命令
  -h, --help            显示帮助
  -f file               处理的文件
  -i imports            导入模块

模块导入格式:
    语法: <模块>:<对象>:<别名>[;<更多导入>[;<更多导入> ...]]
    例子:
        'import os' = '-i os'
        'import os as op_sys' = '-i os::op_sys'
        'from os import environ' = '-i os:environ'
        'from os import environ as env' = '-i os:environ:env'
        'import sys, os' = '-i os;sys'

命令格式:

    特殊变量 _ 为Stream实例
````
"""
    
)
