#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Stream.js
    ~~~~~~~~~~~~~~~~~~

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
'''

cli_help = '''\
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
    特殊变量 _ 为Stream实例\
'''

import sys
from collections import deque
from collections.abc import Iterable
from functools import reduce


def _count(items):
    return sum((1 for _ in items))

def _averaging(items):
    count = 0
    sum = 0
    for item in items:
        count += 1
        sum += item
    return sum / count


class Collectors:
    '''集合转换工具集'''

    @classmethod
    def of(cls, supplier, accumulator, combiner=None, finisher=None):
        def collector(items):
            items = list(items)
            sup = supplier()
            for item in items:
                accumulator(sup, item)
            if combiner:
                sup = combiner(sup, supplier())
            if finisher:
                sup = finisher(sup)
            return sup
        return collector

    def __getattribute__(self, name: str):
        func = object.__getattribute__(self, name)
        if type(func) == type(lambda s:s):
            func = staticmethod(func)
        return func
    
    
    def toList():
        def wrappers(items):
            return list(items)
        return wrappers
    
    
    def toSet():
        def wrappers(items):
            return set(items)
        return wrappers
    
    
    def toArray(types):
        def wrappers(items):
            result = []
            for item in items:
                result.append(types(item))
            return result
        return wrappers
    
    
    def toMap(keys, values):
        def wrappers(items):
            result = {}
            for item in items:
                key = keys(item)
                value = values(item)
                result[key] = value
            return result
        return wrappers
    
    
    def groupingBy(keys, outtype):
        '''分组
        
        参数:
            keys (function) 分组的键值
            outtype (Any)   输出类型的构造函数'''
        def wrappers(items):
            map = {}
            for item in items:
                key = keys(item)
                if key in map:
                    map[key].append(item)
                else:
                    map[key] = [item]
            result = {}
            for key in map.keys():
                result[key] = outtype(map[key])
            return result
        return wrappers

    def partitioningBy(keys, outtype):
        '''分区
        
        参数:
            keys (function) 分区的判断函数, T_ -> boolean
            outtype (Any)   输出类型的构造函数'''
        def wrappers(items):
            map = {True: [], False: []}
            for item in items:
                key = keys(item)
                if key in (True, False):
                    map[key].append(item)
                else:
                    raise TypeError('keys return value must is true or flase')
            result = {}
            for key in map.keys():
                result[key] = outtype(map[key])
            return result
        return wrappers
    
    def collectingAndThen(strategy, method):
        def wrappers(items):
            items = strategy(items)
            return method(items)
        return wrappers
    
    def mapping(keys, outtype):
        def wrappers(items):
            result = []
            for item in items:
                value = keys(item)
                result.append(value)
            return outtype(result)
        return wrappers


    def toCollection(types):
        def wrappers(items):
            return types(items)
        return wrappers
    

    def counting():
        def wrappers(items):
            return _count(items)
        return wrappers

    def reducing(func):
        def wrappers(items):
            return reduce(func, items)
        return wrappers    

    def averagingInt():
        def wrappers(items):
            return int(_averaging(items))
        return wrappers

    
    def averagingDouble():
        def wrappers(items):
            return float(_averaging(items))
        return wrappers

    averagingLong = averagingInt
    
    def summingInt():
        def wrappers(items):
            return int(sum(items))
        return wrappers

    
    def summingDouble():
        def wrappers(items):
            return float(sum(items))
        return wrappers

    summingLong = summingInt

    def summarizingInt():
        def wrappers(items):
            return IntSummaryStatistics(items)
        return wrappers

    def summarizingDouble():
        def wrappers(items):
            return DoubleSummaryStatistics(items)
        return wrappers
    
    summarizingLong = summarizingInt

    def joining(sep, start='', stop=''):
        def wrappers(items):
            return start + sep.join(items) + stop
        return wrappers
    
    def maxBy(com):
        def wrappers(items):
            items = list(items)
            if len(items) == 0:
                return Optional(None)
            try:
                return Optional(reduce(lambda x, y: max(com(x), com(y)), items))
            except Exception:
                return Optional(None)
        return wrappers
    
    def minBy(com):
        def wrappers(items):
            items = list(items)
            if len(items) == 0:
                return Optional(None)
            try:
                return Optional(reduce(lambda x, y: min(com(x), com(y)), items))
            except Exception:
                return Optional(None)
        return wrappers

# 全局变量定义
def _get_collectors_globals():
    result = {}
    for key, value in Collectors.__dict__.items():
        if key in object.__dict__.values():
            continue
        if key.startswith('_'):
            continue
        if key == 'of':
            continue
        result[key] = value
    return result
globals().update(_get_collectors_globals())


class Optional:
    '''表示 一个值存不存在的容器类'''
    def __init__(self, value):
        self._value = value
        self._isnull = value is None

    def isPresent(self, operate=None):
        if not self._isnull and operate:
            operate(self._value)
        return not self._isnull
    
    def get(self):
        if self._isnull:
            raise ValueError('NullPointerException: value is null')
        return self._value
    
    def orElse(self, obj):
        if self._isnull:
            return obj
        return self._value
    
    def orElseGet(self, func):
        if self._isnull:
            return func()
        return self._value
    
    def orElseThrow(self, func):
        if self._isnull:
            raise func()
        return self._value
    
    def map(self, func):
        value = func(self._value)
        self._value = value
        return self
    
    def filter(self, func):
        if not func(self._value):
            self._isnull = True
            self._value = None
        return self
    
    def flatMap(self, func):
        value = func(self._value)._commit()
        self._value = value
        return self

    @classmethod
    def of(cls, value):
        if value is None:
            raise ValueError('NullPointerException: value is null')
        return cls(value)
    
    @classmethod
    def ofNullable(cls, value):
        return cls(value)

class StreamBuilder:
    def __init__(self, stream_class):
        self.stream_class = stream_class
        self.attrs = []
    
    def add(self, item):
        self.attrs.append(item)
        return self
    
    def build(self):
        return self.stream_class(self.attrs)

class Stream:
    '''序列对象处理流'''
    def __init__(self, flow: Iterable):
        self.flow = flow
        self._init()
        self.closed = False

    def _commit(self) -> Iterable:
        '''计算'''
        if self.closed:
            raise ValueError('IllegalStateException: stream has already been operated upon or closed')
        else:
            self.closed = True
        while len(self._operate) > 0:
            operate, *args = self._pop()
            method = getattr(self, '_' + operate, object())
            self.flow = method(self.flow, *args)
        return self.flow
    
    def __call__(self, *args, **kwargs):
        if args or kwargs:
            stream(self.flow, *args, **kwargs)
        return self
    
    def __str__(self):
        return '<%s operate=%r flow=%r>'%(self.__class__.__name__, self._operate, self.flow)
    
    __repr__ = __str__

    def __iter__(self):
        if not self.closed:
            self._commit()
        yield from self.flow
    
    def parallel(self):
        raise RuntimeError('Stream.py Not support parallel')
    
    def __dumps__(self):
        '''将操作转换为人类可读的文字'''
        if len(self._operate) <= 0:
            return ''
        backup = []
        bytes_code = 'LOAD %r\n'%self.flow
        while len(self._operate) > 0:
            operate, *args = self._pop()
            bytes_code += operate
            bytes_code += '\t'
            bytes_code += repr((*args,))
            bytes_code += '\n'
            backup.append((operate, *args))
        for operate in backup:
            self._push(operate)
        return bytes_code

    # ----------------------------------------------------------------------------------------
    # 队列使用, 
    # 覆盖下面方法可实现其他队列

    def _init(self):
        '''初始化'''
        self._operate = deque()
    
    def _push(self, item):
        '''新增元素'''
        return self._operate.append(item)

    def _pop(self):
        '''弹出元素'''
        return self._operate.popleft()
    
    # ----------------------------------------------------------------------------------------
    # 可以执行的操作
    # 创建

    @classmethod
    def of(cls, *items):
        return cls(items)
    
    @classmethod
    def generate(cls, generator):
        return cls(generator())
    
    @classmethod
    def iterate(cls, init, func):
        def generator():
            x = init
            yield x
            while True:
                x = func(x)
                yield x
        return cls(generator())
    
    @classmethod
    def empty(cls):
        return cls(())

    @classmethod
    def builder(cls):
        return StreamBuilder(cls)

    # 输入

    @classmethod
    def concat(cls, s1, s2, *more):
        def stream_proxy():
            yield from s1.flow
            yield from s2.flow
        stream = cls(stream_proxy())
        if len(more) >= 1:
            more = more[1:]
            return cls.concat(stream, more[0], *more)
        return stream

    def _map(self, flow, func):
       for item in flow:
           yield func(item)
    
    def map(self, func):
        self._push(('map', func))
        return self


    def _reduce(self, flow, func):
        flags = False
        x = None
        for item in flow:
            if not flags:
                x = item
                flags = True
            else:
                x = func(x, item)
                yield x

    def reduce(self, func):
        self._push(('reduce', func))
        return self
    

    def _filter(self, flow, func):
        for item in flow:
            if func(item):
                yield item

    
    def filter(self, func):
        if func is None:
            def func(s):
                return s
        self._push(('filter', func))
        return self


    def _sorted(self, flow, com):
        result = []
        for item in flow:
            if len(result) == 0:
                result.append(item)
                continue
            for index, value in enumerate(result, 0):
                if com:
                    compare = not com(value, item)
                else:
                    compare = value < item                        
                if compare:
                    continue
                else:
                    result.insert(index, item)
                    break
            else:
                result.append(item)
        return result
    
    def sorted(self, com=None):
        self._push(('sorted', com))
        return self
    
    
    def _distinct(self, flow):
        for f in set(flow):
            yield f

    def distinct(self):
        self._push(('distinct',))
        return self
    
    def _skip(self, flow, length):
        long = 0
        length = int(length)
        for f in flow:
            long += 1
            if long > length:
                yield f

    def skip(self, length):
        self._push(('skip', length))
        return self

    def _limit(self, flow, length):
        length = int(length)
        for f in flow:
            length -= 1
            if length >= 0:
                yield f
            else:
                return

    def limit(self, length):
        self._push(('limit', length))
        return self
    
    def _peek(self, flow, func):
        for f in flow:
            func(f)
            yield f

    def peek(self, func):
        self._push(('peek', func))
        return self
    
    def _flatMap(self, flow, func):
        for item in flow:
            yield from func(item).flow

    def flatMap(self, func):
        self._push(('flatMap', func))
        return self
    
    # 输出

    def mapToInt(self, func):
        self._push(('map', func))
        return IntStream(self._commit())

    def mapToDouble(self, func):
        self._push(('map', func))
        return DoubleStream(self._commit())
    
    # Python3 int 和 long 没区别
    mapToLong = mapToInt
    
    def forEach(self, func):
        for item in self._commit():
            func(item)
    
    def collect(self, strategy: Collectors):
        items = self._commit()
        return strategy(items)
    
    def count(self):
        return _count(self._commit())
    
    def max(self, com=lambda s: s):
        items = list(self._commit())
        if len(items) == 0:
            return Optional(None)
        try:
            return Optional(reduce(lambda x, y: max(com(x), com(y)), items))
        except Exception:
            return Optional(None)

    def min(self, com=lambda s: s):
        items = list(self._commit())
        if len(items) == 0:
            return Optional(None)
        try:
            return Optional(reduce(lambda x, y: min(com(x), com(y)), items))
        except Exception:
            return Optional(None)
    
    def toArray(self, types): 
        return self.collect(Collectors.toArray(types))
    
    def anyMatch(self, func):
        items = []
        for item in self._commit():
            items.append(func(item))
        return any(items)

    def allMatch(self, func):
        items = []
        for item in self._commit():
            items.append(func(item))
        return all(items)

    def noneMatch(self, func):
        items = []
        for item in self._commit():
            items.append(func(item))
        return not all(items)
    
    def findFirst(self):
        items = self._commit()
        try:
            value = next(iter(items))
        except StopIteration:
            value = None        
        return Optional(value)
    
    # 由于不支持并行, 
    # 所以 findAny 是取第一个元素
    findAny = findFirst
    # forEachOrdered 本来就是有序的
    forEachOrdered = forEach

def FlowWrapper(flow, name, types=None):
    for item in flow:
        if types != None and type(item) != types:
            try:
                item = types(item)
            except (TypeError, ValueError):
                raise TypeError('%s item must be %s data (not %r)'%(name, types.__name__, type(item))) \
                    from None
        yield item


class IntStream(Stream):
    TYPE = int

    def __init__(self, flow):
        self.flow = FlowWrapper(flow, self.__class__.__name__, self.TYPE)
        self._init()
        self.closed = False

    def _commit(self):
        flow = super()._commit()
        return FlowWrapper(flow, self.__class__.__name__, self.TYPE)
    
    def sum(self):
        return sum(self._commit())
    
    def average(self):
        items = self._commit()
        value = _averaging(items)
        if value == 0:
            value = None
        return Optional(value)
    
    def mapToObj(self, func):
        self._push(('map', func))
        return Stream(self._commit())
    
    def boxed(self):
        return Stream(self._commit())
    
    def summaryStatistics(self):
        return IntSummaryStatistics(self._commit())
    
    @classmethod
    def range(self, start, stop):
        def generator(start=start, stop=stop):
            while start < stop:
                yield start
                start += 1
        return self.generate(generator)

    @classmethod
    def rangeClosed(self, start, stop):
        def generator(start=start, stop=stop):
            while start <= stop:
                yield start
                start += 1
        return self.generate(generator)


class DoubleStream(IntStream):
    TYPE = float

    def summaryStatistics(self):
        return DoubleSummaryStatistics(self._commit())

    # 删除 IntStream 和 LongStream 独有的部分

    @classmethod
    def range(self, start, stop):
        raise TypeError('DoubleStream not support range')

    @classmethod
    def rangeClosed(self, start, stop):
        raise TypeError('DoubleStream not support rangeClosed')


class IntSummaryStatistics:
    TYPE = int

    def __init__(self, flow: list):
        self._data = [self.TYPE(data) for data in flow]
        if not self._data:
            self._data = [self.TYPE(0)]
        self.count = _count(self._data)
        self.sum = sum(self._data)
        self.min = min(self._data)
        self.average = float(_averaging(self._data))
        self.max = max(self._data)
    
    def __str__(self):
        string = self.__class__.__name__ + '{'
        string += f'{self.count=}, '
        string += f'{self.sum=}, '
        string += f'{self.min=}, '
        string += f'{self.average=}, '
        string += f'{self.max=}'
        string += '}'
        return string
    
    def getCount(self):
        return self.count

    def getSum(self):
        return self.sum

    def getMin(self):
        return self.min

    def getAverage(self):
        return self.average

    def getMax(self):
        return self.max

class DoubleSummaryStatistics(IntSummaryStatistics):
    TYPE = float

# Python中int和long没有区别
LongStream = IntStream
LongSummaryStatistics = IntSummaryStatistics

# --------------------------------------------------------------------------------------------
# 函数接口

def FunctionalInterface(cls):
    abstract_method = None
    has_abstract = False
    method_list = []
    for name, value in cls.__dict__.items():
        if name in object.__dict__.keys():
            continue
        if hasattr(value, 'ABSTRACT'):
            if has_abstract:
                raise AttributeError('FunctionalInterface %r must has one abstract method'%cls.__name__)
            abstract_method = name
            has_abstract = True
        if type(value) == type(lambda:...) and not (name.startswith('__') or name.endswith('__')):
            method_list.append(name)
    if abstract_method is None:
        if len(method_list) == 1:
            abstract_method = method_list[0]
        else:
            raise AttributeError('FunctionalInterface %r must has one abstract method'%cls.__name__)
    annotations = cls.__dict__.get(abstract_method).__annotations__
    setattr(cls, '__function_interface__', {'name': abstract_method, 'annotations': annotations})
    setattr(cls, '__new__', lambda cls, function:check_function_type(function, cls))
    return cls

def abstract(function):
    setattr(function, 'ABSTRACT', 'abstract')
    return function

def check_function_type(function, interface):
    '''检查函数接口类型
    返回包裹后的函数
    参数:
        function (FunctionType) 函数
        interface (Classes) 函数式接口类
    '''
    if hasattr(interface, '__function_interface__'):
        abstract = getattr(interface, '__function_interface__')
    else:
        interface = FunctionalInterface(interface)
        abstract = getattr(interface, '__function_interface__')
        # raise TypeError('%r must be a FunctionalInterface'%interface)
    annotations = list(abstract['annotations'].items())
    method_name = abstract['name']
    def wrappers(*args):
        index = -1
        returnvalue = object
        for name, value in annotations:
            # value 是类型
            # args[index] 是实例
            index += 1
            if name == 'return':
                returnvalue = value
                continue
            if isinstance(args[index], value):
                if not issubclass(type(args[index]), value):
                    raise TypeError('FunctionalInterface argument %s type must be %r (not %r)'%(name, value, type(args[index])))
        result = function(*args)
        if isinstance(result, returnvalue):
            if not issubclass(type(result), returnvalue):
                raise TypeError('FunctionalInterface return value type must be %r (not %r)'%(returnvalue, type(result)))
        return result
    wrappers.__module__ = function.__module__
    wrappers.__name__ = function.__name__
    wrappers.__qualname__ = function.__qualname__
    wrappers.__doc__ = function.__doc__
    wrappers.__annotations__ = function.__annotations__
    setattr(wrappers, method_name, lambda *args:wrappers(*args))
    return wrappers

def interface_function(interface):
    def wrapper(function):
        return check_function_type(function, interface)
    return wrapper

def func_checker(function):
    code = function.__code__
    vars = code.co_varnames[:code.co_argcount]
    if len(vars) == 0:
        return function
    index2type = {}
    for name, types in function.__annotations__.items():
        if name == 'return':
            continue
        if name in vars:
            index2type[vars.index(name)] = types
    def wrappers(*args, **kwargs):
        args = list(args)
        for index, types in index2type.items():
            args[index] = check_function_type(args[index], types)
        return function(*args, **kwargs)
    wrappers.__module__ = function.__module__
    wrappers.__name__ = function.__name__
    wrappers.__qualname__ = function.__qualname__
    wrappers.__doc__ = function.__doc__
    wrappers.__annotations__ = function.__annotations__
    return wrappers

NoneType = type(None)
Any = object

@FunctionalInterface
class Function:
    def apply(T: Any) -> Any:pass

@FunctionalInterface
class Consumer:
    def accept(T: Any) -> NoneType:pass

@FunctionalInterface
class IntBinaryOperator:
    def applyAsInt(left: int, right: int) -> int:pass

@FunctionalInterface
class Predicate:
    def test(t: Any) -> bool:pass

@FunctionalInterface
class Supplier:
    def get() -> Any:pass

# --------------------------------------------------------------------------------------------
# 将Python内置队列转换为流的便携方法

def stream(obj: Iterable) -> Stream:
    '''将对象转换为流'''
    if isinstance(obj, Iterable):
        return Stream(obj)
    raise TypeError('%r object is not iterable, \nso can not to stream.'%obj)


def dumps(stream: Stream) -> str:
    if hasattr(stream, '__dumps__'):
        return stream.__dumps__()
    raise TypeError('%r object is not a stream, \nso can not dumps'%stream)


def _globals_variable():
    # “强行”扩展 Python 内置序列

    import ctypes    # 祭出大杀器——python API

    def python_extend(types, name=None):

        class PyObject(ctypes.Structure):
            pass

        if hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
            Py_ssize_t = ctypes.c_int64
        else:
            Py_ssize_t = ctypes.c_int

        PyObject._fields_ = [
            ('ob_refcnt', Py_ssize_t),
            ('ob_type', ctypes.POINTER(PyObject)),
        ]

        class SlotsPointer(PyObject):
            _fields_ = [('dict', ctypes.POINTER(PyObject))]

        def proxy_builtin(klass):
            name = klass.__name__
            slots = getattr(klass, '__dict__', name)

            pointer = SlotsPointer.from_address(id(slots))
            namespace = {}

            ctypes.pythonapi.PyDict_SetItem(
                ctypes.py_object(namespace),
                ctypes.py_object(name),
                pointer.dict,
            )

            return namespace[name]
        
        def decorator(function):
            nonlocal name
            if name == None:
                name = function.__name__
            proxy_builtin(types)[name] = lambda self, *args: function(self, *args)
            return function
        
        return decorator

    @python_extend(list, 'stream')
    @python_extend(tuple, 'stream')
    @python_extend(set, 'stream')
    @python_extend(bytearray, 'stream')
    @python_extend(deque, 'stream')
    def _method_stream(self, types=None, start=None, stop=None):
        if types == int:
            klass = IntStream
        elif types == float:
            klass = DoubleStream
        else:
            klass = Stream
        flow = list(self)
        if start != None:
            flow = flow[start:]
        if stop !=None:
            flow = flow[:stop]
        return klass(flow)
    
    @python_extend(dict, 'forEach')
    def _dict_foreach(self, func):
        for k, v in self.items():
            func(k, v)

    from io import StringIO, TextIOWrapper

    @python_extend(TextIOWrapper, 'stream')
    @python_extend(StringIO, 'stream')
    def _file_stream(fp):
        flow = iter(fp)
        return Stream(flow)

    export = _method_stream, _file_stream, _dict_foreach
    return export

try:
    _ = _globals_variable()
except Exception:
    pass

import importlib

def execute_imports(imports):
    glo = {}
    imps = imports.split(';')
    for imp_stx in imps:
        module, _, obj_alias = imp_stx.partition(":")
        obj, _, alias = obj_alias.partition(":")
        if not obj:
            glo[alias or module] = importlib.import_module(module)
        else:
            _garb = importlib.import_module(module)
            glo[alias or obj] = getattr(_garb, obj)
    return glo

def cli(argument: list=sys.argv[1:]):
    if not argument:
        print(cli_help)
    elif argument[0] in ('-h', '--help'):
        print(cli_help)
    else:
        arg = argument.copy()
        file = '-'
        impl = []
        command = 'None'
        if '-f' in argument:
            index = argument.index('-f')
            if index >= len(argument) - 1:
                print('stream.py:错误: -f 参数后面必须是文件名')
                return
            else:
                file = argument[index + 1]
                arg.remove('-f')
                arg.remove(file)
        if '-i' in argument:
            index = argument.index('-i')
            if index >= len(argument) - 1:
                print('stream.py:错误: -i 参数后必须是导入的模块')
                return
            else:
                impl = argument[index + 1]
                arg.remove('-i')
                arg.remove(impl)
        if len(arg) == 1:
            command = arg[0]
        elif len(arg) == 0:
            print("stream.py:错误: 没有命令输入")
            return
        else:
            print("stream.py:错误: 有多个命令输入, (%s)"%'\n'.join(arg))
            return
        print(command)
        g = execute_imports(impl)
        if file == '-':
            s = sys.stdin.stream()
        else:
            try:
                s = open(file, 'r').stream()
            except FileNotFoundError:
                print('stream.py:错误: 文件不存在 %r'%file)
                return
        g['_'] = s
        g.update(_get_collectors_globals())
        
        pipeline = eval(command, g, {})

        if hasattr(pipeline, "__iter__") and not isinstance(pipeline, (str, bytes)):
            for r in pipeline:
                sys.stdout.write(str(r) + "\n")
        elif pipeline is None:
            pass
        else:
            sys.stdout.write(str(pipeline) + "\n")

if __name__ == "__main__":
    print('------- 测试 -------')
    class data:    # 假数据
        fruit = ["Apple", "Banana", "Blackberry", "Coconut", "Avocado", "Cherry", "Apricots"]
        word = ["A", "B", "C", "D", "E", "F"]
        name = ["张三", "李四", "王五"]

    
    (['Stream', 'Python 2', 'Python 3']
        .stream()
        .map(lambda item: item.upper())
        .filter(lambda item: item.startswith('P'))
        .reduce(lambda item, items: str(items) + ' + ' + str(item))
        .forEach(lambda item: print(item, end=''))
    )

    print()

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

    print(Stream.concat(["A", "B", "C"].stream(), ["D", "E"].stream()).collect(Collectors.toList()))

    print(Stream([[1, 2, 3], [4, 5, 6]]).flatMap(lambda s: s.stream()).mapToDouble(lambda s: float(s)).collect(Collectors.toList()))

    Stream.iterate(0, lambda x: x+2).limit(5).forEach(print)

    print(IntStream.range(1, 10).sum())
    print(IntStream.rangeClosed(1, 10).sum())

    print(["张三", "李四", "王五"].stream().collect(Collectors.toCollection(tuple)))

    # DoubleStream.range(1, 10)

    print(["张三", "李四", "王五"].stream().findFirst().isPresent(print))

    s = Supplier(lambda: Stream.of(*data.fruit))

    print(s.get().anyMatch(lambda s: isinstance(s, int)))

    print(s.get().noneMatch(lambda s: isinstance(s, int)))

    personNameCollector =  Collectors.of(
            lambda: list(),
            lambda j, p: j.append(p.upper()),
            lambda j1, j2: j1 + j2,
            lambda j: ' | '.join(j))

    names = data.fruit \
        .stream() \
        .collect(personNameCollector)
    
    print(names)

    strings = ["abc", "", "bc", "efg", "abcd","", "jkl"]
    numbers = [9, 4, 49, 25]
    integers = [1, 2, 13, 4, 15, 6, 17, 8, 19]

    print("使用 stream.py: ")
    print("列表: ", strings)
        
    count = strings.stream().filter(lambda s: len(s) == 0).count()
    print("空字符串数量为: ",  count)
        
    count = strings.stream().filter(lambda s: len(s) == 3).count()
    print("字符串长度为 3 的数量为: ",  count)
        
    filtered = strings.stream().filter(lambda s: len(s) != 0).collect(Collectors.toList())
    print("筛选后的列表: ",  filtered)
        
    mergedString = strings.stream().filter(lambda s: len(s) != 0).collect(Collectors.joining(", "))
    print("合并字符串: ",  mergedString)
        
    squaresList = numbers.stream().map(lambda i: i*i).distinct().collect(Collectors.toList())
    print("平方数列表: ",  squaresList)
    print("列表: ", integers)
        
    stats = integers.stream().mapToInt(lambda x:x).summaryStatistics()
        
    print("列表中最大的数 : ",  stats.getMax())
    print("列表中最小的数 : ",  stats.getMin())
    print("所有数之和 : ",  stats.getSum())
    print("平均数 : ",  stats.getAverage())
    print("随机数: ")
        
    # [sum(list(os.urandom(100))) for _ in range(10)].stream().limit(10).sorted().forEach(print)

    count = strings.stream().filter(lambda s: len(s) == 0).count()
    print("空字符串的数量为: ", count)

    print(check_function_type(lambda x, y: x + y, IntBinaryOperator)(1, 2))
    
    print([1, 2, 3].stream(float).sum())

    s = Stream.builder()

    s.add('A')
    s.add('B')
    s.add('C')
    s.add('D')
    s.add('E')
    s.add('F')

    print(s.build().collect(Collectors.toList()))

    print(IntStream.builder().add(1).add(2).add(3).build().sum())

    class _L(list):
        pass

    print(_L([2, 3, 5, 7, 11]).stream().map(lambda x: x+ 1).collect(Collectors.collectingAndThen(Collectors.toList(), tuple)))

    from io import StringIO

    s = StringIO()
    s.writelines(['1\n', '2\n', '3\n', '4\n', '5'])
    s.seek(0)
    print(s.stream().mapToInt(int).sum())

    import re

    class SimpleEntry:
        def __init__(self, key, value):
            self.key = key
            self.value = value

        def getKey(self):
            return self.key

    with open('hhh.txt', 'r') as f:
        wordCount = f \
            .stream() \
            .flatMap(lambda line: list.stream(re.split("\\s", line.lstrip()))) \
            .map(lambda word: re.sub("[^a-zA-Z]", "", word).lower().lstrip()) \
            .filter(lambda word: len(word) > 0) \
            .map(lambda word: SimpleEntry(word, 1)) \
            .collect(Collectors.groupingBy(SimpleEntry.getKey, Collectors.counting()))
        wordCount.forEach(lambda k, v: print("%s ==>> %d"%(k, v)))
    
    @func_checker
    def test_func(a, b, f: Function, c=0):
        return f.apply(a + int(b, base=16))
    
    print(test_func(1, 'a', lambda x: x*2))
