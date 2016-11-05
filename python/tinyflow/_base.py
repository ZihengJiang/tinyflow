from __future__ import absolute_import as _abs
import os
import sys

if sys.version_info[0] == 3:
    import builtins as __builtin__
else:
    import __builtin__

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

if hasattr(__builtin__, "NNVM_BASE_PATH"):
    assert __builtin__.NNVM_BASE_PATH == curr_path
else:
    __builtin__.NNVM_BASE_PATH = curr_path

if hasattr(__builtin__, "NNVM_LIBRARY_NAME"):
    assert __builtin__.NNVM_LIBRARY_NAME == curr_path
else:
    __builtin__.NNVM_LIBRARY_NAME = "libtinyflow"


import ctypes as _ctypes
from nnvm.name import Prefix
from nnvm._base import c_str, check_call, _LIB
from nnvm import symbol, graph
from nnvm import _symbol_internal

__all__ = ["float32", "Variable", "variable_scope",
            "get_variable", "initialize_all_variables",
            "placeholder", "group", "gradients"]

# data type table
float32 = 0

# global list of all variables
_all_variables = {}

# global list of all variable initializers
_all_variable_inits = []


def Variable(init, name=None, dtype=None):
    if not isinstance(init, symbol.Symbol):
        raise TypeError("Expect initialization expression to be Symbol")
    pre_name = Prefix.current.get(name, 'variable')
    v = symbol.Variable(pre_name)
    _all_variables[name] = v
    _all_variable_inits.append(symbol.assign(v, init))
    return v


class variable_scope(Prefix):
    def __init__(self, name):
        self.prefix = Prefix(name+'/')

    def __enter__(self):
        self.prefix.__enter__()

    def __exit__(self, type, value, traceback):
        self.prefix.__exit__(type, value, traceback)


def get_variable(name, shape=None, dtype=None, initializer=None):
    pre_name = Prefix.current.get(name, 'variable')
    if pre_name not in _all_variable.keys():
        return Variable(initializer, name, dtype)
    else:
        return _all_variables[pre_name]


def placeholder(dtype, shape=None, name=None):
    v = symbol.placeholder(name=name, dtype=dtype)
    return v


def group(*inputs):
    x = _symbol_internal._nop()
    x._add_control_deps(symbol.Group(inputs))
    return x


def initialize_all_variables():
    global _all_variable_inits
    init_op = group(*_all_variable_inits)
    _all_variable_inits = []
    return init_op

def gradients(ys, xs, grad_ys=None):
    if isinstance(ys, list):
        ys = symbol.Group(ys)
    g = graph.create(ys)
    g._set_symbol_list_attr('grad_ys', ys)
    g._set_symbol_list_attr('grad_xs', xs)
    ny = len(ys.list_output_names())
    if grad_ys is None:
        grad_ys = [symbol.ones_like(ys[i]) for i in range(ny)]
    g._set_symbol_list_attr('grad_ys_out_grad', grad_ys)
    sym = g.apply('Gradient').symbol
    nx = len(xs) if isinstance(xs, list) else len(xs.list_output_names())
    ret = [sym[i] for i in range(nx)]
    return ret
