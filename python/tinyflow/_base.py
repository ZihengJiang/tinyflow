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
from nnvm.name import NameManager
from nnvm._base import c_str, check_call, _LIB
from nnvm import symbol, graph
from nnvm import _symbol_internal

__all__ = ["float32", "placeholder", "Variable", "group",
           "variable_scope", "get_variable",
           "initialize_all_variables", "gradients"]

# data type table
float32 = 0

# global list of all variables
_all_variables = {}

# global list of all variable initializers
_all_variable_inits = []

class variable_scope(object):
  current = None
  def __init__(self, prefix):
    self._old_manager = None
    self._prefix = prefix

  def __enter__(self):
    self._old_manager = variable_scope.current
    variable_scope.current = self
    return self

  def __exit__(self, ptype, value, trace):
    assert self._old_manager
    variable_scope.current = self._old_manager

  def get(self, name):
    if self._old_manager:
        prefix = self._old_manager.get(self._prefix) + '/'
    else:
        prefix = self._prefix
    return prefix + name

variable_scope.current = variable_scope("")


def Variable(init=None, name=None):
    name = NameManager.current.get(name, 'variable')
    v = symbol.Variable(name)
    if init is not None:
        if not isinstance(init, symbol.Symbol):
            raise TypeError("Expect initialization expression to be Symbol")
        _all_variable_inits.append(symbol.assign(v, init))
    return v


def get_variable(name, initializer=None):
  scope_name = variable_scope.current.get(name)
  if scope_name not in _all_variables:
    _all_variables[scope_name] = Variable(initializer, scope_name)
  return _all_variables[scope_name]


def initialize_all_variables():
    global _all_variable_inits
    print("num of all_variable_inits: %d" % len(_all_variable_inits))
    for v in _all_variable_inits:
        print(v.list_input_names())
    init_op = group(*_all_variable_inits)
    _all_variable_inits = []
    return init_op


def placeholder(dtype, shape=None, name=None):
    v = symbol.placeholder(name=name, dtype=dtype)
    return v


def group(*inputs):
    x = _symbol_internal._nop()
    x._add_control_deps(symbol.Group(inputs))
    return x


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
