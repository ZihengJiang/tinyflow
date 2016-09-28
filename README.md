# TinyFlow: Build Your Own DL System in 1.6K Lines

TinyFlow is an example code for [NNVM](https://github.com/dmlc/nnvm/).
It demonstrates how can we build a clean, minimum and powerful computational
graph based deep learning system with same API as TensorFlow.
The operator code are implemented with [Torch7](https://github.com/torch/torch7) to reduce the effort to write operators.

TinyFlow is a real deep learning system that can run on GPU and CPUs.
To support [example/mnist_softmax.py](example/mnist_softmax.py), it takes
- 679 lines code for operators
- 671 lines of code for execution runtime
- 71 lines of code for API glue
- 190 lines of code for front-end

Note that more code in operators can easily be added to make it as feature complete
as most existing deep learning systems.

## The Design
- The graph construction API is automatically reused from NNVM
- We choose Torch7 as the default operator execution backend.
  - So TinyFlow can also be called "TorchFlow" since it is literally TensorFlow on top of Torch:)
  - This allows us to quickly implement the operators and focus code on the system part.
- We intentionally choose to avoid using [MXNet](https://github.com/dmlc/mxnet) as front or backend,
  since MXNet already uses NNVM as intermediate layer, and it would be more fun to try something different.

Although it is minimum. TinyFlow still comes with many advanced design concepts in Deep Learning system.
- Automatic differentiation.
- Shape/type inference.
- Static memory allocation for graph for memory efficient training/inference.

The operator implementation is easy Thanks to Torch7. More fun demonstrations will be added to the project.

## Dependencies
Most of TinyFlow's code is self-contained.
- TinyFlow depend on Torch7 for operator supports with minimum code.
  - We use a lightweight lua bridge code from dmlc-core/dmlc/lua.h
- NNVM is used for graph representation and optimizations

## What is it for
As explained in the goal of [NNVM](https://github.com/dmlc/nnvm/).
It is important to make deep learning system modular and reusable and enable us to build new systems that suits our need.
Here are the reasons why we created TinyFlow

- TinyFlow can be a perfect material to teach new student the concepts of deep learning systems.
  - e.g. design homeworks on implementing symbolic differentiation, memory allocation, operator fusion.
- For learning system researchers, TinyFlow allows easy addition with new system features with
  the modular design being portable to other system that reuses NNVM.
- It demonstrates how intermediate representation like NNVM to be able to
  target multiple front-ends(TF, MXNet) and backends(Torch7, MXNet) with common set of optimizations.

We believe the Unix Philosophy can building learning system more fun and everyone can be able to build
and understand learning system better.

## Build
- Install Torch7
- Set up environement variable ```TORCH_HOME``` to root of torch
- Type ```make```
- Setup python path to include tinyflow and nnvm
```bash
export PYTHONPATH=${PYTHONPATH}:/path/to/tinyflow/python:/path/to/tinyflow/nnvm/python
```
- Try example program ```python example/mnist_softmax.py```
