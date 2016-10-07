import tinyflow as tf
import nnvm.graph as graph
import numpy as np

def mytanh(x):
    return (tf.exp(x) - tf.exp(-x)) / (tf.exp(x) + tf.exp(-x))

x = tf.placeholder(tf.float32)
tanh1 = mytanh(x)

sym = tanh1
s1 = sym.debug_str()
print(s1)
g = graph.create(sym)
fg = g.apply('Fusion')
print("Fusion graph")
s2 = fg.symbol.debug_str()
print(s2)
kg = fg.apply('CodeGen')

ax = np.ones((2, 3))
sess = tf.Session("gpu")
ay = sess.run(tanh1, feed_dict={x:ax})
print(ay)
