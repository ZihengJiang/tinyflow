import numpy as np
from numpy.testing import assert_allclose
import tinyflow as tf
import nnvm.graph as graph

if __name__ == '__main__':
  x = tf.placeholder('x')
  y = tf.placeholder('y')
  z = tf.placeholder('z')
  w = tf.placeholder('w')
  m = tf.placeholder('m')
  sym = x + y + z + w + m

  # s1 = sym.debug_str()
  # print(s1)
  # g = graph.create(sym)
  # fg = g.apply('Fusion')
  # print("Fusion graph")
  # s2 = fg.symbol.debug_str()
  # print(s2)
  # kg = fg.apply('CodeGen')

  ax = np.ones((2, 3))
  ay = np.ones((2, 3))
  az = np.ones((2, 3))
  aw = np.ones((2, 3))
  am = np.ones((2, 3))
  sess = tf.Session("gpu")
  ag = sess.run(sym, feed_dict={x:ax, y:ay, z:az,
                              w:aw, m:aw})
  print(ag)
