import tensorflow as tf 

from tf_utils import copy_variable_scope

from_name = "A"
to_name = "B"

class Graph(object):
    def build(self, name, val1, val2):  
        with tf.variable_scope(name):
            self.x = tf.Variable(val1, name='x')
            self.y = tf.Variable(val2, name='y')
            self.z = tf.add(self.x, self.y, name='z')

    def eval(self, sess):
        with sess.as_default():
            x = self.x.eval()
            y = self.y.eval()
            z = self.z.eval()
        return x, y, z

g1 = Graph()
g2 = Graph()
g1.build(from_name, 1, 2)
g2.build(to_name, 3, 4)

copy_op = copy_variable_scope(from_name, to_name)

sess = tf.Session()
# initialize all variables
sess.run(tf.global_variables_initializer())

x1, y1, z1 = g1.eval(sess)
x2, y2, z2 = g2.eval(sess)

print("before copy",x1, y1, z1, x2, y2, z2)
assert x1 != x2 and y1 != y2 and z1 != z2

sess.run([copy_op])

x1, y1, z1 = g1.eval(sess)
x2, y2, z2 = g2.eval(sess)

print("after copy", x1, y1, z1, x2, y2, z2)
assert x1 == x2 and y1 == y2 and z1 == z2
