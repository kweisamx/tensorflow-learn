import tensorflow as tf
#tf.constant is a computing , the result is a tensor which store in a


a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name="b")

result = tf.add(a,b,name="add")

print(result)

print(tf.Session().run(result))
