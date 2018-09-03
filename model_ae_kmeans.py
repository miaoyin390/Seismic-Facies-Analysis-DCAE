import numpy as np
import tensorflow as tf
from struct import *
import matplotlib.pyplot as plt

pre_data = np.load('../../data/model_data/pre_data.npy')
row_num, col_num, sample_num = np.shape(pre_data)
data_num = row_num*col_num
data = np.zeros([data_num, sample_num])
for i in range(row_num):
        data[(i*col_num):(i*col_num+col_num)] = pre_data[i,:,:]

max_val = np.max(data)
min_val = np.min(data)
data = (data-min_val)/(max_val-min_val)

learning_rate = 0.01

k = 5
num_input = sample_num
num_encode_h1 = 75
num_encode_h2 = 60
num_decode_h1 = 75
num_decode_h2 = sample_num

x = tf.placeholder('float', [None, num_input])

w_encode_h1 = tf.Variable(tf.random_normal([num_input, num_encode_h1]))
w_encode_h2 = tf.Variable(tf.random_normal([num_encode_h1, num_encode_h2]))
w_decode_h1 = tf.Variable(tf.random_normal([num_encode_h2, num_decode_h1]))
w_decode_h2 = tf.Variable(tf.random_normal([num_decode_h1, num_decode_h2]))

b_encode_h1 = tf.Variable(tf.random_normal([num_encode_h1]))
b_encode_h2 = tf.Variable(tf.random_normal([num_encode_h2]))
b_decode_h1 = tf.Variable(tf.random_normal([num_decode_h1]))
b_decode_h2 = tf.Variable(tf.random_normal([num_decode_h2]))

encode_h1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w_encode_h1), b_encode_h1))
encode_h2 = tf.nn.sigmoid(tf.add(tf.matmul(encode_h1, w_encode_h2), b_encode_h2))

decode_h1 = tf.nn.sigmoid(tf.add(tf.matmul(encode_h2, w_decode_h1), b_decode_h1))
decode_h2 = tf.nn.sigmoid(tf.add(tf.matmul(decode_h1, w_decode_h2), b_decode_h2))

y = decode_h2
y_ = x

cost = tf.reduce_mean(tf.pow(y_ - y, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(int(data_num/50)):
    batch = np.reshape(data[step*50:((step+1)*50),:],[50,sample_num])
    _, cost_ = sess.run([optimizer, cost], feed_dict={x:batch})
    if step % 100 == 0:
        print('step:%d'%step, 'cost:%f'%cost_)


encode = sess.run(encode_h2, feed_dict={x:data})
features = encode
centroides = tf.Variable(tf.slice(tf.random_shuffle(features),[0,0],[k,-1]))
expanded_features = tf.expand_dims(features, 0)
expanded_centroides = tf.expand_dims(centroides, 1)
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_features, expanded_centroides)), 2), 0)
means = tf.concat(0, [tf.reduce_mean(tf.gather(features, tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])), 1) for c in range(k)])

update_centroides = tf.assign(centroides, means)

y = tf.placeholder('float')

init_op = tf.initialize_all_variables()
sess.run(init_op)

for step in range(150):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
    if step % 10 == 0:
        print('step %d, new centroides is'%step, centroid_values)

result = np.reshape(assignment_values,[row_num, col_num])
plt.imshow(result)
plt.show()
