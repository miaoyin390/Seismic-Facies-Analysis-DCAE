import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

def wiggle(data, lWidth=0.1):
    sampleNum, traceNum = np.shape(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(traceNum):
        traceData = data[:,i]
        maxVal = np.amax(traceData)
        ax.plot(i+traceData/maxVal, [j for j in range(sampleNum)], color='black', linewidth=lWidth)
        for a in range(len(traceData)):
            if(traceData[a] < 0):
                traceData[a] = 0
        ax.fill(i+traceData/maxVal, [j for j in range(sampleNum)], 'k', linewidth=0)
    ax.axis([0,traceNum,sampleNum,0])
    plt.show()

def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)
def corrupt(x):
    """Take an input tensor and add uniform masking.
    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.
    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=2, dtype=tf.int32), tf.float32))
# %%
def autoencoder(input_shape, n_filters=[1, 10, 10, 10],
                filter_sizes=[3, 3, 3, 3],
                strides=[1, 2, 1, 1], padding='SAME'):
    """Build a deep denoising autoencoder w/ tied weights.
    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    Raises
    ------
    ValueError
        Description
    """
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')


    # %%
    # ensure 2-d is converted to square tensor.
    x_tensor = tf.reshape(x, [-1, 24, 6, 1])
    current_input = x_tensor

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[2*layer_i],
                filter_sizes[2*layer_i+1],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        print('layer %d'%layer_i)
        print('input shape:', current_input.get_shape())
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides, padding), b))
        print('output shape:', output.get_shape())
        current_input = output

    # %%
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides, padding), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_tensor))

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost}

if __name__ == '__main__':
    # %%
    ae = autoencoder(input_shape=[None, 144], \
                     n_filters=[1, 10, 10], \
                     filter_sizes=[3, 3, 3, 3], \
                     strides=[1, 2, 1, 1], \
                     padding='VALID')
    data = np.load('model_6positions_24points.npy')
    row_num, col_num, point_num = np.shape(data)
    data_num = row_num * col_num
    data_2d = np.zeros([data_num, point_num])
    for i in range(row_num):
        data_2d[(i*col_num):(i*col_num+col_num)] = data[i,:,:]

    max_val = np.max(data_2d)
    min_val = np.min(data_2d)
    print('max_val:', max_val)
    print('min_val:', min_val)
    data_2d = (data_2d-min_val)/(max_val-min_val)

    # %%
    learning_rate = 0.002
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    batch_size = 50
    for step in range(int(data_num/batch_size)):
        batch = np.reshape(data_2d[(step*batch_size):((step+1)*batch_size)], [batch_size, point_num])
        _, cost_ = sess.run([optimizer, ae['cost']], feed_dict={ae['x']:batch})
        if step % 100 == 0:
            print('step:%d'%step, 'cost:%f'%cost_)

    x_disp = np.reshape(data_2d[200000], [1,point_num]);
    z_disp, y_disp = sess.run([ae['z'], ae['y']], feed_dict={ae['x']:x_disp})
    x_disp = np.reshape(x_disp[0], [6, 24])
    y_disp = np.reshape(y_disp, [6, 24])
    x_disp = x_disp.T;
    y_disp = y_disp.T;
    np.savetxt('Data\\origin.txt', x_disp)
    np.savetxt('Data\\reconstruction.txt', y_disp)
    for fg_i in range(10):
        np.savetxt('Data\\feature'+str(fg_i)+'.txt',z_disp[0,:,:,fg_i])
    fig=plt.subplot(121)
    plt.imshow(x_disp)
    fig.set_xticks([])
    fig.set_yticks([])
    fig=plt.subplot(122)
    plt.imshow(y_disp)
    fig.set_xticks([])
    fig.set_yticks([])
    plt.show()
    #for fg_i in range(10):
    #    fig=plt.subplot(2,5,fg_i+1)
    #    plt.imshow(z_disp[0,:,:,fg_i])
    #    fig.set_xticks([])
    #    fig.set_yticks([])
    #plt.show()
    output = tf.reshape(ae['z'], [-1, 100])
    features = sess.run(output, feed_dict={ae['x']:data_2d})
    print('the final shape:', np.shape(features))
    k = 6
    centroides = tf.Variable(tf.slice(tf.random_shuffle(features),[0,0],[k,-1]))
    expanded_features = tf.expand_dims(features, 0)
    expanded_centroides = tf.expand_dims(centroides, 1)
    assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_features, expanded_centroides)), 2), 0)
    means = tf.concat(axis=0, values=[tf.reduce_mean(tf.gather(features, tf.reshape(tf.where(tf.equal(assignments, c)), [1,-1])), 1) for c in range(k)])
    
    update_centroides = tf.assign(centroides, means)
    
    y = tf.placeholder('float')
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    for step in range(150):
        _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
        if step % 10 == 0:
            print('step %d, new centroides is'%step, centroid_values)
    
    result = np.reshape(assignment_values,[row_num, col_num])
    plt.imshow(result)
    plt.show()
