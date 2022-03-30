from cgi import test
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

# tf.config.set_visible_devices([], 'GPU')

def train():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

    y = tf.one_hot(y, 10)
    # y_test = tf.one_hot(y_test, 10)

    train_data = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

    w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1))
    b1 = tf.Variable(tf.zeros([512]))
    w2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1))
    b2 = tf.Variable(tf.zeros([256]))
    w3 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b3 = tf.Variable(tf.zeros([128]))
    w4 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b4 = tf.Variable(tf.zeros([10]))

    lr = 5e-4

    for epoch in range(100):
        for step, (x, y) in enumerate(train_data):
            x = tf.reshape(x, (-1, 28*28))
            with tf.GradientTape() as tape:
                # from [128, 784] => [128, 512] => [128, 256] => ... => [128, 10]
                h1 = x@w1 + b1
                h1 = tf.nn.relu(h1)
                h2 = h1@w2 + b2
                h2 = tf.nn.relu(h2)
                h3 = h2@w3 + b3
                h3 = tf.nn.relu(h3)
                out = h3@w4 + b4

                # calculate loss
                # mse = mean(sum(y-out)^2)
                loss = tf.square(y-out)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3, w4, b4])

            # the following code has error
            # w1 is a tf.Variable, a substract operation returns a tf.Tensor, 
            # But a tf.Variable is needed to calculate gradient
            # So we need a Variable inplace math operation here:
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])
            w2.assign_sub(lr * grads[2])
            b2.assign_sub(lr * grads[3])
            w3.assign_sub(lr * grads[4])
            b3.assign_sub(lr * grads[5])
            w4.assign_sub(lr * grads[6])
            b4.assign_sub(lr * grads[7])

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        # Evaluate
        total_correct, total_number = 0, 0
        for step, (x, y) in enumerate(test_db):
            x = tf.reshape(x, (-1, 28*28))
            h1 = tf.nn.relu(x@w1 + b1)
            h2 = tf.nn.relu(h1@w2 + b2)
            h3 = tf.nn.relu(h2@w3 + b3)
            out = h3@w4 + b4

            prob = tf.nn.softmax(out, axis=1)
            pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            total_correct += int(tf.reduce_sum(correct))
            total_number += x.shape[0]
        acc = total_correct / total_number
        print(f'Accuracy is {acc}')


if __name__ == '__main__':
    import time
    t1 = time.time()
    train()
    print(f'{time.time() - t1} seconds')