from sklearn.model_selection import learning_curve
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import layers, optimizers
from tensorflow import keras

def main():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    y_train = tf.one_hot(y_train, depth=10)
    db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(200)

    model = keras.Sequential(
        [
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(10)
        ]
    )
    optimizer = optimizers.SGD(learning_rate=0.001)

    def _train_epoch(epoch):
        for step, (x, y) in enumerate(db):
            with tf.GradientTape() as tape:
                x = tf.reshape(x, (-1, 28*28))
                out = model(x)
                loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, 'loss', loss.numpy)

    for epoch in range(30):
        _train_epoch(epoch)

if __name__ == '__main__':
    main()