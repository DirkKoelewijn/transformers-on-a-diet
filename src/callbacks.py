import tensorflow as tf

from src.loss import generator_loss
from src.models import WGAN

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# noinspection PyUnusedLocal
class GANCallback(tf.keras.callbacks.Callback):
    """
    Callback that trains our regular GAN.
    """
    def __init__(self, x, batch_size: int, max_g_loss: float = 0.5, max_g_train: int = 5):
        super().__init__()
        self.x = x
        self.i = 0
        self.batch_size = batch_size
        self.max_g_loss = max_g_loss
        self.max_g_train = max_g_train
        self.g_loss = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.i = 0
        self.g_loss = 0.0

    # noinspection DuplicatedCode
    def on_train_batch_begin(self, batch, logs=None):
        batch = [
            self.x[0][self.batch_size*self.i:self.batch_size*(self.i+1)],
            self.x[1][self.batch_size * self.i:self.batch_size * (self.i + 1)]
        ]
        self.train_step(batch)

    def on_train_batch_end(self, batch, logs=None):
        logs['g_loss'] = self.g_loss / (batch + 1.0)

    @tf.function
    def train_generator(self, x, z):
        with tf.GradientTape() as tape:
            # Perform forward steps
            real_feat, real_logits, real_prob = self.model.real_forward_step(x, train_gen=True)
            fake_feat, fake_logits, fake_prob = self.model.fake_forward_step(z, train_gen=True)

            # Calculate the loss
            g_loss = generator_loss(fake_prob, fake_feat, real_feat)

        # Update weights generator
        g_grads = tape.gradient(g_loss, self.model.generator.trainable_variables)
        self.model.optimizer.apply_gradients(zip(g_grads, self.model.generator.trainable_variables))

        return g_loss

    def train_step(self, x):
        g_loss = tf.constant(self.max_g_loss + 1, dtype=tf.float32)
        g_train = 0

        # Train generator
        while g_loss >= self.max_g_loss and g_train < self.max_g_train:
            z = tf.random.normal((16, self.model.generator.shape_in))
            g_loss = self.train_generator(x, z).numpy()
            g_train += 1

        self.g_loss += g_loss


# noinspection PyUnusedLocal
class WGANCallback(tf.keras.callbacks.Callback):
    """
    Class that updates our WGAN.
    """
    def __init__(self, wgan: WGAN, x, batch_size: int, epochs_per_batch: int = 1):
        super().__init__()
        self.x = x
        self.i = 0
        self.batch_size = batch_size
        self.epb = epochs_per_batch
        self.wgan = wgan
        self.g_loss = 0.0
        self.c_loss = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.i = 0
        self.g_loss = 0.0
        self.c_loss = 0.0

    # noinspection DuplicatedCode
    def on_train_batch_begin(self, batch, logs=None):
        batch = [
            self.x[0][self.batch_size * self.i:self.batch_size * (self.i + 1)],
            self.x[1][self.batch_size * self.i:self.batch_size * (self.i + 1)]
        ]
        self.train_step(batch)

    def on_train_batch_end(self, batch, logs=None):
        logs['g_loss'] = self.g_loss / (batch + 1.0)
        logs['c_loss'] = self.c_loss / (batch + 1.0)

    def train_step(self, x):
        with tf.device('/gpu:0'):
            x_cls = self.model.bert_base(x, training=False)
        history = self.wgan.fit(x_cls, epochs=self.epb, verbose=0)

        self.c_loss += sum(history.history['c_loss']) / self.epb
        self.g_loss += sum(history.history['g_loss']) / self.epb


# noinspection PyUnusedLocal
class EvaluateCallback(tf.keras.callbacks.Callback):
    """
    Callback that can be used to evaluate additional datasets.
    """
    def __init__(self, data, prefix='test'):
        super().__init__()
        self.x, self.y = data
        self.prefix = prefix

    def on_epoch_end(self, epoch: int, logs: dict):
        scores = self.model.evaluate(self.x, self.y, verbose=0, return_dict=True)

        for key, score in scores.items():
            logs[f'{self.prefix}_{key}'] = score
