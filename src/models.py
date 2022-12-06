import os
from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf

from src.components import BertBase, Discriminator, Generator, Critic
from src.loss import discriminator_supervised_loss, discriminator_unsupervised_loss


class SavableModel(tf.keras.Model, ABC):
    """
    Abstract class that allows for saving a model that consists of different sub-models.
    """
    @abstractmethod
    def _to_save(self) -> Dict[str, tf.keras.Model]:
        raise NotImplementedError()

    def save_weights(self, dir_name: str, **kwargs):
        for name, sub_model in self._to_save().items():
            os.makedirs(os.path.join(dir_name, name), exist_ok=True)
            sub_model.save_weights(os.path.join(dir_name, name, 'weights'), **kwargs)

    def load_weights(self, dir_name: str, **kwargs):
        for name, sub_model in self._to_save().items():
            sub_model.load_weights(os.path.join(dir_name, name, 'weights'))


class BaselineModel(SavableModel):
    """
    Model representing the baseline
    """
    def __init__(self, num_classes, **kwargs):
        super(BaselineModel, self).__init__(**kwargs)
        self.bert_base = BertBase()
        self.discriminator = Discriminator(num_classes)

    def call(self, inputs, **kwargs):
        return self.discriminator(self.bert_base(inputs))

    def _to_save(self):
        return {
            'bert_base': self.bert_base,
            'discriminator': self.discriminator
        }


class BaseGAN(SavableModel):
    """
    GAN model that only trains the discriminator. The generator should be updated manually, e.g. by using a callback.
    """
    def __init__(self, num_classes: int, generator=None, **kwargs):
        # Initialize parent, enables usage of tf.keras.Model kwargs
        super(BaseGAN, self).__init__(**kwargs)

        self.num_classes: int = num_classes

        # Define model parts
        self.bert_base = BertBase()
        self.discriminator = Discriminator(num_classes + 1)
        self.generator = Generator() if generator is None else generator

        # Set losses
        self.d_loss = 0.0
        self.d_loss_supervised = 0.0
        self.d_loss_unsupervised = 0.0
        self.batch = 0

    @tf.function
    def call(self, inputs):
        x = self.bert_base(inputs)
        features = self.discriminator.hidden(x)
        out = self.discriminator.out(features)
        return self.discriminator.softmax(out[:, :-1])

    @tf.function
    def real_forward_step(self, x, train_gen=False):
        real_x = self.bert_base(x, training=not train_gen)
        d_real_feat = self.discriminator.hidden(real_x, training=not train_gen)
        d_real_logits = self.discriminator.out(d_real_feat, training=not train_gen)
        d_real_prob = self.discriminator.softmax(d_real_logits)
        return d_real_feat, d_real_logits, d_real_prob

    @tf.function
    def fake_forward_step(self, z, train_gen=False):
        # Forward pass generator
        fake_x = self.generator(z, training=train_gen)

        # Forward pass discriminator
        d_fake_feat = self.discriminator.hidden(fake_x, training=not train_gen)
        d_fake_logits = self.discriminator.out(d_fake_feat, training=not train_gen)
        d_fake_prob = self.discriminator.softmax(d_fake_logits)

        return d_fake_feat, d_fake_logits, d_fake_prob

    @tf.function
    def train_discriminator(self, x, y, z):
        label_mask = tf.math.count_nonzero(y, axis=1, dtype=tf.bool)

        with tf.GradientTape() as tape:
            # Perform forward steps
            real_feat, real_logits, real_prob = self.real_forward_step(x, train_gen=False)
            fake_feat, fake_logits, fake_prob = self.fake_forward_step(z, train_gen=False)

            # Calculate losses
            d_loss_supervised = discriminator_supervised_loss(real_logits, y)
            d_loss_unsupervised = discriminator_unsupervised_loss(fake_prob, real_prob)
            d_loss = d_loss_supervised + d_loss_unsupervised

        d_vars = self.bert_base.trainable_variables + self.discriminator.hidden.dense.trainable_variables + self.discriminator.out.trainable_variables

        # Update weights discriminator
        d_grads = tape.gradient(d_loss, d_vars)
        self.optimizer.apply_gradients(zip(d_grads, d_vars))

        return d_loss, d_loss_supervised, d_loss_unsupervised, tf.boolean_mask(y, label_mask), tf.boolean_mask(
            tf.nn.softmax(real_logits[:, :-1]), label_mask)

    @tf.function
    def train_step(self, data):
        x, y = data
        z = tf.random.normal((16, self.generator.shape_in))

        d_loss, d_loss_supervised, d_loss_unsupervised, y_true, y_pred = self.train_discriminator(x, y, z)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)

        # Update custom losses
        self.batch += 1
        self.d_loss += d_loss
        self.d_loss_supervised += d_loss_supervised
        self.d_loss_unsupervised += d_loss_unsupervised

        # Return a dict mapping metric names to current value
        result = {m.name: m.result() for m in self.metrics}
        result['d_loss'] = self.d_loss / self.batch
        result['d_loss_sup'] = self.d_loss_supervised / self.batch
        result['d_loss_unsup'] = self.d_loss_unsupervised / self.batch

        return result

    def _to_save(self) -> Dict[str, tf.keras.Model]:
        return {
            'bert_base': self.bert_base,
            'discriminator': self.discriminator,
            'generator': self.generator,
        }


class WGAN(tf.keras.Model):
    """
    Model representing our Wasserstein GAN.
    """
    def __init__(self, generator: Generator = None, critic_steps: int = 5,
                 gp_weight: float = 10.0, **kwargs):
        # Initialize parent, enables usage of tf.keras.Model kwargs
        super(WGAN, self).__init__(**kwargs)

        # Define components
        self.generator = Generator() if generator is None else generator
        self.critic = Critic()

        # Define optimizers
        self.g_optimizer = None
        self.c_optimizer = None

        # Define configuration
        self.latent_dim = self.generator.shape_in
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        super(WGAN, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer

    @tf.function
    def gradient_penalty(self, batch_size, real_cls, fake_cls):
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_cls - real_cls
        interpolated = real_cls + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, real_cls):
        if isinstance(real_cls, tuple):
            real_cls = real_cls[0]

        # Get the batch size
        batch_size = tf.shape(real_cls)[0]

        for i in range(self.critic_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_cls = self.generator(random_latent_vectors, training=False)
                fake_logits = self.critic(fake_cls, training=True)
                real_logits = self.critic(real_cls, training=True)

                # Calculate the critic loss using the fake and real image logits
                real_loss = tf.reduce_mean(real_cls)
                fake_loss = tf.reduce_mean(fake_cls)
                c_cost = fake_loss - real_loss

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_cls, fake_cls)
                # Add the gradient penalty to the original critic loss
                c_loss = c_cost + gp * self.gp_weight

            # Get the gradients w.r.t the critic loss
            d_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            # Update the weights of the critic using the critic optimizer
            self.c_optimizer.apply_gradients(zip(d_gradient, self.critic.trainable_variables))

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_cls = self.generator(random_latent_vectors, training=True)
            gen_cls_logits = self.critic(generated_cls, training=False)
            g_loss = -tf.reduce_mean(gen_cls_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"c_loss": c_loss, "g_loss": g_loss}
