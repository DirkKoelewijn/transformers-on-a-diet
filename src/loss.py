import tensorflow as tf


@tf.function
def discriminator_supervised_loss(real_logits, y, epsilon: float = 1e-15):
    label_mask = tf.math.count_nonzero(y, axis=1) == True

    log_probs = tf.nn.log_softmax(real_logits[:, :-1])
    per_example_loss = tf.boolean_mask(-tf.reduce_sum(y * log_probs, axis=-1), label_mask)
    labeled_example_count = tf.cast(tf.size(per_example_loss), tf.float32)

    return tf.divide(tf.reduce_sum(per_example_loss) + epsilon, tf.maximum(labeled_example_count, 1))


@tf.function
def discriminator_unsupervised_loss(fake_prob, real_prob, epsilon: float = 1e-15):
    part1 = -1 * tf.reduce_mean(tf.math.log(1 - real_prob[:, -1] + epsilon))
    part2 = -1 * tf.reduce_mean(tf.math.log(fake_prob[:, -1] + epsilon))
    return part1 + part2


@tf.function
def generator_loss(fake_prob, fake_feat, real_feat, epsilon: float = 1e-15):
    part1 = -1 * tf.reduce_mean(tf.math.log(1 - fake_prob[:, -1] + epsilon))
    part2 = tf.reduce_mean(tf.square(tf.reduce_mean(real_feat, axis=0) - tf.reduce_mean(fake_feat, axis=0)))
    return part1 + part2
