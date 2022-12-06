import tensorflow as tf
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
import tensorflow_text  # This import is necessary for BERT preprocessing to work, even though it's not used here


class BertPreprocessingLayer(tf.keras.layers.Layer):
    """
    Processing layer for BERT, taking an array of strings of input and processing these to be used as input for BERT.
    """
    def __init__(self, max_seq_length: int = 128, **kwargs):
        super(BertPreprocessingLayer, self).__init__(**kwargs)
        preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        self.tokenize = hub.KerasLayer(preprocessor.tokenize)
        self.bert_pack_inputs = hub.KerasLayer(
            preprocessor.bert_pack_inputs,
            arguments=dict(seq_length=max_seq_length)
        )

    def call(self, inputs, **kwargs):
        tokenized_inputs = [self.tokenize(segment) for segment in inputs]
        return self.bert_pack_inputs(tokenized_inputs)


class BertBase(tf.keras.Model):
    """
    Model representing BERT-base.
    """
    def __init__(self, max_seq_length: int = 128, **kwargs):
        super(BertBase, self).__init__(**kwargs)
        self.preprocessing = BertPreprocessingLayer(max_seq_length=max_seq_length)
        self.bert_base = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
            trainable=True,
            name='bert_base'
        )

    def call(self, inputs, **kwargs):
        return self.bert_base(self.preprocessing(inputs))['pooled_output']


class Hidden(tf.keras.layers.Layer):
    """
    Hidden layer consisting of a Dense, Leaky ReLU and Dropout layer
    """
    def __init__(self, units: int = 768, dropout: float = 0.1, **kwargs):
        super(Hidden, self).__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(units)
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, **kwargs):
        return self.dropout(self.leaky_relu(self.dense(inputs)))


class Generator(tf.keras.Model):
    """
    Model representing a simple generator with one hidden layer.
    """
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.hidden = Hidden(768)
        self.out = tf.keras.layers.Dense(768, dtype=tf.float32)

        # Specify input shape
        self.shape_in = 100
        self.build((None, self.shape_in))

    def call(self, inputs, **kwargs):
        return self.out(self.hidden(inputs))


class ComplexGenerator(tf.keras.Model):
    """
    Model representing a more complex generator with three hidden layer.
    """
    def __init__(self, **kwargs):
        super(ComplexGenerator, self).__init__(**kwargs)

        self.hidden1 = Hidden(96)
        self.hidden2 = Hidden(192)
        self.hidden3 = Hidden(384)
        self.out = tf.keras.layers.Dense(768, dtype=tf.float32)

        # Specify input shape
        self.shape_in = 96
        self.build((None, self.shape_in))

    def call(self, inputs, **kwargs):
        return self.out(self.hidden3(self.hidden2(self.hidden1(inputs))))


class Discriminator(tf.keras.Model):
    """
    Model representing the discriminator.
    """
    def __init__(self, num_classes: int, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.hidden = Hidden(768)
        self.out = tf.keras.layers.Dense(num_classes, dtype=tf.float32)
        self.softmax = tf.keras.layers.Softmax()

        # Specify input shape
        self.build((None, 768))

    def call(self, inputs, **kwargs):
        return self.softmax(self.out(self.hidden(inputs)))


class Critic(tf.keras.Model):
    """
    Model representing the critic.
    """
    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)

        self.hidden = Hidden(768)
        self.out = tf.keras.layers.Dense(1)

        # Specify input shape
        self.build((None, 768))

    @tf.function
    def call(self, inputs):
        return self.out(self.hidden(inputs))
