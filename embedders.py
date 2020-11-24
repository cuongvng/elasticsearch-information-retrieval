import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text as text

class EmbedderBase(object):
    def format_text(self, plain_text):
        """
        Modify this function when formatting input of other embedders.
        """
        pass

    def embed_single_text(self, single_text):
        pass

    def embed_list_text(self, list_text):
        pass

class SwivelEmbedder(EmbedderBase):
    def __init__(self, model_link="https://tfhub.dev/tensorflow/cord-19/swivel-128d/3"):
        self.embedder = hub.KerasLayer(model_link)
        print(f"Loaded pre-trained model {model_link} successfully!")

    def embed_single_text(self, plain_text):
        formated_text = tf.constant([plain_text])
        return np.array(self.embedder(formated_text))

    def embed_list_text(self, list_text):
        return np.array(self.embedder(tf.constant(list_text)))

