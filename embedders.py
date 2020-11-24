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
