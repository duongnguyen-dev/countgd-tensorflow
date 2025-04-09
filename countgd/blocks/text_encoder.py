import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

class TextEncoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 bert_preprocessor,
                 bert_model,
                 sequence_length
                 ):
        super().__init__()
        self.sequence_length = sequence_length
        self.preprocessor = hub.load(bert_preprocessor)
        self.tokenize = hub.KerasLayer(self.preprocessor.tokenize)
        self.bert_model = hub.KerasLayer(bert_model, trainable=False)

    def call(self, text_input):
        tokenized_input = self.tokenize(text_input)
        packer = hub.KerasLayer(
            self.preprocessor.bert_pack_inputs,
            arguments=dict(seq_length=self.sequence_length)
        )
        encoder_inputs = packer([tokenized_input])
        encoder_outputs = self.bert_model(encoder_inputs)
        encoder_outputs = tf.keras.layers.Dense(256)(encoder_outputs["pooled_output"])

        return encoder_outputs