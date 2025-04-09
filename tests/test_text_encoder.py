import tensorflow as tf
from countgd.blocks.text_encoder import TextEncoderBlock

def text_encode(inputs):
    text_encoder = TextEncoderBlock(
        bert_preprocessor="https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3",
        bert_model="https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-12-h-768-a-12/2",
        sequence_length=1
    )
    res = text_encoder(inputs)
    return tf.shape(res)

def test_img_encoder():
    inputs = tf.constant(["hello bro"])
    assert text_encode(inputs).numpy().tolist() == [1, 256]