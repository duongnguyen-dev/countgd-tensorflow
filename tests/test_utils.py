import tensorflow as tf
from countgd.utils import create_swintransformerB

def test_swint():
    input_image = tf.zeros((1, 224, 224, 3))
    model = create_swintransformerB()
    print(model(input_image))

test_swint()