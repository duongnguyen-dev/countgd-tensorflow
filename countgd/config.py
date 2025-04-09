class SwintBConfig:
    INPUT_SHAPE = (224, 224, 3)
    PATCH_SIZE = 4
    EMBED_DIM = 128
    NUM_HEADS = [4, 8, 16, 32]
    WINDOW_SIZE = 7
    DEPTHS = [2, 2, 18, 2]
    MLP_RATIO=4.
    QK_SCALE=None 
    QKV_BIAS=True
    PROJ_DROP=0.
    ATTN_DROP=0. 
    DROP_PATH=0.

class BertConfig:
    BERT_PREPROCESSOR="https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3"
    BERT_MODEL="https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-12-h-768-a-12/2"
    SEQUENCE_LENGTH=1

class CountGDConfig: 
    H=224
    W=224