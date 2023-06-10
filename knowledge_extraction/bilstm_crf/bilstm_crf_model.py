# coding=utf-8
import keras
from crf_layer import CRF

class BiLstmCrfModel(object):
    def __init__(
            self, 
            max_len, #转入模型句子的最大长度
            vocab_size, #使用词向量字典的大小
            embedding_dim, #词向量的一个维度
            lstm_units, #lstm隐藏单元的数量
            class_nums, #标签的数量
            embedding_matrix=None #池向量矩阵，如果有预训练的池向量的话，就可以把这个参数传进去，没有也没有关系，这里模型会随机生成一个池向量。
        ):
        super(BiLstmCrfModel, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.class_nums = class_nums
        self.embedding_matrix = embedding_matrix
        if self.embedding_matrix is not None:
            self.vocab_size,self.embedding_dim = self.embedding_matrix.shape
     #建立模型
    def build(self):
        inputs = keras.layers.Input(
                shape=(self.max_len,),    #这个max是一个长度，小于这个长度会做一个padding(填充)
                dtype='int32'
            )
        x = keras.layers.Masking(   #尽量减少padding对模型训练的影响
                mask_value=0
            )(inputs)
        x = keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                trainable=False,
                weights=self.embedding_matrix,
                mask_zero=True
            )(x)
        x = keras.layers.Bidirectional(
                keras.layers.LSTM(
                    self.lstm_units, 
                    return_sequences=True   #需要把每个字的表达输出来，而不是一个字的表达
                )
            )(x)
        x = keras.layers.TimeDistributed(
                keras.layers.Dropout(    #对每一个进行dropout
                    0.2
                )
            )(x)
        crf = CRF(self.class_nums)
        outputs = crf(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam', 
            loss=crf.loss_function, 
            metrics=[crf.accuracy]   #这个卷曲率算的不是一个实体的卷曲率，是一个字符的卷曲率
            )
        print(model.summary())

        return model
        