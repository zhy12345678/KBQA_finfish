# coding=utf-8
import keras
import numpy as np
from bilstm_crf_model import BiLstmCrfModel
from crf_layer import CRF
from data_helpers import NerDataProcessor

max_len = 80
vocab_size = 2410
embedding_dim = 200
lstm_units = 128

if __name__ == '__main__':
    ndp = NerDataProcessor(max_len,vocab_size)
    train_X,train_y = ndp.read_data(
            "E:\毕设\命名实体识别\\train.txt",
            is_training_data=True
        )
    train_X,train_y = ndp.encode(train_X,train_y)
    dev_X,dev_y = ndp.read_data(
            "E:\毕设\命名实体识别\\dev.txt",
            is_training_data=False
        )
    dev_X,dev_y = ndp.encode(dev_X,dev_y)
    test_X,test_y = ndp.read_data(
            "E:\毕设\命名实体识别\\test.txt",
            is_training_data=False
        )
    test_X,test_y = ndp.encode(test_X,test_y)

    class_nums = ndp.class_nums
    word2id = ndp.word2id    #这个是几个字典，后面的话会运用到，所以要进行序列化的保存
    tag2id = ndp.tag2id
    id2tag = ndp.id2tag
    import pickle
    pickle.dump(
            (word2id,tag2id,id2tag),
            open("./checkpoint/word_tag_id.pkl","wb")
        )

    bilstm_crf = BiLstmCrfModel(
            max_len,
            vocab_size,
            embedding_dim,
            lstm_units,
            class_nums
        )
    model = bilstm_crf.build()

    reduce_lr = keras.callbacks.ReduceLROnPlateau(   #回调函数
        monitor='val_loss', 
        factor=0.5, 
        patience=4, 
        verbose=1)

    earlystop = keras.callbacks.EarlyStopping(   #杀减策略
        monitor='val_loss', 
        patience=10, 
        verbose=2, 
        mode='min'
        )
    bast_model_filepath = './checkpoint/best_bilstm_crf_model.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(
        bast_model_filepath, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True,
        mode='min'
        )
    model.fit(
        x=train_X, 
        y=train_y, 
        batch_size=32, 
        epochs=80, 
        validation_data=(dev_X, dev_y),   #验算集
        shuffle=True, 
        callbacks=[reduce_lr,earlystop,checkpoint]
        )
    model.load_weights(bast_model_filepath)   #把最佳的权重加载到模型里边，进行预测
    model.save('./checkpoint/bilstm_crf_model.h5')

    pred = model.predict(test_X)  #预测测试集

    from metrics import *
    y_true, y_pred = [],[]

    for t_oh,p_oh in zip(test_y,pred):
        t_oh = np.argmax(t_oh,axis=1)
        t_oh = [id2tag[i].replace('_','-') for i in t_oh if i!=0]
        p_oh = np.argmax(p_oh,axis=1)
        p_oh = [id2tag[i].replace('_','-') for i in p_oh if i!=0]

        y_true.append(t_oh)
        y_pred.append(p_oh)

    f1 = f1_score(y_true,y_pred,suffix=False)
    p = precision_score(y_true,y_pred,suffix=False)
    r = recall_score(y_true,y_pred,suffix=False)
    acc = accuracy_score(y_true,y_pred)
    print("f1_score: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, accuracy_score: {:.4f}".format(f1,p,r,acc))
    print(classification_report(y_true, y_pred, digits=4, suffix=False))