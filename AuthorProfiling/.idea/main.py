
import tensorflow as tf
import numpy as np
from numpy import argmax
import pandas as pd
import nltk
import random
# nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.utils import to_categorical
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer,MinMaxScaler,StandardScaler
from keras.preprocessing.text import Tokenizer,hashing_trick,text_to_word_sequence,one_hot
from nltk import word_tokenize
from keras.layers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Activation,Embedding,Flatten,LSTM,GRU,RNN,SimpleRNN,Reshape
from keras.layers import Input,merge,add
from keras.layers.merge import concatenate
from sklearn.model_selection import GridSearchCV
from keras.models import Model,save_model,load_model
from keras import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error,roc_curve, auc,f1_score
# import tensorflow_gpu as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot
from nltk import ngrams, SnowballStemmer,pos_tag,pos_tag_sents,word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata

from keras_preprocessing import text
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
from numpy import argmax,array
from sklearn import preprocessing
import os
from shutil import copyfile
from keras import callbacks
from numpy import array
from keras import initializers
# nltk.download('averaged_perceptron_tagger')


labels=[]
texts=[]
test_text=[]
texts2=[]


path = 'C:\\Users\\petro\\Desktop\\thesis\\csicorpus\\'
path4 = "C:\\Users\\petro\\Desktop\\thesis\\csicorpus\\essays\\"
path5 = "C:\\Users\\petro\\Desktop\\thesis\\csicorpus\\reviews\\"
path6 = "C:\\Users\\petro\\Desktop\\thesis\\csicorpus\\retrain\\"#allazw se train2,finalTrain
#path7 = "C:\\Users\\petro\\Desktop\\thesis\\csicorpus\\test2\\"#allazw se test2
codes1=[]
codes2=[]
genres = []
sexualpref = []
ages=[]
genres2 = []
sexualpref2 = []
openess=[]
openess2=[]
Conscientiousness=[]
Conscientiousness2=[]
Extraversion=[]
Extraversion2=[]
Agreeableness=[]
Agreeableness2=[]
Neuroticity=[]
Neuroticity2=[]
ages2=[]
cnt=0
count=0;
count2=0;

counter1=0
counter2=0
counter3=0
counter4=0
for fname in os.listdir(path6):
    if fname[-4:]== '.txt':
        f1=open(os.path.join(path6,fname),encoding="utf8")

        c=f1.name
        c2=c.partition(path6)[2]



        # codes1.append(c.partition(path6)[2].partition('_')[0])#mono id author
        # cnt=cnt+1

        with open(path+'List.CSI.DocumentData.1.4.0.BV.2016-02-08.txt','r',encoding="utf8") as f3:
            for line in f3:
                txtname=line.partition('\t')[0]
                txtname = txtname.replace("\ufeff",'')

                # print(c2)
                if c2==txtname:
                    # print("mpla")
                    code2 = line.partition('\t')[1]
                    code2 = code2.replace("\ufeff",'')
                    id=c2.partition('_')[0]
                    # print(id)









                    with open(path+'List.CSI.AuthorData.1.4.0.BV.2016-02-08.txt','r',encoding="utf8") as f2:
                        for line in f2:
                            code3 = line.partition('\t')[0]
                            code3 = code3.replace("\ufeff",'')
                            age = line.split('\t')[1]
                            age = age.replace("\ufeff",'')
                            age2=2019-int(age.partition('-')[0])
                            gen = line.split('\t')[2]
                            gen = gen.replace("\ufeff",'')
                            sexpr = line.split('\t')[3]
                            sexpr = sexpr.replace("\ufeff",'')
                            opn=line.split('\t')[6]

                            opn=opn.replace("\ufeff",'')

                            if opn.split('-')[0]== '':
                                opens=0
                            else:
                                opens=int(opn.split('-')[0])

                            Consc=line.split('\t')[6]

                            Consc=Consc.replace("\ufeff",'')

                            if Consc.split('-')[1]== '':
                                Conscs=0
                            else:
                                Conscs=int(Consc.split('-')[1])

                            extr=line.split('\t')[6]

                            extr=extr.replace("\ufeff",'')

                            if extr.split('-')[2]== '':
                                extr=0
                            else:
                                extr=int(extr.split('-')[2])

                            agr=line.split('\t')[6]

                            agr=agr.replace("\ufeff",'')

                            if agr.split('-')[3]== '':
                                agr=0
                            else:
                                agr=int(agr.split('-')[3])

                            neur=line.split('\t')[6]

                            neur=neur.replace("\ufeff",'')

                            if neur.split('-')[4]== '':
                                neur=0
                            else:
                                neur=int(neur.split('-')[4])



                            if id==code3:

                                texts.append(f1.read())
                                codes1.append(id)#mono id author

                                if gen=='Male':

                                    genres.append(1)


                                elif gen=='Female':

                                    genres.append(0)
                                    counter1=counter1+1



                                if sexpr=='Straight':

                                    sexualpref.append(1)



                                elif sexpr=='Gay':

                                    sexualpref.append(0)



                                elif sexpr=='NA':

                                    sexualpref.append(2)
                                    print("ok1")

                                if age2<=26:

                                    ages.append(1)

                                elif age2>26:

                                    ages.append(0)

                                if opens>=50:
                                    openess.append(1)



                                elif opens<50:
                                    openess.append(0)

                                if Conscs>=47:
                                    Conscientiousness.append(1)


                                elif Conscs<47:
                                    Conscientiousness.append(0)

                                if extr>=53:
                                    Extraversion.append(1)
                                    counter3=counter3+1

                                elif extr<53:
                                    Extraversion.append(0)
                                    counter4=counter4+1
                                if agr>=43:
                                    Agreeableness.append(1)

                                elif agr<43:
                                    Agreeableness.append(0)
                                if neur>=54:
                                    Neuroticity.append(1)

                                elif neur<54:
                                    Neuroticity.append(0)




#
# for fname in os.listdir(path7):
#     if fname[-4:]== '.txt':
#         f1=open(os.path.join(path7,fname),encoding="utf8")
#
#         c=f1.name
#         c2=c.partition(path7)[2]
#
#
#         with open(path+'List.CSI.DocumentData.1.4.0.BV.2016-02-08.txt','r',encoding="utf8") as f3:
#             for line in f3:
#                 txtname=line.partition('\t')[0]
#                 txtname = txtname.replace("\ufeff",'')
#
#
#                 if c2==txtname:
#
#                     code2 = line.partition('\t')[1]
#                     code2 = code2.replace("\ufeff",'')
#                     id=c2.partition('_')[0]
#
#
#
#
#
#
#
#
#
#
#                     with open(path+'List.CSI.AuthorData.1.4.0.BV.2016-02-08.txt','r',encoding="utf8") as f2:
#                         for line in f2:
#                             code3 = line.partition('\t')[0]
#                             code3 = code3.replace("\ufeff",'')
#                             age = line.split('\t')[1]
#                             age = age.replace("\ufeff",'')
#                             age2=2019-int(age.partition('-')[0])
#                             gen = line.split('\t')[2]
#                             gen = gen.replace("\ufeff",'')
#                             sexpr = line.split('\t')[3]
#                             sexpr = sexpr.replace("\ufeff",'')
#                             opn=line.split('\t')[6]
#
#                             opn=opn.replace("\ufeff",'')
#                             if opn.partition('-')[0]== '':
#                                 opens=0
#                             else:
#                                 opens=int(opn.partition('-')[0])
#
#                             Consc=line.split('\t')[6]
#
#                             Consc=Consc.replace("\ufeff",'')
#
#                             if Consc.split('-')[1]== '':
#                                 Conscs=0
#                             else:
#                                 Conscs=int(Consc.split('-')[1])
#
#                             extr=line.split('\t')[6]
#
#                             extr=extr.replace("\ufeff",'')
#
#                             if extr.split('-')[2]== '':
#                                 extr=0
#                             else:
#                                 extr=int(extr.split('-')[2])
#
#                             agr=line.split('\t')[6]
#
#                             agr=agr.replace("\ufeff",'')
#
#                             if agr.split('-')[3]== '':
#                                 agr=0
#                             else:
#                                 agr=int(agr.split('-')[3])
#
#                             neur=line.split('\t')[6]
#
#                             neur=neur.replace("\ufeff",'')
#
#                             if neur.split('-')[4]== '':
#                                 neur=0
#                             else:
#                                 neur=int(neur.split('-')[4])
#
#
#
#
#
#
#
#                             if id==code3:
#
#                                 texts2.append(f1.read())
#                                 codes2.append(id)#mono id author
#
#                                 if gen=='Male':
#
#                                     genres2.append(1)
#
#                                 elif gen=='Female':
#
#                                     genres2.append(0)
#
#
#
#                                 if sexpr=='Straight':
#
#                                     sexualpref2.append(1)
#
#
#                                 elif sexpr=='Gay':
#
#                                     sexualpref2.append(0)
#
#
#                                 elif sexpr=='NA':
#
#                                     sexualpref2.append(sexpr)
#                                     print("ok")
#
#
#                                 if age2<=26:
#
#                                     ages2.append(1)
#
#                                 elif age2>26:
#
#                                     ages2.append(0)
#
#                                 if opens>=50:
#                                     openess2.append(1)
#
#                                 elif opens<50:
#                                     openess2.append(0)
#                                 if Conscs>=47:
#                                     Conscientiousness2.append(1)
#
#                                 elif Conscs<47:
#                                     Conscientiousness2.append(0)
#                                 if extr>=53:
#                                     Extraversion2.append(1)
#
#                                 elif extr<53:
#                                     Extraversion2.append(0)
#                                 if agr>=43:
#                                     Agreeableness2.append(1)
#
#                                 elif agr<43:
#                                     Agreeableness2.append(0)
#                                 if neur>=54:
#                                     Neuroticity2.append(1)
#
#                                 elif neur<54:
#                                     Neuroticity2.append(0)
#













maxlen=len(texts)
training_samples=len(texts)
validation_samples=len(texts2)
max_words=len(texts)
max_words2=len(texts2)

texts3=[]
normal=[]
normal2=[]

bad_chars = ["*",'\n',')',
             '(',"'",'‘','’','“','”',
             '"','\t','/','%','æ'
             ,'[',']','\u2003','€','°','\u2028','„','—',
             '®','§','<','\xa0','ʃ','±','\u2029','=','$','_']
spec_chars=['1','2','3','4','5','6','7','8','9','10','0','1.','2.','3.']



t1=[]
t2=[]



# txt2=pos_tag_sents(t2,lang='dutch')

# remove bad_chars
for i in bad_chars :
    for j in range(len(texts)):
        texts[j]=texts[j].lower()
        texts[j] = texts[j].replace(i, '')

for i in bad_chars :
    for j in range(len(texts2)):
        texts2[j]=texts2[j].lower()
        texts2[j] = texts2[j].replace(i, '')

#replace special chars with 1
for i in spec_chars :
    for j in range(len(texts)):
        texts[j] = texts[j].replace(i, '1')

for i in spec_chars :
    for j in range(len(texts2)):
        texts2[j] = texts2[j].replace(i, '1')



for i in range(len(texts)):
    t1.append(word_tokenize(texts[i],language='dutch'))





# for i in range(len(texts2)):
#     t2.append(word_tokenize(texts2[i],language='dutch'))


countWords=0
for i in range(len(t1)):
    for j in range(len(t1)):
        countWords=countWords+1
countWords2=0
for i in range(len(t2)):
    for j in range(len(t2)):
        countWords2=countWords2+1


tokens=Tokenizer(num_words=2000,filters=None,char_level=True,lower=True)



tokens.fit_on_texts(texts)
sequences=tokens.texts_to_sequences(texts)
#
# onehot1=[]
# onehot2=[]
#
# for i in texts:
#
#     onehot1.append(one_hot(i,n=max_words,lower=False, split=' '))
#     # onehot1.append(hashing_trick(i,n=550,lower=False, split=" "))
#
#
# for i in texts2:
#
#     onehot2.append(one_hot(i,n=max_words2,lower=True, split=' '))
# #     onehot2.append(hashing_trick(i,n=550,lower=False, split=" "))




data=pad_sequences(sequences,maxlen=maxlen)




# d1=to_categorical(data)
# d2=to_categorical(data2)
#
# print(d1)

# print(d1.shape)
# tokens.fit_on_texts(t1)
# tokens2.fit_on_texts(t2)
#


# tokens2.fit_on_texts(texts2)





# sequences=tokens.texts_to_sequences(t1)
# sequences2=tokens2.texts_to_sequences(t2)


# data=pad_sequences(sequences,maxlen=maxlen)
# data2=pad_sequences(sequences2,maxlen=maxlen)



# onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
# integer_encoded1 = data.reshape(data.shape)
# onehot_encoded1 = onehot_encoder.fit_transform(integer_encoded1)


#
# integer_encoded2 = data2.reshape(data2.shape)
#
# onehot_encoded2 = onehot_encoder.fit_transform(integer_encoded2)



# # # integer encode
label_encoder = LabelEncoder()

transformed_label1 = label_encoder.fit_transform(genres)
transformed_label2 = label_encoder.fit_transform(sexualpref)
transformed_label3 = label_encoder.fit_transform(ages)
transformed_label4 = label_encoder.fit_transform(openess)
transformed_label5 = label_encoder.fit_transform(Conscientiousness)
transformed_label6 = label_encoder.fit_transform(Extraversion)
transformed_label7 = label_encoder.fit_transform(Agreeableness)
transformed_label8 = label_encoder.fit_transform(Neuroticity)


# #for test
# transformed_label12 = label_encoder.fit_transform(genres2)
# transformed_label22 = label_encoder.fit_transform(sexualpref2)
# transformed_label32 = label_encoder.fit_transform(ages2)
# transformed_label42 = label_encoder.fit_transform(openess2)
# transformed_label52 = label_encoder.fit_transform(Conscientiousness2)
# transformed_label62 = label_encoder.fit_transform(Extraversion2)
# transformed_label72 = label_encoder.fit_transform(Agreeableness2)
# transformed_label82 = label_encoder.fit_transform(Neuroticity2)



# indices=np.arange(d1.shape[0])
# indices2=np.arange(d2.shape[0])
indices=np.arange(data.shape[0])
# indices2=np.arange(onehot_encoded2.shape[0])

# print(indices)
data=data[indices]
# d1=d1[indices]
# d2=d2[indices2]
# onehot_encoded1=onehot_encoded1[indices]
# onehot_encoded2=onehot_encoded2[indices2]


transformed_label1=transformed_label1[indices]
transformed_label2=transformed_label2[indices]
transformed_label3=transformed_label3[indices]
transformed_label4=transformed_label4[indices]
transformed_label5=transformed_label5[indices]
transformed_label6=transformed_label6[indices]
transformed_label7=transformed_label7[indices]
transformed_label8=transformed_label8[indices]



# data2=data2[indices2]
# transformed_label12=transformed_label12[indices2]
# transformed_label22=transformed_label22[indices2]
# transformed_label32=transformed_label32[indices2]
# transformed_label42=transformed_label42[indices2]
# transformed_label52=transformed_label52[indices2]
# transformed_label62=transformed_label62[indices2]
# transformed_label72=transformed_label72[indices2]
# transformed_label82=transformed_label82[indices2]

x_train=data[:training_samples]
# x_train=onehot_encoded1[:training_samples]


y1=transformed_label1[:training_samples]
y2=transformed_label2[:training_samples]
y3=transformed_label3[:training_samples]
y4=transformed_label4[:training_samples]
y5=transformed_label5[:training_samples]
y6=transformed_label6[:training_samples]
y7=transformed_label7[:training_samples]
y8=transformed_label8[:training_samples]

# x_test=d2[:validation_samples]
# x_test=onehot_encoded2[:validation_samples]


# y_test1=transformed_label12[:validation_samples]
# y_test2=transformed_label22[:validation_samples]
# y_test3=transformed_label32[:validation_samples]
# y_test4=transformed_label42[:validation_samples]
# y_test5=transformed_label52[:validation_samples]
# y_test6=transformed_label62[:validation_samples]
# y_test7=transformed_label72[:validation_samples]
# y_test8=transformed_label82[:validation_samples]






Y1=to_categorical(y1)
Y2=to_categorical(y2)
Y3=to_categorical(y3)
Y4=to_categorical(y4)
Y5=to_categorical(y5)
Y6=to_categorical(y6)
Y7=to_categorical(y7)
Y8=to_categorical(y8)



# Ytest1=to_categorical(y_test1)
# Ytest2=to_categorical(y_test2)
# Ytest3=to_categorical(y_test3)
# Ytest4=to_categorical(y_test4)
# Ytest5=to_categorical(y_test5)
# Ytest6=to_categorical(y_test6)
# Ytest7=to_categorical(y_test7)
# Ytest8=to_categorical(y_test8)





#
#
#
#
def f1(y_true, y_pred):
    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     K.get_session().run(tf.local_variables_initializer())
#     return auc

#
# def recall(y_true, y_pred):
#
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
# def precision(y_true, y_pred):
#
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

length = len(x_train)
mask = np.random.choice(length, length, replace=False)
# Use the same mask to maintain the shuffling sequence between data and labels
x_train = x_train[mask]
Y1 = Y1[mask]
Y2 = Y2[mask]
Y3 = Y3[mask]
Y4 = Y4[mask]
Y5 = Y5[mask]
Y6 = Y6[mask]
Y7 = Y7[mask]
Y8 = Y8[mask]


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
# x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)



visible1 = Input(batch_shape=(None,None,1))

hidden1 = LSTM(256,batch_input_shape=(x_train.shape[0],x_train.shape[1],1),
               # recurrent_dropout=0.5,                                                                
               # go_backwards=True,
               dropout=0.6,bias_initializer='zeros',kernel_initializer='glorot_uniform')(visible1)

output1 = Dense(2, activation='softmax')(hidden1)

output2 = Dense(3, activation='softmax')(hidden1)

output3=Dense(2,activation='softmax')(hidden1)

output4=Dense(2,activation='softmax')(hidden1)

output5=Dense(2,activation='softmax')(hidden1)

output6=Dense(2,activation='softmax')(hidden1)

output7=Dense(2,activation='softmax')(hidden1)

output8=Dense(2,activation='softmax')(hidden1)



model = Model(inputs=visible1, outputs=[output1,output2,output3,output4,output5,output6,output7,output8])


model.summary()



# stoped=[callbacks.EarlyStopping(patience=800)]

model.compile(loss='categorical_crossentropy',
               #learning rate=0.001 default value
               optimizer=Adam(lr=0.001), #'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
              metrics=['accuracy',f1])



history = model.fit(x_train,[Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8],
                    batch_size=64,#auto to noumero logw arthrou
                    epochs=1,
                    validation_split=0.20,
                    verbose=1

                    )
model.save_weights("finalWeights.hdf5",overwrite=True)
model.save("finalmodel.h5",overwrite=True)

# results=model.evaluate(x_test,[Ytest1,Ytest2,Ytest3,Ytest4,Ytest5,Ytest6,Ytest7,Ytest8],batch_size=64,verbose=1)
# print(results)


#apo katw ksana kanw train sto saved model

# mySavedModel=load_model("model4.h5")
# mySavedModel.load_weights("model4we.hdf5")

#print(mySavedModel.get_weights())

# mySavedModel.fit(x_train,[Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8],batch_size=8,epochs=30,verbose=1)

# score = mySavedModel.evaluate(x_test,[Ytest1,Ytest2,Ytest3,Ytest4,Ytest5,Ytest6,Ytest7,Ytest8],batch_size=8, verbose=1)

# print(score)

dirname='C:\\Users\\petro\\Desktop\\thesis\\csicorpus\\test'
codes1=[]
test_text=[]
for fname in os.listdir(dirname):
    if fname[-4:]== '.txt':
        f=open(os.path.join(dirname,fname),encoding="utf8")
        c=f.name
        c2=c.partition(dirname)[2]
        c3=c2.split('_')[0]
        c3=c3.replace(c3[0][0],'')
        codes1.append(c3)
        test_text.append(f.read())



maxword=len(test_text)
#
for i in bad_chars :
    for j in range(len(test_text)):
        test_text[j] = test_text[j].replace(i, '')


for i in spec_chars :
    for j in range(len(test_text)):
        test_text[j] = test_text[j].replace(i, '1')


tt=[]
for i in range(len(test_text)):
    tt.append(word_tokenize(test_text[i],language='dutch'))


countWords=0
for i in range(len(tt)):
    for j in range(len(tt)):
        countWords=countWords+1



# tokens3=Tokenizer(num_words=2000,filters=None,char_level=True,lower=True)
# tokens3.fit_on_texts(tt)
tokens.fit_on_texts(test_text)
sequences3=tokens.texts_to_sequences(test_text)
# sequences3=tokens3.texts_to_sequences(tt)
# data3=pad_sequences(sequences3,maxlen=maxlen)




# onehotTest=[]
# for i in range(len(test_text)):
#     onehotTest.append(one_hot(test_text[i],n=maxword,lower=True, split=' '))
    # onehotTest.append(hashing_trick(test_text[i],n=550,lower=True, split=" "))
data3=pad_sequences(sequences3,maxlen=maxlen)
# d3=to_categorical(data3)
# onehot_encoder2 = OneHotEncoder(sparse=False,categories='auto')
# integer_encoded3 = data3.reshape(data3.shape)
# onehot_encoded3 = onehot_encoder2.fit_transform(integer_encoded3)
indices3=np.arange(data3.shape[0])
# data3=np.array(onehotTest)
# indices3=np.arange(onehot_encoded3.shape[0])
# onehot_encoded3=onehot_encoded3[indices3]



data3=data3[indices3]
# onehot_encoded3=onehot_encoded3[indices3]
x3=data3[:len(test_text)]

# scaler3=StandardScaler()
# scaler3=scaler3.fit_transform(x3)

# x3=onehot_encoded3[:len(test_text)]

# X3=to_categorical(x3)
# x3=scaler3.reshape(scaler3.shape[0],scaler3.shape[1],1)
x3=x3.reshape(x3.shape[0],x3.shape[1],1)
# X3=X3.reshape(X3.shape[0],X3.shape[1],1)


pred=model.predict(x3)
predict1=pred[0]
predict3=pred[2]
predict4=pred[3]
predict5=pred[4]
predict6=pred[5]
predict7=pred[6]
predict8=pred[7]

predict11=pred[0][:,1]
predict33=pred[2][:,1]
predict44=pred[3][:,1]
predict55=pred[4][:,1]
predict66=pred[5][:,1]
predict77=pred[6][:,1]
predict88=pred[7][:,1]
# pred=mySavedModel.predict(x3)


probas1=[]
probas3=[]

probas4=[]

probas5=[]
probas6=[]

probas7=[]

probas8=[]






def results1(pred1):
    listofPred=[]
   

    for i in range(len(pred1)):


        if pred1[i][0]>pred1[i][1]:
            listofPred.append(0)
          
        elif pred1[i][0]<pred1[i][1]:
            listofPred.append(1)
            
    

    return listofPred




def results2(pred1):
    listofPred=[]

    for i in range(len(pred1)):


        if pred1[i][0]>pred1[i][1] and pred1[i][0]>pred1[i][2]:
            listofPred.append(0)
        elif pred1[i][0]<pred1[i][1] and pred1[i][1]>pred1[i][2]:
            listofPred.append(1)
        elif pred1[i][2]>pred1[i][0] and pred1[i][2]>pred1[i][1]:
            listofPred.append(2)


    return listofPred


def results3(pred1):
    listofPred=[]
  
    for i in range(len(pred1)):


        if pred1[i][0]>pred1[i][1]:
            listofPred.append(0)
            
        elif pred1[i][0]<pred1[i][1]:

            listofPred.append(1)
            
    
    return listofPred


def results4(pred1):
    listofPred=[]
   
    for i in range(len(pred1)):


        if pred1[i][0]>pred1[i][1]:
            listofPred.append(0)
            
        elif pred1[i][0]<pred1[i][1]:
            listofPred.append(1)
           
    
    return listofPred


def results5(pred1):
    listofPred=[]
    probabilitites=[]
   

    for i in range(len(pred1)):


        if pred1[i][0]>pred1[i][1]:
            listofPred.append(0)
           
        elif pred1[i][0]<pred1[i][1]:
            listofPred.append(1)
    
    return listofPred




def results6(pred1):
    listofPred=[]
    
  

    for i in range(len(pred1)):


        if pred1[i][0]>pred1[i][1]:
            listofPred.append(0)
            
        elif pred1[i][0]<pred1[i][1]:
            listofPred.append(1)
            
  
    return listofPred





def results7(pred1):
    listofPred=[]

    for i in range(len(pred1)):


        if pred1[i][0]>pred1[i][1]:
            listofPred.append(0)

        elif pred1[i][0]<pred1[i][1]:
            listofPred.append(1)

            

    return listofPred

def results8(pred1):


    for i in range(len(pred1)):


        if pred1[i][0]>pred1[i][1]:
            listofPred.append(0)

            
        elif pred1[i][0]<pred1[i][1]:
            listofPred.append(1)

            

    return listofPred


genres = []
sexualpref = []
ages=[]

sexualpref2 = []
openess=[]
openess2=[]
Conscientiousness=[]
Conscientiousness2=[]
Extraversion=[]
Extraversion2=[]
Agreeableness=[]
Agreeableness2=[]
Neuroticity=[]
Neuroticity2=[]
ages2=[]
codes3=[]
genders=[]
with open(path+'List.CSI.AuthorData.1.4.0.BV.2016-02-08.txt','r',encoding="utf8") as f2:


    for line in f2:
        code3 = line.partition('\t')[0]
        code3 = code3.replace("\ufeff",'')
        codes3.append(code3)
        age = line.split('\t')[1]
        age = age.replace("\ufeff",'')
        age2=2019-int(age.partition('-')[0])
        gen = line.split('\t')[2]
        gen = gen.replace("\ufeff",'')

        sexpr = line.split('\t')[3]
        sexpr = sexpr.replace("\ufeff",'')

        opn=line.split('\t')[6]

        opn=opn.replace("\ufeff",'')

        if opn.split('-')[0]== '':
            opens=0
        else:
            opens=int(opn.split('-')[0])

        Consc=line.split('\t')[6]

        Consc=Consc.replace("\ufeff",'')

        if Consc.split('-')[1]== '':
            Conscs=0
        else:
            Conscs=int(Consc.split('-')[1])

        extr=line.split('\t')[6]

        extr=extr.replace("\ufeff",'')

        if extr.split('-')[2]== '':
            extr=0
        else:
            extr=int(extr.split('-')[2])

        agr=line.split('\t')[6]

        agr=agr.replace("\ufeff",'')

        if agr.split('-')[3]== '':
            agr=0
        else:
            agr=int(agr.split('-')[3])

        neur=line.split('\t')[6]

        neur=neur.replace("\ufeff",'')

        if neur.split('-')[4]== '':
            neur=0
        else:
            neur=int(neur.split('-')[4])


        if gen=='Male':
            genders.append(1)
        elif gen=='Female':
            genders.append(0)

        if sexpr=='Straight':

            sexualpref2.append(1)


        elif sexpr=='Gay':

            sexualpref2.append(0)


        elif sexpr=='NA':

            sexualpref2.append(2)


        if age2<=27:

            ages2.append(1)


        elif age2>27:

            ages2.append(0)

        if opens>=50:
            openess2.append(1)


        elif opens<50:
            openess2.append(0)

        if Conscs>=47:
            Conscientiousness2.append(1)

        elif Conscs<47:
            Conscientiousness2.append(0)
        if extr>=53:
            Extraversion2.append(1)

        elif extr<53:
            Extraversion2.append(0)
        if agr>=43:
            Agreeableness2.append(1)

        elif agr<43:
            Agreeableness2.append(0)
        if neur>=54:
            Neuroticity2.append(1)

        elif neur<54:
            Neuroticity2.append(0)

c = list(dict.fromkeys(codes3))






for j in range(len(codes1)):
    for i in range(len(c)):

        if c[i]==codes1[j]:

            genres.append(genders[i])
            sexualpref.append(sexualpref2[i])
            ages.append(ages2[i])
            openess.append(openess2[i])
            Conscientiousness.append(Conscientiousness2[i])
            Extraversion.append(Extraversion2[i])
            Agreeableness.append(Agreeableness2[i])
            Neuroticity.append(Neuroticity2[i])




counter=0
predList1=[]
predList2=[]
predList3=[]
predList4=[]
predList5=[]
predList6=[]
predList7=[]
predList8=[]
predList1=results1(pred[0])
predList2=results2(pred[1])
predList3=results3(pred[2])
predList4=results4(pred[3])
predList5=results5(pred[4])
predList6=results6(pred[5])
predList7=results7(pred[6])
predList8=results8(pred[7])

list_common=[]
for a, b in zip(genres, predList1):
    if a == b:
        list_common.append(a)

print ("Gender: Swsta vre8ikan "+str(len(list_common))+" apo ta "+str(len(genres)))

print("Gender acc: "+str(len(list_common)/len(genres)))

list_common=[]
for a, b in zip(sexualpref, predList2):
    if a == b:
        list_common.append(a)

print ("Sexual Preference: Swsta vre8ikan "+str(len(list_common))+" apo ta "+str(len(sexualpref)))
print("Sexual Preference acc: "+str(len(list_common)/len(sexualpref)))
list_common=[]
for a, b in zip(ages, predList3):
    if a == b:
        list_common.append(a)

print ("Ages: Swsta vre8ikan "+str(len(list_common))+" apo ta "+str(len(ages)))
print("Ages acc: "+str(len(list_common)/len(ages)))


list_common=[]
for a, b in zip(openess, predList4):
    if a == b:
        list_common.append(a)

print ("Openess: Swsta vre8ikan "+str(len(list_common))+" apo ta "+str(len(openess)))
print("Openess acc: "+str(len(list_common)/len(openess)))

list_common=[]
for a, b in zip(Conscientiousness, predList5):
    if a == b:
        list_common.append(a)

print ("Conscientiousness: Swsta vre8ikan "+str(len(list_common))+" apo ta "+str(len(Conscientiousness)))
print("Conscientiousness acc: "+str(len(list_common)/len(Conscientiousness)))

list_common=[]
for a, b in zip(Extraversion, predList6):
    if a == b:
        list_common.append(a)

print ("Extraversion: Swsta vre8ikan "+str(len(list_common))+" apo ta "+str(len(Extraversion)))
print("Extraversion acc: "+str(len(list_common)/len(Extraversion)))

list_common=[]
for a, b in zip(Agreeableness, predList7):
    if a == b:
        list_common.append(a)

print ("Agreeableness: Swsta vre8ikan "+str(len(list_common))+" apo ta "+str(len(Agreeableness)))
print("Agreeableness acc: "+str(len(list_common)/len(Agreeableness)))
list_common=[]
for a, b in zip(Neuroticity, predList8):
    if a == b:
        list_common.append(a)

print ("Neuroticity: Swsta vre8ikan "+str(len(list_common))+" apo ta "+str(len(Neuroticity)))
print("Neuroticity acc: "+str(len(list_common)/len(Neuroticity)))







print("--------------------------------")





predic1=predict1[:,1]
predic3=predict3[:,1]
predic4=predict4[:,1]
predic5=predict5[:,1]
predic6=predict6[:,1]
predic7=predict7[:,1]
predic8=predict8[:,1]



fpr_keras1,tpr_keras1,thresholds_keras1=roc_curve(genres,predic1)
# fpr_keras2,tpr_keras2,thresholds_keras2=roc_curve(sexualpref,predList2)
fpr_keras3,tpr_keras3,thresholds_keras3=roc_curve(ages,predic3)
fpr_keras4,tpr_keras4,thresholds_keras4=roc_curve(openess,predic4)
fpr_keras5,tpr_keras5,thresholds_keras5=roc_curve(Conscientiousness,predic5)
fpr_keras6,tpr_keras6,thresholds_keras6=roc_curve(Extraversion,predic6)
fpr_keras7,tpr_keras7,thresholds_keras7=roc_curve(Agreeableness,predic7)
fpr_keras8,tpr_keras8,thresholds_keras8=roc_curve(Neuroticity,predic8)





auc_keras1=auc(fpr_keras1,tpr_keras1)
#auc_keras2=auc(fpr_keras,tpr_keras2)
auc_keras3=auc(fpr_keras3,tpr_keras3)
auc_keras4=auc(fpr_keras4,tpr_keras4)
auc_keras5=auc(fpr_keras5,tpr_keras5)
auc_keras6=auc(fpr_keras6,tpr_keras6)
auc_keras7=auc(fpr_keras7,tpr_keras7)
auc_keras8=auc(fpr_keras8,tpr_keras8)



plt.figure(1)
plt.plot([0,1],[0,1],'k--')

plt.plot(fpr_keras1,tpr_keras1,label=format(auc_keras1))

plt.plot(fpr_keras3,tpr_keras3,label=format(auc_keras3))
plt.plot(fpr_keras4,tpr_keras4,label=format(auc_keras4))
plt.plot(fpr_keras5,tpr_keras5,label=format(auc_keras5))
plt.plot(fpr_keras6,tpr_keras6,label=format(auc_keras6))
plt.plot(fpr_keras7,tpr_keras7,label=format(auc_keras7))
plt.plot(fpr_keras8,tpr_keras8,label=format(auc_keras8))

# plt.plot(tpr_keras1,fpr_keras1,label=format(auc_keras1))
# #plt.plot(fpr_keras2,tpr_keras2)
# plt.plot(tpr_keras3,fpr_keras3,label=format(auc_keras3))
# plt.plot(tpr_keras4,fpr_keras4,label=format(auc_keras4))
# plt.plot(tpr_keras5,fpr_keras5,label=format(auc_keras5))
# plt.plot(tpr_keras6,fpr_keras6,label=format(auc_keras6))
# plt.plot(tpr_keras7,fpr_keras7,label=format(auc_keras7))
# plt.plot(tpr_keras8,fpr_keras8,label=format(auc_keras8))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


