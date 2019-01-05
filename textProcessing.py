
######## SONUCA GÖRE ##########
## DOĞRULUK ORANI %87
## YANILMA PAYI %33


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3)

print(tf.__version__)

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#numwords bu datada en çok kullanılan kelimeleri tutar, nadir kullanılanları atar.
#pythonın kendi veri seti olan imdb veri seti hazır işlenmiş bir veri setidir. Yani 0'ın negatif olduğunu
# 1'in isse pozitif olduğunu anlar.
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

#her satırdaki veri setinin uzunluğunun birbirinden farklı olduğunu kanıtlamış olduk.
#yapay sinir ağları için hepsinin aynı uzunlukta olması gerekir ve bunu sonradan düzenleyeceğiz.

#Daha önceden metni tam sayıya dönüştürdük, şimdi de
#Tam sayıların tekrar metne dönüştürmek için
#string mapping için tamsayı içeren bir sözlük nesnesini sorgulamak için bir fonksiyon oluşturuyoruz.
print(len(train_data[0]), len(train_data[1]))

word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

#Film incelemelerinin aynı uzunlukta olması gerektiğinden,
# uzunlukları standartlaştırmak için pad_sequences işlevini kullanacağız
print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

#örneklerin uzunlukları
print(len(train_data[0]), len(train_data[1]))

#ilk yorumun incelenmesi
print(train_data[0])

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

#İlk katman bir Gömme katmanıdır. Bu katman tamsayı ile kodlanmış kelime dağarcığını alır ve her kelime indeksi için gömme vektörünü arar.
#Bu vektörler model trenler olarak öğrenilir. Vektörler, çıktı dizisine bir boyut ekler. Elde edilen boyutlar: (parti, sekans, gömme).
#GlobalAveragePooling1D katmanı, her bir örnek için dizi boyutunun ortalaması alınarak sabit uzunluklu bir çıktı vektörü döndürür.
#Şimdi, modeli optimize ediyoruz ve loss işlevini kullanacak şekilde yapılandırıyoruz:

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Doğrulama seti oluşturuyoruz.
#Eğitim sırasında, modelin daha önce görmediği verilerdeki doğruluğunu kontrol etmek istiyoruz.
#Orijinal eğitim verilerinden 10.000 örnek ayırarak bir doğrulama oluşturun.


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
#Eğitim setini oluşturuyoruz. Eğitim sırasında,
# validasyon setindeki 10.000 verideki modelin kaybını ve doğruluğunu görücez.

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)
#sonuca göre yanılma payı yaklaşık olarak %33 doğruluk oranı ise %87 olarak gözükmektedir.

#Projeye ek olarak grafikle desteklemek istiyorum.

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

######### SONUCA GÖRE ##########
## DOĞRULUK ORANI %87
## YANILMA PAYI %33XX