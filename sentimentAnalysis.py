##### SONUCA GÖRE #####
## DOĞRULUK ORANI: %90 ####

# Yardımcı kaynak Text Classification Google Machine Learning Guides.
# Bu analizde pozitif oylamaları 1 olarak ve olumsuz oylamaları da 0 olarak düzenledim.
# tf-idf ve Multi-layer Perceptron kullanıldı.
# Keras API ile birlikte tensorflow kullanıldı.


### GEREKLİ KÜTÜPHANELER YÜKLENDİ ####
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

path = 'aclImdb'

#### VERİ SETİ YÜKLEMESİ ###
def shuffle(X, y):
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    return X, y

def load_imdb_dataset(path):
    imdb_path = os.path.join('aclImdb')

    # veri tabanını yüklüyoruz.
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    for dset in ['train', 'test']:
        for cat in ['pos', 'neg']:
            dset_path = os.path.join(imdb_path, dset, cat)
            for fname in sorted(os.listdir(dset_path)):
                if fname.endswith('.txt'):
                    with open(os.path.join(dset_path, fname)) as f:
                        if dset == 'train': train_texts.append(f.read())
                        else: test_texts.append(f.read())
                    label = 0 if cat == 'neg' else 1
                    if dset == 'train': train_labels.append(label)
                    else: test_labels.append(label)

    # Veri setini np.array şekline dönüştürüyoruz.
    train_texts = np.array(train_texts)
    train_labels = np.array(train_labels)
    test_texts = np.array(test_texts)
    test_labels = np.array(test_labels)

    # Aslında textprocessing.py dosyasında yaptığımız işlemin aynısını burada da uyguluyoruz.
    # Bunun tam adına veri setini karıştırma deniyor. Metini ilk önce integer değerlere
    # sonra da tekrar metin haline çeviriyoruz. Sadece farklı metodlar uygulandı.
    train_texts, train_labels = shuffle(train_texts, train_labels)
    test_texts, test_labels = shuffle(test_texts, test_labels)

    # Veri setini tekrar eski haline getirme.
    return train_texts, train_labels, test_texts, test_labels

trX, trY, ttX, ttY = load_imdb_dataset(path)

print('Eğitim seti örnek hali :', trX.shape)
print('Eğitim seti etiketi  :', trY.shape)
print('Test seti örnek hali  :', ttX.shape)
print('Test etiketi   :', ttY.shape)

uniq_class_arr, counts = np.unique(trY, return_counts=True)

print('Unique sınıf :', uniq_class_arr)
print('Unique sınıf sayısı : ', len(uniq_class_arr))

for _class in uniq_class_arr:
    print('Sınıf sayısı ', uniq_class_arr[_class], ' : ', counts[_class])

size_of_samp = 10
rand_samples_to_check = np.random.randint(len(trX), size=size_of_samp)

for samp_num in rand_samples_to_check:
    print ('============================================================')
    print (trX[samp_num], '||', trY[samp_num])
    print ('============================================================')

plt.figure(figsize=(15, 10))
plt.hist([len(sample) for sample in list(trX)], 50)
plt.xlabel('Eğitim seti örnek hali')
plt.ylabel('Örnek sayısı')
plt.title('Örnek uzunluk dağılımı')
plt.show()

kwargs = {
    'ngram_range' : (1, 1),
    'dtype' : 'int32',
    'strip_accents' : 'unicode',
    'decode_error' : 'replace',
    'analyzer' : 'word'
}

vectorizer = CountVectorizer(**kwargs)
vect_texts = vectorizer.fit_transform(list(trX))
all_ngrams = vectorizer.get_feature_names()
num_ngrams = min(50, len(all_ngrams))
all_counts = vect_texts.sum(axis=0).tolist()[0]

all_ngrams, all_counts = zip(*[(n, c) for c, n in sorted(zip(all_counts, all_ngrams), reverse=True)])
ngrams = all_ngrams[:num_ngrams]
counts = all_counts[:num_ngrams]

idx = np.arange(num_ngrams)

plt.figure(figsize=(30, 30))
plt.bar(idx, counts, width=0.8)
plt.xlabel('N-gramlar')
plt.ylabel('Frekanslar')
plt.title('Ngramların frekans dağılımları')
plt.xticks(idx, ngrams, rotation=45)
plt.show()

NGRAM_RANGE = (1, 2)
TOP_K = 20000
TOKEN_MODE = 'word'
MIN_DOC_FREQ = 2


def ngram_vectorize(train_texts, train_labels, val_texts):
    kwargs = {
        'ngram_range': NGRAM_RANGE,
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': TOKEN_MODE,
        'min_df': MIN_DOC_FREQ,
    }

    # Kelimeleri eğitim setinden öğrenme ve doğrulama setiyle vektörize etme
    tfidf_vectorizer = TfidfVectorizer(**kwargs)
    x_train = tfidf_vectorizer.fit_transform(train_texts)
    x_val = tfidf_vectorizer.transform(val_texts)

    # F_classif tarafından ölçülen özellik önemine sahip en iyi k'yı seçme
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val

def get_last_layer_units_and_activation(num_classes):
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    op_units, op_activation = get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


def train_ngram_model(data, learning_rate=1e-3, epochs=1000, batch_size=128, layers=2, units=64,
                      dropout_rate=0.2):
    num_classes = 2

    # Datayı elde ediyoruz.
    trX, trY, ttX, ttY = data

    # Datayı vektörler haline çeviriyoruz.
    x_train, x_val = ngram_vectorize(trX, trY, ttX)

    # Artık model örneği oluşturuyoruz.
    model = mlp_model(layers, units=units, dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:], num_classes=num_classes)

    # Modeli parametreyle derliyoruz.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Doğrulama kaybını erken durdurmak için bir arama daha oluşturuyoruz.
    # İki ardışık denemede hata payı azalmazsa, eğitimi durdur.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Eğitim ve doğrulama modeli
    history = model.fit(x_train, trY, epochs=epochs, validation_data=(x_val, ttY),
                        verbose=2, batch_size=batch_size, callbacks=callbacks)

    # Sonuçları yazdır
    history = history.history
    val_acc = history['val_acc'][-1]
    val_loss = history['val_loss'][-1]
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=val_acc, loss=val_loss))

    # Modeli kaydet
    model.save('IMDB_mlp_model_' + str(val_acc) + '_' + str(loss) + '.h5')
    return val_acc, val_loss

results = train_ngram_model((trX, trY, ttX, ttY))

print ('With lr=1e-3 | val_acc={results[0]} | val_loss={results[1]}'.format(results=results))
print ('===========================================================================================')

results

##### SONUCA GÖRE #####
## DOĞRULUK ORANI: %90 ####