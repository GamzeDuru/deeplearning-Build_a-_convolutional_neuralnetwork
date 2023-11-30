import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#CIFAR-10 (yani sinir ağımızı eğitirken kullanacağımız resimler) veri seti Keras modülüne dahil edilmiştir.Eğitim ve test veri kümelerine bölünmüştür.

(X_train, y_train), (X_test,y_test)=tf.keras.datasets.cifar10.load_data()

#Oran 5e 1dir. train:5k , test:1k
print(f"X_train: {len(X_train)}")
print(f"X_test: {len(X_test)}")

# resimdeki pikselleri sayı halinde gösterir. 
print(X_test[789])

#pikselleri sayı halinde değilde gorsel halindde inceleyelim. Bunun içim matplotlib kutuphanesini kullacagız(imshow:numpy dizisi alır)
plt.imshow(X_test[789])
plt.show()
print(X_test[789].shape)

''' derin öğrenme modelini eğitmek, doğrulamak(val) ve test etmek için üç benzersiz veri kümesine ihtiyacımız var.
     4'e 1'e 1.

'''

#4000: ->son 10k goruntuyu ata
X_val=X_train[40000:]
y_val=y_train[40000:]

#ilk 40k goruntuyu ata
X_train=X_train[:40000]
y_train=y_train[:40000]


print(f"X_train: {len(X_train)}")
print(f"X_val: {len(X_val)}")
print(f"X_test: {len(X_test)}")


#Görüntünün piksel değerlerini normalleştirmek model eğitimine büyük ölçüde yardımcı olacaktır. Görüntülerdeki piksel değerleri 0 ile 255 arasındadır.
#Görüntülerdeki piksel değerleri 0 ile 255 arasındadır. Bu değerleri 255'e bölerek 0 ile 1 arasında ölçeklendireceğiz.

X_train=X_train/255
X_val=X_val/255
X_test=X_test/255

#model oluşturma aşaması

model=tf.keras.Sequential()

'''
 Keras'ın Conv2D katmanı ile ilk evrişim katmanımızı ekliyoruz.
 Düğüm sayısı : 32
 Çekirdek boyutu : 3x3 
 Strides parametresi : Evrişim çekirdeğinin adımlarını belirtir(adım 1 ise filtre her seferinde 1 piksel hareket)
 Padding:görsellerin kenarlarında bilgi kaybını önlemek için dolgu yapma işlemi(Dolgunun kenar değerleriyle aynı olması gerekir)
 activation fonk. : Relu 
 Giriş şekli: (32,32,3)
'''

'''
*kernel_size : 2B evrişim penceresinin yüksekliğini ve genişliğini belirten bir tam sayı veya 2 tam sayıdan oluşan bir grup/liste.
Tüm uzamsal boyutlar için aynı değeri belirtmek amacıyla tek bir tam sayı olabilir.

*strides : Evrişimin yükseklik ve genişlik boyunca adımlarını belirten bir tamsayı veya demet/2 tam sayıdan oluşan liste. 
Tüm uzamsal boyutlar için aynı değeri belirtmek amacıyla tek bir tam sayı olabilir

*dolgu"valid" : veya biri "same"(büyük/küçük harfe duyarlı). "valid"dolgu yok anlamına gelir. 
"same"girişin sol/sağ veya yukarı/aşağı eşit şekilde sıfırlarla doldurulmasına neden olur.
padding="same" ve olduğunda strides=1, çıktı girişle aynı boyuta sahiptir.

*etkinleştirme : Kullanılacak etkinleştirme işlevi. Hiçbir şey belirtmezseniz herhangi bir etkinleştirme uygulanmaz

*filtreler : Tamsayı, çıktı alanının boyutluluğu (yani evrişimdeki çıktı filtrelerinin sayısı).

*pool_size : Girişin her kanalı için bir giriş penceresi (boyutu ile tanımlanan) üzerinden maksimum değeri alarak,
 girişin uzamsal boyutları (yükseklik ve genişlik) boyunca alt örneğini alır 
'''

model.add ( tf.keras.layers.Conv2D(
          32,
          kernel_size=(3,3),
          strides=(1,1),
          padding="same",
          activation = "relu" ,
          input_shape = (32,32,3) ))

model.add(tf.keras.layers.MaxPooling2D(2,2))

'''
Bir evrişim katmanı, bir maksimum havuzlama katmanı ve tekrar bir evrişim katmanı eklenir.
Görüntünün boyutunu küçültmek ve tespit edilen bazı özellikleri daha sağlam hale getirmek için maksimum havuzlama katmanını ekleriz.(eğitimi hızlandırır)

'''

model.add ( tf.keras.layers.Conv2D(
          64,
          kernel_size=(3,3),
          strides=(1,1),
          padding="same",
          activation = "relu" ,
          input_shape = (32,32,3) ))

model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add ( tf.keras.layers.Conv2D(
          64,
          kernel_size=(3,3),
          strides=(1,1),
          padding="same",
          activation = "relu" ,
          input_shape = (32,32,3) ))



"""
 sınıflandırma kısmına geçiyoruz. 2B evrişim katmanlarını ve 1B yoğun katmanları birbirine bağlamak için onları aynı boyutlara getirmemiz gerekiyor. 
 İşte tam bu noktada “düzleştirme” yöntemi imdadımıza yetişiyor. Yani tüm piksel değerlerini alıp tek tek 1 boyutlu bir diziye yerleştiriyoruz.
"""

model.add(tf.keras.layers.Flatten())

"""
iki yoğun katman ekliyoruz. Her biri 64 düğüme, ReLU aktivasyonuna ve 0,5 ayrılmaya sahiptir.
 Bunların daha sonra performansı artırmak için ayarlanabilen rastgele sayılar olduğunu unutmayın.
‘Dense’, katman türüdür. Dense, çoğu durumda çalışan standart bir katman türüdür. Yoğun bir katmanda, önceki katmandaki tüm düğümler mevcut katmandaki düğümlere bağlanır.
rastgele seçilen belirli nöron setlerinin eğitim aşamasında birimlerin (yani nöronların) göz ardı edilmesi anlamına gelir. “Yok saymak” derken, belirli bir ileri veya geri geçiş sırasında bu birimlerin dikkate alınmamasını kastediyorum.
"""
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0,5))

model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0,5))
"""
 Çıkış katmanının zamanı geldi. 10 sınıfımız olduğu için 10 düğüm ekliyoruz ve Softmax aktivasyon fonksiyonunu kullanıyoruz. 
 Çok sınıflı sınıflandırma problemleri için çıktı katmanında Softmax kullandığımızı unutmayın. Son olarak modeli derliyoruz.
 Optimizasyon aracımız olan “Adam”ı tanımlıyoruz ve çok sınıflı bir sınıflandırma problemini çözmeye çalıştığımız için “Seyrek Kategorik Çapraz Entropi” 
 kayıp fonksiyonunu kullanıyoruz. Bununla modelimiz eğitilmeye hazır!
"""
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

"""
Veri setimizde 60.000 adet görselimiz var, bunların 40.000 tanesini eğitim için kullanıyoruz. Ama dikkatli olmalıyız! 
Tüm bu resimlerle aynı anda eğitim almak hafızamızı ve diğer donanım kaynaklarımızı hızla tüketir. 
Bunu önlemek için mini-batching adı verilen bir teknik kullanıyoruz.
 Mini gruplamayla, tüm örneklere aynı anda bakmak yerine model aynı anda yalnızca az sayıda örneği görecektir. 
 Bu yöntemin birçok avantajı vardır; Mini gruplama, eğitimi hızlandırarak ve daha az bellek kullanarak model eğitimini önemli ölçüde artırır.
 Ayrıca birden fazla GPU'muz varsa, bu GPU'ları aynı anda farklı grupları eğitmek için kullanabiliriz.
 Bu yüzden buraya “batch_size” adında yeni bir parametre eklememiz gerekiyor. 
 Ve muhtemelen tahmin edebileceğiniz gibi bu başka bir hiperparametredir, yani onun için optimum değeri bulmamız gerekecek.
 Bazen daha düşük, bazen daha yüksek daha iyidir.
 Ancak bilgisayarlar ikili sayılarla çalıştığından, hesaplama hızının artması için 2'nin üssü olan bir sayının seçilmesi şiddetle tavsiye edilir. 
 50 dönem için 128'lik parti büyüklüğüyle deneyelim. Bu biraz zaman alacak.

"""
results= model.fit (X_train, y_train,
                    batch_size=128,
                    epochs=50,
                    validation_data=(X_val,y_val)
                   )

plt.plot(results.history["loss"],label="loss")
plt.plot(results.history["val_loss"],label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.plot(results.history["accuracy"],label="accuracy")
plt.plot(results.history["val_accuracy"],label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
"""
 modelin aşırı uyum sağlamaya başladığını görüyoruz. Çünkü hem eğitim hem de değerlendirme hatları birbirinden uzaklaşmaya başlıyor diyebiliriz. 
 Bunu önlemenin yöntemlerinden biri de “Erken Durdurma” yöntemidir.
 Erken durdurma, modelin aşırı uyum sağlamaya başladığı dönemde eğitim sürecini durdurmaktır.
 Yani modeli yalnızca 20 dönem boyunca eğitirsek aşırı uyumun önüne geçmiş oluruz.
 Kayıp ve doğruluğu hesaplamak ve modelin performansı hakkında daha fazla bilgi edinmek için test veri kümesini de kullanabiliriz. 
 1.21 kayıplı ve %70'in üzerinde doğruluğa sahip bir modelimiz var.
 Eğer bu sonuçlar elimizdeki sorun için yeterliyse, modeli daha fazla optimize etmemize gerek yok.
 Ancak kaybın mümkün olduğu kadar 0'a yakın olmasını istediğimizi unutmayın. Ve doğruluk mümkün olduğunca %100'e yakındır.
 Yani gerekirse hiperparametre optimizasyonu yaparak bu değerlere yaklaşmayı deneyebiliriz. 
 
"""
#evaluate:degerlendirmek, ölçmek
model.evaluate(X_test,y_test)
#tensör, çok boyutlu verinin simgelenebildiği geometrik bir nesnedir. 

"""
Doğru sınıflandırılıp sınıflandırılmadığını görmek için önceki hücrelerdeki gemi görüntüsünü kontrol edelim.
Yeniden şekillendirme yöntemini kullanıyoruz çünkü tahmin yöntemi, bir grup görüntünün tahmin edilmesini bekliyor. 
Ama biz tek bir görüntü üzerinden tahmin yapmak istiyoruz.
Görüntüyü (1,32,32,3) olarak yeniden şekillendirerek 32'ye 32'ye 3'lük bir görüntümüz olduğunu söylüyoruz. 
Daha sonra tahmin sonuçlarını yazdırıyoruz. Çıktı, her sınıfa ait bir görüntünün olasılığını veriyor. 

"""
#prediction:tahmin
prediction_result=model.predict(X_test[789].reshape(1,32,32,3))
prediction_result
"""
En yüksek olasılığı bulmak için argmax ve max yöntemlerini kullanabiliriz.
argmax, tahmin edilen sınıf olan en yüksek değerin dizinini döndürür ve max, olasılığı döndürür.
Bunları bulalım ve değişkenlere atayalım.
 Bu görüntü 0,99 olasılıkla 8. sınıfa aittir. Yani bu, yüzde doksan dokuz ihtimalle 8. sınıfa ait olduğu anlamına geliyor.
 8. sınıfın “gemi” sınıfı olduğunu biliyoruz. Yani modelin bu görüntüyü doğru tahmin ettiğini söyleyebiliriz.
 Daha önce hafif bir aşırı uyum durumumuzun olduğunu söylemiştik. Ancak hiperparametrelere ince ayar yaparak bunun üstesinden gelebiliriz.
"""
predicted_class=prediction_result.argmax()
predicted_probability=prediction_result.max()

print(f"This image belongs to class {predicted_class} with {predicted_probability} probability % ")


