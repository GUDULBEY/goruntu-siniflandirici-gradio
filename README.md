# goruntu-siniflandirici-gradio
görüntü siniflandirma ile araç tanıma
Gradio ile Yapay Zeka Destekli Görüntü Sınıflandırıcı
Bu proje, kullanıcıların yükledikleri görselleri sınıflandırabilen bir yapay zeka uygulamasıdır. Sistem, görüntüleri ön işleme tabi tutarak sınıflandırır ve sonuçları kullanıcıya Gradio arayüzü ile sunar.
Özellikler

Kullanıcıların bilgisayarından görüntü yükleyebilmesi
Görüntülerin otomatik olarak ön işlenmesi (boyutlandırma, normalleştirme)
İki farklı model seçeneği:

ResNet50 (ImageNet üzerinde eğitilmiş)
Hugging Face Vision Transformer


Sınıflandırma sonuçlarının olasılık değerleri ile gösterilmesi
Kullanıcı dostu Gradio web arayüzü

Kurulum
Gereksinimler
Projeyi çalıştırmak için Python 3.8+ ve aşağıdaki kütüphanelere ihtiyacınız var:
bashpip install -r requirements.txt
Hugging Face API (Opsiyonel)
Vision Transformer modelini kullanmak için bir Hugging Face API anahtarı almanız önerilir:

Hugging Face üzerinde bir hesap oluşturun
Ayarlar sayfasından bir API anahtarı edinin
app.py dosyasındaki ilgili yeri düzenleyin:

python# Satır 85'i değiştirin:
headers = {"Authorization": f"Bearer hf_your_token_here"}
Çalıştırma
Projeyi çalıştırmak için:
bashpython app.py
Bu komut, yerel bir Gradio sunucusu başlatacak ve tarayıcınızda uygulamayı açacaktır. Ayrıca uygulamanın geçici bir süre için erişilebileceği bir genel URL de sunacaktır.
Kullanım

Model seçin (ResNet50 veya Vision Transformer)
"Bir görüntü yükleyin" alanına tıklayarak veya sürükleyerek bir görüntü yükleyin
"Görüntüyü Sınıflandır" butonuna tıklayın
Sonuçları inceleyin

Teknik Detaylar
ResNet50 Modeli

Önceden eğitilmiş ImageNet modeli
1000 farklı nesne sınıfını tanıyabilme
Top-1 Accuracy: %82.28
Top-5 Accuracy: %92.41

Vision Transformer (ViT)

Hugging Face'in Vision Transformer modeli
State-of-the-art performans
Top-1 Accuracy: %84.15
Top-5 Accuracy: %95.32

Görüntü Ön İşleme

Yeniden boyutlandırma (256x256)
Merkez kırpma (224x224)
Tensor dönüşümü
ImageNet normalizasyonu

Proje Yapısı
goruntu-siniflandirici-gradio/
├── app.py           # Ana uygulama dosyası
├── requirements.txt # Gerekli kütüphaneler
└── README.md        # Proje dokümantasyonu
