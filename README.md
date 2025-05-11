🚗 Akıllı Araç Tanıma Sistemi
Bu proje, yüklenen görsellerdeki araçları yapay zeka modelleri kullanarak tanımlayan bir web uygulamasıdır.
📋 Proje Açıklaması
Akıllı Araç Tanıma Sistemi, kullanıcıların yükledikleri görsellerdeki araçları tespit eden ve sınıflandıran bir yapay zeka uygulamasıdır. Uygulama, iki farklı model seçeneği sunarak kullanıcılara farklı sınıflandırma yaklaşımları arasında tercih yapma imkanı sağlar:

ResNet50 (Sadece Araçlar): Bu model, özellikle araç kategorilerine odaklanmıştır ve yalnızca araç sınıflarını (spor araba, minibüs, itfaiye aracı vb.) tespit eder.
Hugging Face Vision Transformer: Daha genel amaçlı bir görüntü sınıflandırma modeli olup, çeşitli nesneleri tanıyabilir.

🔧 Teknolojiler ve Kütüphaneler
Bu proje aşağıdaki teknolojileri kullanmaktadır:

Python - Ana programlama dili
Gradio - Web arayüzü oluşturmak için kullanılan kütüphane
PyTorch - Derin öğrenme modelleri için kullanılan framework
torchvision - Görüntü işleme ve hazır modeller için PyTorch kütüphanesi
ResNet50 - Önceden eğitilmiş derin öğrenme modeli
Hugging Face API - Vision Transformer modeline erişim için
PIL (Python Imaging Library) - Görüntü işleme işlemleri için
NumPy - Sayısal hesaplamalar için

🚀 Kurulum
Projeyi çalıştırmak için aşağıdaki adımları izleyin:

Bu repoyu klonlayın:
git clone https://github.com/GUDULBEY/goruntu-siniflandirici-gradio/tree/main.git


Gerekli paketleri yükleyin:
pip install gradio torch torchvision pillow requests numpy

Uygulamayı çalıştırın:
python app.py

Tarayıcınızda gösterilen adresi açarak uygulamayı kullanmaya başlayabilirsiniz (genellikle http://127.0.0.1:7860).

🖥️ Kullanım

Uygulama arayüzünde "Görüntü Yükle" kısmına bir araç resmi yükleyin.
"Model Seçimi" kısmından kullanmak istediğiniz modeli seçin:

ResNet50 (Sadece Araçlar)
Hugging Face Vision Transformer


"🔍 Sınıflandır" butonuna tıklayın.
Sonuçları görüntüleyin:

"Tahmin" kısmında tespit edilen araç türü
"Detaylı Sonuçlar" kısmında olasılık değerleriyle birlikte en yüksek olasılıklı sınıflar
"Sonuç Görseli" kısmında yüklediğiniz görüntü



🌟 Özellikler

İki farklı yapay zeka modeli seçeneği
Araç türlerini yüksek doğrulukla tespit etme
Basit ve kullanıcı dostu arayüz
Detaylı sınıflandırma sonuçları
Türkçe arayüz desteği

📝 Notlar

ResNet50 modeli, sadece belirli araç türlerini tespit edecek şekilde filtrelenmiştir.
Hugging Face Vision Transformer modeli, ilk kullanımda yüklenme süresi gerektirebilir.
Uygulamanın doğru çalışması için internet bağlantısı gereklidir (model dosyaları ve sınıf etiketleri indirilir).

🤝 Katkıda Bulunma
Bu projeye katkıda bulunmak isterseniz:

Bu repoyu fork edin.
Yeni bir branch oluşturun (git checkout -b feature/yeni-ozellik).
Değişikliklerinizi commit edin (git commit -am 'Yeni özellik: Açıklama').
Branch'inizi push edin (git push origin feature/yeni-ozellik).
Bir Pull Request oluşturun.

📜 Lisans
Bu proje MIT Lisansı ile lisanslanmıştır.
📧 İletişim
Sorularınız veya geri bildirimleriniz için lütfen GitHub profil sayfamdaki iletişim bilgilerini kullanın.
