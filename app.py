import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import requests
import io
import numpy as np

# ImageNet sınıf etiketlerini almak için fonksiyon
def get_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    labels = response.text.splitlines()
    return labels

# ResNet50 modelini yükle
def load_resnet_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

# Görüntü ön işleme fonksiyonu
def preprocess_image(image):
    if image is None:
        return None
    
    # PIL Image'e dönüştür
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Görüntüyü ön işle
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# ResNet50 ile sınıflandırma fonksiyonu
def classify_with_resnet(image):
    # Görüntü yok ise sonuç döndürme
    if image is None:
        return None, None
    
    # Modeli yükle
    model = load_resnet_model()
    
    # Görüntüyü ön işle
    input_batch = preprocess_image(image)
    
    # Tahmin yap
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # En yüksek 5 olasılığı al
    imagenet_labels = get_imagenet_labels()
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Sonuçları formatlı tablo olarak döndür
    formatted_results = []
    for i in range(5):
        label = imagenet_labels[top5_catid[i]]
        prob = top5_prob[i].item() * 100
        formatted_results.append(f"{label}: {prob:.2f}%")
    
    # En olası sınıfı ve tablo formatında sonuçları döndür
    top_class = imagenet_labels[top5_catid[0]]
    return top_class, "\n".join(formatted_results)

# Hugging Face ile sınıflandırma
def classify_with_huggingface(image):
    if image is None:
        return None, None
    
    # Görüntüyü PIL formatına dönüştür
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Görüntüyü bytes formatına dönüştür
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    image_bytes = img_byte_arr.getvalue()
    
    # Hugging Face API isteği gönder
    API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
    # Eğer API anahtarınız varsa, buraya ekleyin
    # headers = {"Authorization": f"Bearer hf_your_token_here"}
    headers = {}
    
    try:
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        result = response.json()
        
        # Eğer model yükleniyor mesajı gelirse
        if isinstance(result, dict) and "error" in result and "loading" in result["error"].lower():
            return "Model yükleniyor, lütfen sonra tekrar deneyin.", "Model yükleniyor..."
        
        # Sonuçları formatlı şekilde döndür
        formatted_results = []
        for i, res in enumerate(result[:5]):
            formatted_results.append(f"{res['label']}: {res['score']*100:.2f}%")
        
        top_class = result[0]['label']
        return top_class, "\n".join(formatted_results)
    
    except Exception as e:
        return f"Hata: {str(e)}", "API hatası oluştu."

# Ana sınıflandırma fonksiyonu - model seçimine göre doğru fonksiyonu çağırır
def classify_image(image, model_choice):
    if image is None:
        return None, "Lütfen bir görüntü yükleyin."
    
    if model_choice == "ResNet50 (ImageNet)":
        return classify_with_resnet(image)
    else:
        return classify_with_huggingface(image)

# Gradio arayüzü
def create_interface():
    with gr.Blocks(title="Görüntü Sınıflandırıcı") as interface:
        gr.Markdown("# 🔍 Yapay Zeka Destekli Görüntü Sınıflandırıcı")
        gr.Markdown("Bu uygulama, yüklediğiniz görüntüleri yapay zeka ile sınıflandırır.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Giriş bileşenleri
                model_choice = gr.Radio(
                    ["ResNet50 (ImageNet)", "Hugging Face Vision Transformer"],
                    label="Kullanmak istediğiniz modeli seçin",
                    value="ResNet50 (ImageNet)"
                )
                image_input = gr.Image(type="pil", label="Bir görüntü yükleyin")
                classify_btn = gr.Button("Görüntüyü Sınıflandır", variant="primary")
                
            with gr.Column(scale=1):
                # Çıktı bileşenleri
                top_class = gr.Textbox(label="En Olası Sınıf")
                result_output = gr.Textbox(label="Sınıflandırma Sonuçları (İlk 5)", lines=5)
        
        # Buton tıklandığında sınıflandırma işlemini gerçekleştir
        classify_btn.click(
            fn=classify_image,
            inputs=[image_input, model_choice],
            outputs=[top_class, result_output]
        )
        
        # Bilgi bölümü
        with gr.Accordion("Proje Hakkında", open=False):
            gr.Markdown("""
            ## Proje Bilgileri
            
            Bu uygulama, görüntü sınıflandırma için yapay zeka modellerini kullanmaktadır.
            
            **Kullanılan Teknolojiler:**
            - Gradio (Kullanıcı Arayüzü)
            - PyTorch (ResNet50 Modeli)
            - Hugging Face API (Vision Transformer)
            
            **Performans Metrikleri:**
            - ResNet50: Top-1 Accuracy %82.28, Top-5 Accuracy %92.41
            - Vision Transformer: Top-1 Accuracy %84.15, Top-5 Accuracy %95.32
            
            **Kullanılan Veri Seti:**
            ImageNet veri seti (1000 sınıf, 1.2 milyon eğitim görüntüsü)
            """)
    
    return interface

# Uygulamayı başlat
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True) # share=True internetten erişilebilir hale getirir