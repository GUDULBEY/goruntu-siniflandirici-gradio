import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import requests
import io
import numpy as np

def get_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    labels = response.text.splitlines()
    return labels

vehicle_labels = [
    "car wheel", "sports car", "convertible", "jeep", "limousine", "minivan", "cab", "racer", "station wagon",
    "pickup", "police van", "ambulance", "fire engine", "garbage truck", "tow truck", "trailer truck",
    "recreational vehicle", "snowplow", "tank"
]

def load_resnet_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def preprocess_image(image):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def classify_with_resnet(image):
    if image is None:
        return None, None, []
    model = load_resnet_model()
    input_batch = preprocess_image(image)

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    imagenet_labels = get_imagenet_labels()
    top100_prob, top100_catid = torch.topk(probabilities, 100)

    filtered_results = []
    chart_data = []
    for i in range(100):
        label = imagenet_labels[top100_catid[i]]
        if label in vehicle_labels:
            prob = top100_prob[i].item() * 100
            filtered_results.append(f"{label}: {prob:.2f}%")
            chart_data.append({"sınıf": label, "olasılık": prob})
            if len(filtered_results) >= 5:
                break

    if filtered_results:
        top_class = filtered_results[0].split(":")[0]
        return top_class, "\n".join(filtered_results), chart_data
    else:
        return "Araç tespit edilemedi", "Yüklenen görüntüde araç bulunamadı.", []

def classify_with_huggingface(image):
    if image is None:
        return None, None, []
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    image_bytes = img_byte_arr.getvalue()

    API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
    headers = {}

    try:
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        result = response.json()

        if isinstance(result, dict) and "error" in result and "loading" in result["error"].lower():
            return "Model yükleniyor", "Model henüz hazır değil, lütfen tekrar deneyin.", []

        formatted_results = []
        chart_data = []
        for i, res in enumerate(result[:5]):
            formatted_results.append(f"{res['label']}: {res['score']*100:.2f}%")
            chart_data.append({"sınıf": res['label'], "olasılık": res['score']*100})

        top_class = result[0]['label']
        return top_class, "\n".join(formatted_results), chart_data

    except Exception as e:
        return f"Hata: {str(e)}", "API hatası oluştu.", []

def generate_result_image(image, label):
    return image

def classify_image(image, model_choice):
    if image is None:
        return None, "Lütfen bir görüntü yükleyin.", None, []

    if model_choice == "ResNet50 (Sadece Araçlar)":
        top_class, details, chart_data = classify_with_resnet(image)
    else:
        top_class, details, chart_data = classify_with_huggingface(image)

    result_image = generate_result_image(image, top_class)
    return top_class, details, result_image, chart_data

with gr.Blocks() as demo:
    gr.Markdown("# 🚗 Akıllı Araç Tanıma Sistemi")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Görüntü Yükle")
        model_choice = gr.Radio(
            ["ResNet50 (Sadece Araçlar)", "Hugging Face Vision Transformer"],
            label="Model Seçimi",
            value="ResNet50 (Sadece Araçlar)"
        )

    classify_btn = gr.Button("🔍 Sınıflandır")
    result_label = gr.Textbox(label="Tahmin")
    result_details = gr.Textbox(label="Detaylı Sonuçlar", lines=5)
    result_image = gr.Image(label="Sonuç Görseli")

    classify_btn.click(
        classify_image,
        inputs=[image_input, model_choice],
        outputs=[result_label, result_details, result_image, gr.State()]
    )

demo.launch()
