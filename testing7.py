import streamlit as st
import os
import torch
from torchvision import transforms, models
from PIL import Image
from models.models import BaseCMNModel
from modules.tokenizers import Tokenizer

# Configuration
model_checkpoint_path = r'C:\Users\Hp\OneDrive\Desktop\results\iu_xray\current_checkpoint.pth'
max_seq_length = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Args:
    def __init__(self):
        self.image_dir = r"C:\Users\Hp\OneDrive\Desktop\iu_xray\images"
        self.ann_path = r"C:\Users\Hp\OneDrive\Desktop\iu_xray\annotation.json"
        self.dataset_name = 'iu_xray'
        self.max_seq_length = 40
        self.visual_extractor = 'resnet101'
        self.visual_extractor_pretrained = True
        self.d_model = 512
        self.d_ff = 2048  # Feed-forward network dimension
        self.d_vf = 2048  # Visual feature dimension
        self.num_heads = 8  # Number of attention heads
        self.num_layers = 6  # Number of layers
        self.beam_size = 3
        self.dropout = 0.1  # Dropout rate
        self.topk = 5  # Top-k predictions
        self.cmm_size = 2048  # Common memory size
        self.cmm_dim = 512  # Common memory dimension
        self.save_dir = r'C:\Users\Hp\OneDrive\Desktop\results\iu_xray'
        self.weight_decay = 5e-5
        self.threshold = 3  # Threshold value
        self.drop_prob_lm = 0.5  # Dropout probability for language modeling
        self.bos_idx = 0  # Beginning-of-sequence token index
        self.eos_idx = 1  # End-of-sequence token index
        self.pad_idx = 2  # Padding token index
        self.use_bn = False  # Use Batch Normalization


def preprocess_images(uploaded_files):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        images.append(transform(image))
    return torch.stack(images).to(device)


def generate_report(model, tokenizer, images, max_seq_length):
    with torch.no_grad():
        features = model.visual_extractor(images)
        if isinstance(features, tuple):
            features = features[0]
        report_tensor = model.forward_decoder(features, max_seq_length)
        vocab_size = 500  # Example vocabulary size
        report_texts = []
        for batch in report_tensor:
            tokens = batch.tolist()
            valid_tokens = [token for token in tokens if 0 <= token < vocab_size]
            decoded_text = tokenizer.decode(valid_tokens)
            formatted_text = decoded_text.strip().capitalize() + "."
            report_texts.append(formatted_text)
        return "\n\n".join(report_texts)


# Streamlit App
st.title("Radiology Report Generation")
st.write("Upload images to generate a radiology report.")

args = Args()
tokenizer = Tokenizer(args)
model = BaseCMNModel(args, tokenizer).to(device)

uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    try:
        # Preprocess images
        image_tensors = preprocess_images(uploaded_files)
        # Generate report
        report = generate_report(model, tokenizer, image_tensors, max_seq_length)
        st.success("Generated Radiology Report:")
        st.text_area("Report", value=report, height=300)
    except Exception as e:
        st.error(f"Error: {e}")
