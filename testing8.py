import os
import torch
from torchvision import transforms, models
from PIL import Image
from models.models import BaseCMNModel
from modules.tokenizers import Tokenizer
import streamlit as st
import time

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


class VisualExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet101_Weights
        self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-2]))

    def forward(self, images):
        features = self.resnet(images)
        if isinstance(features, tuple):
            features = features[0]
        return features


class BaseCMNModel(torch.nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.visual_extractor = VisualExtractor()
        self.tokenizer = tokenizer
        self.decoder = torch.nn.Linear(args.d_vf * 7 * 7, 500)  # Output logits for vocab size

    def forward_decoder(self, features, max_seq_length):
        batch_size = features.size(0)
        features = features.view(batch_size, -1)  # Flatten
        logits = self.decoder(features)
        output = torch.nn.functional.softmax(logits, dim=-1)
        token_indices = torch.multinomial(output, num_samples=max_seq_length, replacement=True)
        return token_indices


def preprocess_images(image_folder_path):
    image_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) == 0:
        raise ValueError(f"No valid image files found in the folder: {image_folder_path}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = [transform(Image.open(image_path).convert("RGB")) for image_path in image_files]
    images = torch.stack(images, dim=0)
    return images.to(device)


def generate_report(model, tokenizer, images, max_seq_length):
    with torch.no_grad():
        features = model.visual_extractor(images)

        if isinstance(features, tuple):
            features = features[0]

        try:
            report_tensor = model.forward_decoder(features, max_seq_length)
        except AttributeError as e:
            print(f"Error: {e}")
            print("Ensure the correct text generation method is used.")
            return

        # Decode the report
        try:
            vocab_size = 500  #  actual vocabulary size
            report_texts = []
            for batch in report_tensor:
                tokens = batch.tolist()
                valid_tokens = [token for token in tokens if 0 <= token < vocab_size]
                decoded_text = tokenizer.decode(valid_tokens)
                formatted_text = decoded_text.strip().capitalize() + "."
                report_texts.append(formatted_text)

            report_text = "\n\n".join(report_texts)  # Combine with paragraph breaks
        except Exception as decode_error:
            print(f"Error in decoding: {decode_error}")
            raise

        return report_text


if __name__ == "__main__":
    try:
        st.title("Medical Report Generation")
        st.markdown(
            "<h3 style='text-align: center;'>Enter the image path to generate a caption</h3>", unsafe_allow_html=True
        )

        path = st.text_input("Enter the path:")

        if path:
            try:
                # Check if the path is a directory
                if os.path.isdir(path):
                    # Iterate over the files in the directory
                    for filename in os.listdir(path):
                        # Check if the file is an image
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                            # Construct the full path to the image
                            image_path = os.path.join(path, filename)
                            # Open the image using PIL
                            img = Image.open(image_path)
                            # Display the image
                            st.image(img, caption=filename, width=400)
                else:
                    st.error("The path is not a directory.")
            except Exception as e:
                st.error(f"Error loading image: {e}")

        if st.button("Generate Report"):
            args = Args()
            tokenizer = Tokenizer(args)
            model = BaseCMNModel(args, tokenizer).to(device)

            image_tensors = preprocess_images(path)

            report_container = st.empty()

            report=generate_report(model, tokenizer, image_tensors, max_seq_length)
            report_container.write(report)
                

    except Exception as e:
        st.error(f"Error: {e}")
