import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Paths
images_path = "/kaggle/input/r2gencmn/iu_xray/images"
annotations_path = "/kaggle/input/r2gencmn/iu_xray/annotation.json"

# Load Metadata
with open(annotations_path, 'r') as f:
    metadata = json.load(f)

# Dataset Class
class XRayDataset(Dataset):
    def _init_(self, data, images_path, feature_extractor, tokenizer, max_length=512):
        self.data = []
        self.images_path = images_path
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Process data to include all images from subfolders
        for item in data:
            subfolder_images = item['image_path']  # List of image paths in the subfolder
            report = item['report']  # Corresponding report
            for image_file in subfolder_images:
                self.data.append({"image_path": os.path.join(self.images_path, image_file), "report": report})

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        report = item['report']

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenize report
        report_tokens = self.tokenizer(
            report,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": report_tokens.input_ids.squeeze(0),
            "attention_mask": report_tokens.attention_mask.squeeze(0),
            "image_path": image_path  # Include full image path for later use
        }

# Evaluation Metrics Function
def evaluate_predictions(predictions, references):
    """
    Evaluate model predictions using BLEU, ROUGE-L, and CIDEr scores.
    """
    bleu_scores = {"BLEU-1": [], "BLEU-2": [], "BLEU-3": [], "BLEU-4": []}
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    chencherry = SmoothingFunction()

    for pred, ref in zip(predictions, references):
        for n in range(1, 5):
            weights = tuple([1/n] * n) + (0,) * (4 - n)
            bleu_scores[f"BLEU-{n}"].append(
                sentence_bleu([ref.split()], pred.split(), weights=weights, smoothing_function=chencherry.method1)
            )
        rouge_score = rouge_scorer_obj.score(ref, pred)['rougeL'].fmeasure
        rouge_scores.append(rouge_score)

    avg_bleu_scores = {key: np.mean(value) for key, value in bleu_scores.items()}
    avg_rouge_l = np.mean(rouge_scores)

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score({i: [r] for i, r in enumerate(references)},
                                                {i: [p] for i, p in enumerate(predictions)})

    return {**avg_bleu_scores, "ROUGE-L": avg_rouge_l, "CIDEr": cider_score}

# Precision, Recall, F1-Score Function
def calculate_precision_recall_f1(predictions, references):
    """
    Calculate precision, recall, and F1-score for the model's predictions.
    """
    pred_tokens = [set(pred.split()) for pred in predictions]
    ref_tokens = [set(ref.split()) for ref in references]

    # Multi-label binarization
    mlb = MultiLabelBinarizer()
    all_tokens = pred_tokens + ref_tokens
    mlb.fit(all_tokens)

    pred_binarized = mlb.transform(pred_tokens)
    ref_binarized = mlb.transform(ref_tokens)

    precision = precision_score(ref_binarized, pred_binarized, average="micro")
    recall = recall_score(ref_binarized, pred_binarized, average="micro")
    f1 = f1_score(ref_binarized, pred_binarized, average="micro")

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

# Prediction and Evaluation Function with Image Paths
def predict_and_evaluate_with_paths(model, test_loader, tokenizer, device):
    """
    Predict captions and save results with image paths and generated reports.
    """
    predictions, references, image_paths = [], [], []
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"]

            image_paths_batch = batch["image_path"]

            # Generate captions
            generated_ids = model.generate(pixel_values)
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            predictions.extend(preds)
            references.extend(refs)
            image_paths.extend(image_paths_batch)

    # Save results to CSV
    results_df = pd.DataFrame({
        "Image Path": image_paths,
        "Generated Report": predictions,
        "Reference Report": references
    })
    results_df.to_csv("predictions_with_paths.csv", index=False)

    # Evaluate predictions
    scores = evaluate_predictions(predictions, references)

    # Calculate Precision, Recall, and F1-Score
    prf_scores = calculate_precision_recall_f1(predictions, references)

    return scores, prf_scores, results_df

# Model Setup
def setup_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding token
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, feature_extractor

# Training Loop
def train_model(model, train_loader, optimizer, device, num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            outputs = model(pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Validation Loop
def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            outputs = model(pixel_values=pixel_values, labels=input_ids)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

# Main Workflow
if _name_ == "_main_":
    # Model and Data Setup
    model, tokenizer, feature_extractor = setup_model()
    train_data, val_test_data = train_test_split(metadata["train"], test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

    train_dataset = XRayDataset(train_data, images_path, feature_extractor, tokenizer)
    val_dataset = XRayDataset(val_data, images_path, feature_extractor, tokenizer)
    test_dataset = XRayDataset(test_data, images_path, feature_extractor, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Training Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Train and Validate
    train_model(model, train_loader, optimizer, device)
    validate_model(model, val_loader, device)

    # Predict and Evaluate with Image Paths
    scores, prf_scores, results_df = predict_and_evaluate_with_paths(model, test_loader, tokenizer, device)

    # Print Evaluation Metrics
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
    print(f"Precision: {prf_scores['Precision']:.4f}")
    print(f"Recall: {prf_scores['Recall']:.4f}")
    print(f"F1-Score: {prf_scores['F1-Score']:.4f}")

    print("Results saved to predictions_with_paths.csv")