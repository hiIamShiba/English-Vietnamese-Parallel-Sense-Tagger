import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.amp import autocast

# =======================
# CONFIG
# =======================
MODEL_NAME = "xlm-roberta-base"
MODEL_PATH = "best_model.pt"
TEST_FILE = "test.jsonl"

MAX_LEN = 128
TEST_BATCH_SIZE = 16      # tăng lên 16 nếu GPU local khỏe
USE_FP16 = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

# =======================
# MODEL DEFINITION
# =======================
class XLMRParallelSenseTagger(nn.Module):
    def __init__(self, model_name, num_en_labels, num_vi_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.en_classifier = nn.Linear(hidden, num_en_labels)
        self.vi_classifier = nn.Linear(hidden, num_vi_labels)

    def forward(self, en_inputs, vi_inputs):
        en_out = self.encoder(**en_inputs)
        vi_out = self.encoder(**vi_inputs)

        en_logits = self.en_classifier(en_out.last_hidden_state)
        vi_logits = self.vi_classifier(vi_out.last_hidden_state)

        return en_logits, vi_logits, en_out.last_hidden_state, vi_out.last_hidden_state


# =======================
# LOAD MODEL
# =======================
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

EN_LABEL2ID = ckpt["en_label2id"]
VI_LABEL2ID = ckpt["vi_label2id"]

EN_ID2LABEL = {v: k for k, v in EN_LABEL2ID.items()}
VI_ID2LABEL = {v: k for k, v in VI_LABEL2ID.items()}

model = XLMRParallelSenseTagger(
    MODEL_NAME,
    num_en_labels=len(EN_LABEL2ID),
    num_vi_labels=len(VI_LABEL2ID)
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =======================
# TOKENIZATION (BATCH)
# =======================
def batch_encode(sentences):
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return {k: v.to(DEVICE) for k, v in encoded.items()}

# =======================
# TEST LOOP
# =======================
total_en, sense_en = 0, 0
total_vi, sense_vi = 0, 0

total_cosine = 0.0
total_pairs = 0

batch_en, batch_vi = [], []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Running test inference"):
        sample = json.loads(line)
        batch_en.append(sample["en"])
        batch_vi.append(sample["vi"])

        if len(batch_en) == TEST_BATCH_SIZE:
            en_inputs = batch_encode(batch_en)
            vi_inputs = batch_encode(batch_vi)

            with torch.inference_mode():
                with autocast("cuda", enabled=USE_FP16):
                    en_logits, vi_logits, en_hidden, vi_hidden = model(
                        en_inputs, vi_inputs
                    )

            en_preds = torch.argmax(en_logits, dim=-1)
            vi_preds = torch.argmax(vi_logits, dim=-1)

            for i in range(len(batch_en)):
                en_len = en_inputs["attention_mask"][i].sum().item()
                vi_len = vi_inputs["attention_mask"][i].sum().item()

                # Sense ratio
                for j in range(en_len):
                    total_en += 1
                    if EN_ID2LABEL[en_preds[i, j].item()] != "O":
                        sense_en += 1

                for j in range(vi_len):
                    total_vi += 1
                    if VI_ID2LABEL[vi_preds[i, j].item()] != "O":
                        sense_vi += 1

                # Cosine similarity (sense tokens only)
                en_mask = en_preds[i, :en_len] != EN_LABEL2ID["O"]
                vi_mask = vi_preds[i, :vi_len] != VI_LABEL2ID["O"]

                if en_mask.any() and vi_mask.any():
                    en_vecs = en_hidden[i, :en_len][en_mask]   # (n_en, hidden)
                    vi_vecs = vi_hidden[i, :vi_len][vi_mask]   # (n_vi, hidden)

                    if en_vecs.size(0) > 0 and vi_vecs.size(0) > 0:
                        # Normalize
                        en_norm = F.normalize(en_vecs, dim=-1)
                        vi_norm = F.normalize(vi_vecs, dim=-1)

                        # Cosine similarity matrix: (n_en, n_vi)
                        cos_matrix = torch.matmul(en_norm, vi_norm.T)

                        total_cosine += cos_matrix.sum().item()
                        total_pairs += cos_matrix.numel()

            batch_en, batch_vi = [], []

# =======================
# FINAL METRICS
# =======================
EN_sense_ratio = sense_en / total_en if total_en > 0 else 0.0
VI_sense_ratio = sense_vi / total_vi if total_vi > 0 else 0.0
mean_cosine = total_cosine / total_pairs if total_pairs > 0 else 0.0

print("\n===== TEST RESULTS =====")
print(f"English sense ratio:     {EN_sense_ratio:.4f}")
print(f"Vietnamese sense ratio:  {VI_sense_ratio:.4f}")
print(f"Mean EN–VI cosine sim:   {mean_cosine:.4f}")