"""Cross-race transfer: apply a model trained on one race to the other sample.

Fills the two cells the training runs cannot produce -- White-trained predicting
Black participants, and Black-trained predicting White participants. This is the
comparison Tu's FFM paper reports ("the White training-Black test combination").

No retraining. For each of the five folds it reloads the saved fine-tuned RoBERTa
and its feed-forward head, then runs the other race's narratives through the same
pipeline the training script used: 500-word sliding windows with 100-word
overlap, CLS embeddings from hidden layer 23, averaged across windows, into the
FFN.

True labels are scaled with the SOURCE run's saved scaler, not refit on the
target sample, so predictions and observed values stay on one metric.

Run on a GPU node. It is inference only -- expect a few minutes per transfer.

Transfers that already have all five fold files are skipped, so a run that was
cut short can simply be repeated. Pass --force to redo them anyway.

    python cross_race_predict.py phcpsmw          one transfer
    python cross_race_predict.py phcpsmw phcpsmb  several
    python cross_race_predict.py --all            every completed w/b run
    python cross_race_predict.py --all --force    redo even finished ones
"""

import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RQ = "/path/to/project"
RUNS_DIR = f"{RQ}/roberta_runs"
DATA_DIR = f"{RQ}/HSI_analysis/model_data"

WINDOW_SIZE = 500
OVERLAP = 100
LAYER_NUM = 23
EMBED_DIM = 1024
HIDDEN1, HIDDEN2 = 256, 128
N_FOLDS = 5

# suffix of the source run -> (race it was trained on, race we transfer to, target file)
DIRECTION = {
    "w": ("White", "Black", "hsi_black.json"),
    "b": ("Black", "White", "hsi_white.json"),
}

OUTCOME_BY_CODE = {
    "pf": "Physical functioning",
    "rp": "Role limitations due to physical health",
    "re": "Role limitations due to emotional problems",
    "ef": "Energy/fatigue",
    "ew": "Emotional well-being",
    "sf": "Social Functioning",
    "pn": "Pain",
    "gh": "General health",
    "phc": "Physical Health Composite",
    "mhc": "Mental Health Composite",
}


class FeedforwardNN(nn.Module):
    """Must match the architecture in the training script exactly."""

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)
        out = self.dropout2(out)
        return self.fc3(out)


def sliding_windows(text, size=WINDOW_SIZE, overlap=OVERLAP):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), overlap)]


def participant_embedding(model, tokenizer, windows, device):
    """Mean of the CLS embeddings across a participant's windows."""
    vectors = []
    for window in windows:
        inputs = tokenizer(window, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        cls = outputs.hidden_states[LAYER_NUM][:, 0, :].detach().cpu().numpy()
        vectors.append(cls.squeeze())
    return np.mean(np.asarray(vectors), axis=0)


def parse(stem):
    """'phcpsmw' -> ('phc', 'w'). Raises on anything else."""
    if not stem.endswith(("psmw", "psmb")):
        raise SystemExit(f"'{stem}' is not a single-race run (expected a name ending psmw or psmb).")
    return stem[:-4], stem[-1]


def already_done(stem):
    """True when all five fold files exist for this transfer."""
    code, suffix = parse(stem)
    _, target_race, _ = DIRECTION[suffix]
    out_dir = os.path.join(RUNS_DIR, stem, f"Model{stem}", f"crossrace_{target_race.lower()}")
    return len(glob.glob(os.path.join(out_dir, "crossrace_test_predictions_fold*.csv"))) >= N_FOLDS


def transfer(stem, device):
    code, suffix = parse(stem)
    trained_on, target_race, target_file = DIRECTION[suffix]
    outcome = OUTCOME_BY_CODE[code]

    model_dir = os.path.join(RUNS_DIR, stem, f"Model{stem}")
    if not os.path.isdir(model_dir):
        raise SystemExit(f"No run directory: {model_dir}")

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise SystemExit(f"No scaler at {scaler_path} -- did that run finish?")
    scaler = joblib.load(scaler_path)

    target = pd.read_json(os.path.join(DATA_DIR, target_file), lines=True)
    target = target.dropna(subset=[outcome])
    target["text"] = target["text"].astype(str).str.strip()
    target = target[target["text"] != ""].reset_index(drop=True)
    true_scaled = scaler.transform(
        target[outcome].astype(float).to_numpy().reshape(-1, 1)
    ).flatten()
    windows = [sliding_windows(t) for t in target["text"]]

    out_dir = os.path.join(model_dir, f"crossrace_{target_race.lower()}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"{stem}: trained on {trained_on}, predicting {len(target)} {target_race} participants")
    print(f"outcome: {outcome}")
    print(f"{'=' * 70}")

    written = []
    for fold in range(1, N_FOLDS + 1):
        rb_dir = os.path.join(model_dir, f"fold_{fold}", "CV_A")
        ffn_path = os.path.join(model_dir, "ffn_models", f"feedforward_nn_model_fold{fold}.pth")
        if not os.path.isdir(rb_dir) or not os.path.exists(ffn_path):
            print(f"  fold {fold}: missing artifacts, skipped")
            continue

        model = AutoModelForSequenceClassification.from_pretrained(rb_dir).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(rb_dir)

        embeddings = np.vstack([
            participant_embedding(model, tokenizer, w, device) for w in windows
        ])

        ffn = FeedforwardNN(EMBED_DIM, HIDDEN1, HIDDEN2, 1).to(device)
        ffn.load_state_dict(torch.load(ffn_path, map_location=device))
        ffn.eval()
        with torch.no_grad():
            preds = ffn(torch.tensor(embeddings, dtype=torch.float32).to(device))
        preds = preds.cpu().numpy().flatten()

        frame = pd.DataFrame({
            "PARTID": target["PARTID"].values,
            "true_label": true_scaled,
            "predicted_label": preds,
        })
        path = os.path.join(out_dir, f"crossrace_test_predictions_fold{fold}.csv")
        frame.to_csv(path, index=False)
        written.append(path)

        r = np.corrcoef(frame["true_label"], frame["predicted_label"])[0, 1]
        print(f"  fold {fold}: r = {r:.3f}  (n = {len(frame)})")

        del model, ffn
        torch.cuda.empty_cache()

    if written:
        print(f"  wrote {len(written)} fold files to {out_dir}/")
    return written


def main():
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    if not args:
        raise SystemExit(__doc__)

    if args == ["--all"]:
        stems = sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(RUNS_DIR, "*psm[wb]"))
            if os.path.isdir(p)
        )
        if not stems:
            raise SystemExit(f"No single-race runs found under {RUNS_DIR}")
        print(f"Found {len(stems)} single-race run(s).")
    else:
        stems = args

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible. Run this inside a GPU job.")
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")

    if not force:
        finished = [s for s in stems if already_done(s)]
        stems = [s for s in stems if s not in finished]
        if finished:
            print(f"skipping {len(finished)} already complete: {', '.join(finished)}")

    if not stems:
        print("Nothing left to do. Pass --force to redo completed transfers.")
        return

    print(f"{len(stems)} transfer(s) to run.")
    for i, stem in enumerate(stems, 1):
        print(f"\n[{i}/{len(stems)}]", end="")
        transfer(stem, device)

    print("\nDone. Analyse with training_set_comparison.py.")


if __name__ == "__main__":
    main()
