"""Can race be decoded from the narrative embeddings?

The directional bias result -- the model predicting lower health for Black
participants at equivalent self-reported health -- has two competing
explanations that the main analyses cannot separate:

  (a) narrative content genuinely differs between groups
  (b) the model has learned race-correlated linguistic features and applies
      them as a proxy for group membership

This tests whether (b) is even mechanically available. It trains a regularized
logistic regression to predict race from the same RoBERTa embeddings the
feed-forward network uses, and reports cross-validated AUC.

  AUC near .50   race is not recoverable from the narratives; proxy
                 discrimination is implausible and (a) is favoured
  AUC high       race is strongly encoded in how people write; the model COULD
                 be using it, though this does not establish that it does

The embeddings come from the held-out test folds, so each participant's
embedding was produced by a model that never saw them during fine-tuning.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hsidata import LIWC_PATH, RACE_MAP, ensure_out_dir
from rq2_results import RUNS_DIR

N_FOLDS = 5
N_PERMUTATIONS = 1000
SEED = 42


def load_embeddings(stem):
    """Participant-level embeddings from the held-out test folds, with PARTIDs.

    Each pickle is a list over participants of a list over sliding windows; the
    feed-forward network averages the windows, so we do the same. Row order
    within a fold matches the test rows recorded in fold_partitions.csv.
    """
    model_dir = os.path.join(RUNS_DIR, stem, f"Model{stem}")
    partitions_path = os.path.join(model_dir, "fold_partitions.csv")
    if not os.path.exists(partitions_path):
        raise SystemExit(f"Missing {partitions_path}")
    partitions = pd.read_csv(partitions_path)

    vectors, partids = [], []
    for fold in range(1, N_FOLDS + 1):
        path = os.path.join(model_dir, "embeddings", f"test_embeddings_fold{fold}.pkl")
        if not os.path.exists(path):
            print(f"  fold {fold}: no embeddings, skipped")
            continue
        with open(path, "rb") as fh:
            per_participant = pickle.load(fh)

        ids = partitions[(partitions["fold"] == fold) & (partitions["set"] == "test")]["PARTID"].tolist()
        if len(ids) != len(per_participant):
            raise SystemExit(
                f"fold {fold}: {len(per_participant)} embeddings but {len(ids)} test PARTIDs. "
                "Row order cannot be trusted; aborting rather than guessing."
            )
        for pid, windows in zip(ids, per_participant):
            vectors.append(np.mean(np.asarray(windows), axis=0))
            partids.append(pid)

    return np.vstack(vectors), pd.Series(partids, name="PARTID")


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "phcpsmc"
    out_dir = ensure_out_dir()
    if not os.path.isdir(RUNS_DIR):
        raise SystemExit(f"Cannot reach {RUNS_DIR}\nIs the ACCOUNT share mounted?")

    print(f"Loading embeddings from {stem}...")
    X, partids = load_embeddings(stem)
    print(f"  {X.shape[0]} participants x {X.shape[1]} dimensions")

    race = pd.read_csv(LIWC_PATH, low_memory=False, usecols=["PARTID", "PRACE"])
    race["race_name"] = race["PRACE"].map(RACE_MAP)
    merged = pd.DataFrame({"PARTID": partids}).merge(race, on="PARTID", how="left")
    keep = merged["race_name"].isin(["White", "Black"]).to_numpy()
    X, y = X[keep], (merged.loc[keep, "race_name"] == "Black").astype(int).to_numpy()
    print(f"  {len(y)} with race: {(y == 0).sum()} White, {(y == 1).sum()} Black")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, probs)
    acc = accuracy_score(y, (probs > 0.5).astype(int))

    print("\nPermutation test on AUC...")
    rng = np.random.default_rng(SEED)
    null = []
    for i in range(N_PERMUTATIONS):
        shuffled = rng.permutation(y)
        p = cross_val_predict(model, X, shuffled, cv=cv, method="predict_proba")[:, 1]
        null.append(roc_auc_score(shuffled, p))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{N_PERMUTATIONS}")
    null = np.array(null)
    p_perm = (np.sum(null >= auc) + 1) / (N_PERMUTATIONS + 1)

    print("\n" + "=" * 72)
    print("RACE DECODABILITY FROM NARRATIVE EMBEDDINGS")
    print("=" * 72)
    print(f"  cross-validated AUC   {auc:.3f}")
    print(f"  accuracy              {acc:.3f}  (chance = {max((y==0).mean(), (y==1).mean()):.3f})")
    print(f"  permutation p         {p_perm:.4f}   (null AUC mean {null.mean():.3f})")
    print()
    if auc < 0.60:
        verdict = ("Race is weakly encoded in the narratives. Proxy discrimination is a\n"
                   "  less plausible account of the directional bias.")
    elif auc < 0.75:
        verdict = ("Race is moderately recoverable from the narratives. The model could be\n"
                   "  drawing on race-correlated language, though this does not show that it does.")
    else:
        verdict = ("Race is strongly encoded in the narratives. Race-correlated linguistic\n"
                   "  features are readily available to the model as a proxy.")
    print("  " + verdict)

    pd.DataFrame([{
        "run": stem, "n": len(y), "n_White": int((y == 0).sum()), "n_Black": int((y == 1).sum()),
        "auc": round(auc, 3), "accuracy": round(acc, 3),
        "permutation_p": p_perm, "null_auc_mean": round(null.mean(), 3),
    }]).to_csv(f"{out_dir}/race_decodability.csv", index=False)
    print(f"\nSaved: {out_dir}/race_decodability.csv")


if __name__ == "__main__":
    main()
