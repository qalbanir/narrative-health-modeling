"""Pre-flight check for the HSI RoBERTa jobs on RIS.

Run this in an INTERACTIVE GPU job before submitting anything in batch. It walks
the same path the real scripts take -- imports, GPU, data, label column, sliding
windows, and a miniature fine-tune -- and stops at the first thing that breaks,
so an environment problem costs a minute instead of a queued GPU hour.

Nothing is written outside project, and the throwaway model goes to a temp dir
that is deleted on the way out.

    python smoke_test.py                 # default outcome
    python smoke_test.py "Pain"          # any HSI column
"""

import os
import shutil
import sys
import tempfile
import traceback

RIS_ROOT = "/path/to/project"
DATA_DIR = f"{RIS_ROOT}/HSI_analysis/model_data"
DEFAULT_LABEL = "Physical Health Composite"

CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def say(msg, indent=2):
    print(" " * indent + msg, flush=True)


@check("Python and core packages import")
def _imports(state):
    import numpy, pandas, sklearn, scipy
    say(f"python      {sys.version.split()[0]}")
    say(f"numpy       {numpy.__version__}")
    say(f"pandas      {pandas.__version__}")
    say(f"scikit-learn {sklearn.__version__}")
    say(f"scipy       {scipy.__version__}")
    import contractions  # noqa: F401
    say("contractions ok")


@check("PyTorch sees a GPU")
def _gpu(state):
    import torch

    say(f"torch       {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. You are on a CPU node or the "
            "container has no CUDA. Request a GPU in the bsub line."
        )
    say(f"CUDA        {torch.version.cuda}")
    say(f"device      {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    say(f"GPU memory  {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")
    if total < 20e9:
        say("WARNING: roberta-large at batch 16 x 512 tokens usually needs ~24 GB.")
    state["torch"] = torch


@check("simpletransformers imports")
def _simpletransformers(state):
    from simpletransformers.classification import ClassificationArgs, ClassificationModel  # noqa: F401
    import transformers

    # simpletransformers does not expose __version__, so ask the package metadata.
    try:
        from importlib.metadata import version

        say(f"simpletransformers {version('simpletransformers')}")
    except Exception:
        say("simpletransformers imported (version not reported)")
    say(f"transformers       {transformers.__version__}")
    state["ready"] = True


@check("Data file loads with the expected columns")
def _data(state):
    import pandas as pd

    path = os.path.join(DATA_DIR, "hsi_combined.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}\nCheck the path, and mind the space in 'project'.")
    df = pd.read_json(path, lines=True)
    say(f"rows        {len(df)}")
    missing = [c for c in ["PARTID", "text", "PRACE", state["label"]] if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}\nPresent: {list(df.columns)[:14]}")
    say(f"columns ok  PARTID, text, PRACE, {state['label']!r}")
    say(f"PRACE       {df['PRACE'].value_counts().to_dict()}")
    state["df"] = df


@check("Label column is usable")
def _label(state):
    import pandas as pd

    col = state["df"][state["label"]]
    valid = pd.to_numeric(col, errors="coerce").dropna()
    say(f"non-missing {len(valid)} of {len(col)}")
    say(f"range       {valid.min():.1f} to {valid.max():.1f}, mean {valid.mean():.1f}")
    if len(valid) < 100:
        raise ValueError("Fewer than 100 usable labels -- wrong column?")


@check("Text field and sliding windows")
def _text(state):
    df = state["df"]
    words = df["text"].astype(str).str.split().str.len()
    say(f"words       median {words.median():.0f}, max {words.max():.0f}")
    if (words < 5).any():
        say(f"WARNING: {(words < 5).sum()} narrative(s) under 5 words")

    def windows(text, size=500, overlap=100):
        parts = text.split()
        return [" ".join(parts[i:i + size]) for i in range(0, len(parts), overlap)]

    sample = windows(df["text"].iloc[0])
    say(f"windows     {len(sample)} from the first narrative ({words.iloc[0]} words)")


@check("Miniature fine-tune runs end to end")
def _tiny_train(state):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from simpletransformers.classification import ClassificationArgs, ClassificationModel

    df = state["df"].copy()
    df[state["label"]] = pd.to_numeric(df[state["label"]], errors="coerce")
    df = df.dropna(subset=[state["label"]]).head(24)
    df["labels"] = StandardScaler().fit_transform(
        df[state["label"]].to_numpy().reshape(-1, 1)
    ).flatten()
    train = df[["text", "labels"]].reset_index(drop=True)

    tmp = tempfile.mkdtemp(prefix="hsi_smoke_", dir=os.path.expanduser("~"))
    state["tmp"] = tmp
    say(f"scratch     {tmp}")
    say("downloading roberta-large if not cached -- this can take a few minutes...")

    args = ClassificationArgs(
        output_dir=os.path.join(tmp, "out"),
        cache_dir=os.path.join(tmp, "cache"),
        num_train_epochs=1,
        train_batch_size=2,
        eval_batch_size=2,
        max_seq_length=128,      # deliberately small; the real runs use 512
        sliding_window=True,
        regression=True,
        no_cache=True,
        overwrite_output_dir=True,
        save_model_every_epoch=False,
        save_eval_checkpoints=False,
        save_steps=-1,
        silent=True,
        use_multiprocessing=False,
        use_multiprocessing_for_evaluation=False,
        dataloader_num_workers=0,
    )
    model = ClassificationModel("roberta", "roberta-large", num_labels=1,
                                args=args, use_cuda=True)
    say("model loaded onto GPU")
    model.train_model(train)
    say(f"trained on {len(train)} rows for 1 epoch")
    preds, _ = model.predict(train["text"].tolist()[:2])
    say(f"predicted   {[round(float(p), 3) for p in preds]}")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LABEL
    state = {"label": label, "tmp": None}

    print("=" * 66)
    print("HSI RoBERTa smoke test")
    print(f"outcome: {label}")
    print("=" * 66)

    passed = 0
    try:
        for i, (name, fn) in enumerate(CHECKS, 1):
            print(f"\n[{i}/{len(CHECKS)}] {name}")
            try:
                fn(state)
            except Exception:
                print("  FAILED\n")
                traceback.print_exc()
                print("\n" + "=" * 66)
                print(f"STOPPED at check {i}: {name}")
                print(f"{passed} of {len(CHECKS)} checks passed. Fix this before submitting.")
                print("=" * 66)
                return 1
            say("PASS")
            passed += 1
    finally:
        if state.get("tmp") and os.path.isdir(state["tmp"]):
            shutil.rmtree(state["tmp"], ignore_errors=True)
            print(f"\ncleaned up {state['tmp']}")

    print("\n" + "=" * 66)
    print(f"ALL {passed} CHECKS PASSED")
    print("The environment, data, and training loop all work. Safe to submit")
    print("phcpsmc.py as a batch job.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
