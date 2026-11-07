"""Generate HSI fine-tuning scripts from Tu's NEO templates.

Takes apsmc.py (combined-sample training) and apsm1.py (single-race training with
an external test set) and rewrites the label column, input data, and output paths
for each Health Status Inventory outcome.

Everything is written under project. Nothing outside that tree is touched, and
the templates themselves are only read.

Every substitution is asserted, so a template that has drifted fails loudly here
rather than silently producing a script that trains on the wrong column.
"""

import os
import sys

RIS_ROOT = "/path/to/project"
LOCAL_ROOT = "/path/to/project"

TEMPLATE_DIR = os.path.join(LOCAL_ROOT, "HSI_analysis", "roberta_templates")
SCRIPT_OUT = os.path.join(LOCAL_ROOT, "HSI_analysis", "roberta_scripts")

DATA_DIR = f"{RIS_ROOT}/HSI_analysis/model_data"
RUN_DIR = f"{RIS_ROOT}/roberta_runs"

# short code -> HSI column name, following Tu's <code><sample> filename convention
OUTCOMES = {
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

SUBSCALES = ["pf", "rp", "re", "ef", "ew", "sf", "pn", "gh"]
COMPOSITES = ["phc", "mhc"]

# sample code -> training data file.
#
# All three are built from apsmc.py, the current template. The deprecated
# apsm1.py is not used: its extra machinery was an external cross-race test set,
# which is handled separately by cross_race_predict.py against saved fold models
# rather than by retraining.
#
#   c  train on all 900, 5-fold CV, evaluate_by_race splits the test folds
#   w  train on the 450 White participants, 5-fold CV within that sample
#   b  train on the 450 Black participants, 5-fold CV within that sample
SAMPLES = {
    "c": "hsi_combined.json",
    "w": "hsi_white.json",
    "b": "hsi_black.json",
}


def sub(text, old, new, label, expected=1):
    """Replace and assert the pattern was actually present."""
    found = text.count(old)
    if found != expected:
        raise SystemExit(
            f"  FAILED [{label}]: expected {expected} occurrence(s), found {found}.\n"
            f"  Pattern: {old[:90]}"
        )
    return text.replace(old, new)


def strip_notebook_imports(text):
    """Drop Jupyter-only imports that crash a batch job.

    apsmc.py was developed in a notebook and imports IPython.display without
    ever using it. IPython is not in the cluster venv, so the import raises
    ModuleNotFoundError before any work starts.
    """
    return sub(
        text,
        "from IPython.display import display, Javascript\n",
        "",
        "IPython import",
    )


def rewrite_training_block(text, column, train_file):
    """Swap the NEO label column and input file for an HSI column."""
    text = sub(
        text,
        '"/path/to/project/Active/TD/SPAN/psm_combined.json"',
        f'"{DATA_DIR}/{train_file}"',
        "train data path",
    )

    text = sub(
        text,
        "train_data = train_data[train_data['PNEOA_scaled'] != '.']",
        f"LABEL_COL = {column!r}\ntrain_data = train_data.dropna(subset=[LABEL_COL])",
        "missing-value filter",
    )
    text = sub(
        text,
        "train_data['PNEOA_scaled_new'] = scaler.fit_transform("
        "train_data['PNEOA_scaled'].to_numpy().reshape(-1, 1))",
        "train_data['label_scaled'] = scaler.fit_transform("
        "train_data[LABEL_COL].astype(float).to_numpy().reshape(-1, 1))",
        "label scaling",
    )
    text = sub(
        text,
        "train_data_df = train_data[['PARTID', 'text', 'PNEOA_scaled_new']].rename(columns={\n"
        "    'PNEOA_scaled_new': 'labels'\n"
        "})",
        "train_data_df = train_data[['PARTID', 'text', 'label_scaled']].rename(columns={\n"
        "    'label_scaled': 'labels'\n"
        "})",
        "label column selection",
    )
    return text


def build(code, column, sample, template_text):
    """One script: apsmc.py with the label column, input file and output dir swapped."""
    train_file = SAMPLES[sample]
    stem = f"{code}psm{sample}"

    text = strip_notebook_imports(template_text)
    text = rewrite_training_block(text, column, train_file)
    text = sub(
        text,
        '"/path/to/project/Active/TD/SPAN/apsmc/Modelapsmc/"',
        f'"{RUN_DIR}/{stem}/Model{stem}/"',
        "output dir",
    )

    trained_on = {
        "c": "combined sample (n=900)",
        "w": "White sample (n=450)",
        "b": "Black sample (n=450)",
    }[sample]
    header = (
        f'"""HSI fine-tuning: {column}\n\n'
        f"Trained on the {trained_on}, 5-fold CV.\n"
        f"Generated from Tu's apsmc.py by make_hsi_scripts.py.\n"
        f"Data:    {DATA_DIR}/{train_file}\n"
        f'Outputs: {RUN_DIR}/{stem}/\n"""\n'
    )
    return stem, header + text


def main():
    which = sys.argv[1:] or list(OUTCOMES)
    unknown = [c for c in which if c not in OUTCOMES]
    if unknown:
        raise SystemExit(f"Unknown outcome code(s): {unknown}. Valid: {list(OUTCOMES)}")

    for path, label in [(TEMPLATE_DIR, "template dir"), (LOCAL_ROOT, "project")]:
        if not os.path.isdir(path):
            raise SystemExit(f"Cannot reach {label}: {path}\nIs the ACCOUNT share mounted?")

    template = open(os.path.join(TEMPLATE_DIR, "apsmc.py")).read()

    os.makedirs(SCRIPT_OUT, exist_ok=True)
    written = []

    for code in which:
        column = OUTCOMES[code]
        for sample in SAMPLES:
            stem, text = build(code, column, sample, template)
            path = os.path.join(SCRIPT_OUT, f"{stem}.py")
            with open(path, "w") as fh:
                fh.write(text)
            written.append((stem, column, sample))

    print(f"Generated {len(written)} scripts in:\n  {SCRIPT_OUT}\n")
    print(f"{'script':14s} {'sample':10s} HSI outcome")
    print("-" * 70)
    for stem, column, sample in written:
        kind = {"c": "combined", "w": "White only", "b": "Black only"}[sample]
        print(f"{stem + '.py':14s} {kind:12s} {column}")

    print(f"\nAll outputs write to {RUN_DIR}/ -- nothing outside project is touched.")


if __name__ == "__main__":
    main()
