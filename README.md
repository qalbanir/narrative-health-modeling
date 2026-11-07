# Language-Based AI Modeling of Health Status: Racial Bias Analyses

Analysis code for *Can Language-Based AI Predict Health Status from Life Narratives
Equally for Black and White Older Adults?*

Sample: 900 community-dwelling older adults (450 Black, 450 White), propensity-matched
on education and annual household income. Outcomes: the eight RAND-36 subscales plus the
Physical and Mental Health Composites. Predictors: LIWC-22 categories and fine-tuned
RoBERTa-large embeddings of life narrative interviews.

## Data availability

**No participant data are included in this repository.** The life narrative interviews
contain potentially identifying information and are not publicly available. The scored
RAND-36 data are deposited separately (see the manuscript's Transparency and Openness
statement). The `output/` directory contains only aggregate results.

## Paths

Cluster paths in these scripts have been replaced with `/path/to/project`. Set them to
your own locations before running. Local scripts read source data from `~/Downloads`
(configured at the top of `hsidata.py`) and write to `output/`.

## Local analyses

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python hsidata.py
```

`hsidata.py` is the shared loader: RAND-36 item assignments, subscale and composite
definitions, the race map, and the correlation and reliability helpers every other
script imports. Start there.

## Fine-tuning (GPU)

`requirements-cluster.txt` lists the fine-tuning environment (Python 3.9, CUDA 12.8).
`submit_hsi.sbatch` and `submit_crossrace.sbatch` are Slurm job scripts; set
`--account`, `--qos`, and `--partition` for your own cluster.

## Script map

| Stage | Scripts |
|---|---|
| Data loading | `hsidata.py`, `prep_roberta_data.py` |
| Descriptives | `demotable.py`, `results_tables.py`, `hsireliability.py` |
| LIWC | `liwc_full_sweep.py`, `hsiliwccorrelation.py`, `liwc_table_by_race.py`, `table3.py`, `supplemental_table1.py` |
| Fine-tuning | `make_hsi_scripts.py`, `finetune.py`, `smoke_test.py` |
| Model results | `rq2_results.py`, `roberta_table.py`, `training_set_comparison.py`, `cross_race_predict.py` |
| Bias analyses | `bias_analyses.py`, `bias_summary_table.py`, `bias_tables_per_outcome.py`, `biasac.py`, `calibration_slope_difference.py`, `racedifftest.py` |
| Mechanism | `race_decodability.py`, `feature_overlap.py` |
| Other | `intersectional.py`, `figures.py` |
