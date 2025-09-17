```markdown
# subtom_qc

Subtomogram quality-control and classifier pipeline. Small, practical PyTorch code for: simulated-data pretraining (contrastive), classifier finetune, scoring, and simple evaluation.

## Quickstart (local GPU)

1. Create a venv and install requirements:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

2. Run the notebook `notebooks/train_on_simulated_data.ipynb` or run scripts in `src/`.

3. Generate simulated data:

```bash
python src/simulate.py --out-dir data/sim --n-usable 500 --n-junk 500
```

4. Pretrain encoder (contrastive):

```bash
python src/train_pretrain.py --data-dir data/sim --out models/pretrained.pth
```

5. Train classifier:

```bash
python src/train_classifier.py --data-dir data/sim --pretrained models/pretrained.pth --out models/clf.pth
```

6. Score a folder of subtomograms:

```bash
python src/score_subtomos.py --model models/clf.pth --encoder models/pretrained.pth --folder /path/to/mrcs --out scores.csv
```

See notebooks for a guided run.
```