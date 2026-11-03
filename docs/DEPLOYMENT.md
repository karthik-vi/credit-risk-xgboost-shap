# Deployment

## Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_eval.py --out results/metrics.json     # downloads data, ~minutes on 4 CPUs; writes docs/figures/*.png
pytest -q
```

## Docker
```bash
docker build -t credit . && docker run --rm credit          # runs tests
docker run --rm -v $PWD/results:/app/results credit python scripts/train_eval.py
```

## Not production-ready
- Trained on 700 rows of 1990s German data; it is a method demonstration, not a usable credit model.
- No reject inference, no population-stability monitoring, no champion/challenger process, no override logging.
- Reason codes are raw feature names; a real adverse-action notice needs reviewed, customer-readable wording.
