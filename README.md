# SpamGuard AI — ML Spam Detection Dashboard

A neon-themed Streamlit web app that classifies SMS/email text as **spam** or **ham** using TF-IDF + Logistic Regression.

## Features
- Interactive message detection with confidence score
- CSV batch predictions and downloadable results
- Dataset label analytics
- Optional 80/20 train/test evaluation (accuracy, precision, recall, F1)
- Dark neon-glass UI
- Runs locally; no external prediction API needed

## Run on Windows (PowerShell)

Open PowerShell in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, skip activation and use:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If you have multiple Python installations, use the same Python executable for installation and launch, for example:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The browser should open automatically. If not, open the `Local URL` printed in the terminal, usually `http://localhost:8501`.

## CSV batch format

Upload a CSV with a `text` or `message` column:

```csv
text
"Congratulations, you won a prize. Claim now!"
"Hey, see you at the library at 3."
```

## Model details
- Vectorizer: `TfidfVectorizer(ngram_range=(1, 2))`
- Classifier: `LogisticRegression(class_weight="balanced")`
- Labels: `spam` and `ham`

## Important
The included CSV is a small illustrative dataset to make the project run immediately. Metrics may be unstable and the model can make mistakes. For a serious report, use a larger, documented dataset such as the UCI SMS Spam Collection, follow its terms, and evaluate on a held-out test set. Do not use the confidence score as a guarantee of safety.
