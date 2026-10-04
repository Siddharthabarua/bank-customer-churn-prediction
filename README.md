# Bank Churn Dashboard v3

This version assigns unique keys to every Streamlit widget, fixes the Geography churn-rate column issue, and trains Gradient Boosting on app startup instead of loading a serialized `.joblib` model.

## Folder layout
Keep these files together:
- `app.py`
- `requirements.txt`
- `European_Bank.csv`

If the dataset is in a `data` folder, the app checks `data/European_Bank.csv` too.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
