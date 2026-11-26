# AI Stock Analyst — Streamlit Finance Workshop

An educational, interactive Streamlit dashboard for demonstrating AI/ML applications in finance. Students can enter a stock ticker (for example, `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, or `AAPL`) and explore market data, technical indicators, next-session direction classification, model evaluation, and headline sentiment.

> **Disclaimer:** This is a classroom demonstration, not investment advice or a trading strategy. ML predictions are uncertain. Historical performance is not a guarantee of future outcomes.

## Features

- Yahoo Finance market data via `yfinance`
- Historical price chart and 20-/50-session moving averages
- RSI (14), daily returns, 5-/20-session returns, annualized volatility
- Next-session Up/Down classification using:
  - Random Forest
  - Logistic Regression
  - XGBoost (optional; the app falls back if unavailable)
- Chronological train/test split (80% training, latest 20% test)
- Accuracy, precision, recall, F1, confusion matrix
- Random Forest feature importance
- Recent headline sentiment using VADER
- Learning Corner explaining features, labels, models, metrics, and limitations

## Run locally

1. Install Python 3.10 or newer (recommended).
2. Download `app.py` and `requirements.txt` into the same directory.
3. In a terminal, change to that directory and run:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL Streamlit prints (typically `http://localhost:8501`).

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and this `README.md` to the repository root.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Deploy. Streamlit Cloud installs packages from `requirements.txt` automatically.

No API key is needed for the basic app. Market/news access through Yahoo Finance is subject to availability and provider limits.

## Example ticker symbols

| Company / instrument | Symbol |
|---|---|
| Reliance Industries | `RELIANCE.NS` |
| TCS | `TCS.NS` |
| Infosys | `INFY.NS` |
| HDFC Bank | `HDFCBANK.NS` |
| ICICI Bank | `ICICIBANK.NS` |
| State Bank of India | `SBIN.NS` |
| Apple | `AAPL` |
| Microsoft | `MSFT` |
| NVIDIA | `NVDA` |
| Tesla | `TSLA` |

## Suggested classroom exercise

1. Start with `RELIANCE.NS` and review the price chart and indicators.
2. Compare Random Forest and Logistic Regression.
3. Discuss the test metrics and confusion matrix.
4. Explore feature importance.
5. Review headline sentiment and consider whether headline sentiment should be used as a predictive feature.
6. Change ticker and history length; discuss why market prediction is challenging and why an apparently high metric can be misleading.

## Technical notes / limitations

- The target is whether the following session's close is higher than the current session's close. It does **not** predict the exact future price.
- Feature rows are based on data available at that session's close; the target is the next-session direction.
- The model uses a simple chronological split and is a teaching example, not a rigorous walk-forward study.
- Trading signals are illustrative rules only and do not account for costs, slippage, liquidity, taxes, or investor-specific objectives.
- RSI and sentiment models have limitations; VADER is general-purpose and may not interpret financial language reliably.
- Yahoo Finance may occasionally return incomplete data, change its interface, or limit requests.

## License

This repository is made publicly available for educational, academic, and classroom use. Permission is granted to access and use the code for learning, teaching, and academic demonstration purposes. Redistribution, commercial use, modification for public distribution, or incorporation into other projects requires prior written permission from the copyright holder, unless otherwise specified. This repository is provided "as is", without warranty of any kind. The author assumes no responsibility for any loss or damage arising from its use. For other permissions or licensing inquiries, please contact the author.
