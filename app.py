# AI Stock Analyst - Educational Streamlit Application
# Run locally: streamlit run app.py
#
# Install dependencies:
# pip install -r requirements.txt
#
# Note: This application is for educational purposes only. It is not financial advice.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="AI Stock Analyst",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI Stock Analyst")
st.caption(
    "An interactive educational dashboard combining market data, technical "
    "indicators, risk analysis, machine learning and news sentiment."
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 10px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

st.sidebar.header("Analysis settings")

ticker = st.sidebar.text_input(
    "Stock ticker",
    value="RELIANCE.NS",
    help="Examples: RELIANCE.NS, TCS.NS, INFY.NS, AAPL, MSFT",
).strip().upper()

period = st.sidebar.selectbox(
    "Historical data period",
    ["6mo", "1y", "2y", "5y", "10y"],
    index=2,
)

interval = st.sidebar.selectbox(
    "Data interval",
    ["1d", "1wk", "1mo"],
    index=0,
)

model_choice = st.sidebar.selectbox(
    "Machine-learning model",
    ["Random Forest", "Logistic Regression"],
)

test_fraction = st.sidebar.slider(
    "Test data proportion",
    min_value=0.15,
    max_value=0.40,
    value=0.25,
    step=0.05,
)

st.sidebar.info(
    "Indian NSE tickers usually use the suffix .NS. "
    "For example, Reliance Industries is RELIANCE.NS."
)

# ---------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def download_stock_data(symbol, selected_period, selected_interval):
    data = yf.download(
        symbol,
        period=selected_period,
        interval=selected_interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    # yfinance may return MultiIndex columns for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.copy()
    data.columns = [str(c).title() for c in data.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        return pd.DataFrame()

    data = data[required].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    return data


with st.spinner(f"Downloading market data for {ticker or 'the selected ticker'}..."):
    raw = download_stock_data(ticker, period, interval)

if not ticker:
    st.warning("Enter a valid stock ticker in the sidebar.")
    st.stop()

if raw.empty or len(raw) < 60:
    st.error(
        "No usable data was returned, or the available history is too short. "
        "Check the ticker symbol, internet connection and selected period."
    )
    st.stop()

# ---------------------------------------------------------------------
# Technical indicators and feature engineering
# ---------------------------------------------------------------------

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_indicators(data):
    df = data.copy()

    df["Return"] = df["Close"].pct_change()
    df["DailyReturnPct"] = df["Return"] * 100

    for window in [5, 10, 20, 50, 100, 200]:
        if len(df) >= window:
            df[f"SMA_{window}"] = df["Close"].rolling(window).mean()

    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    df["RSI"] = calculate_rsi(df["Close"], 14)

    middle = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Middle"] = middle
    df["BB_Upper"] = middle + 2 * std
    df["BB_Lower"] = middle - 2 * std

    previous_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR_14"] = true_range.rolling(14).mean()

    df["Volatility_20"] = df["Return"].rolling(20).std() * np.sqrt(252) * 100
    df["RollingHigh_20"] = df["High"].rolling(20).max()
    df["RollingLow_20"] = df["Low"].rolling(20).min()

    # Machine-learning predictors. The target is next-period direction.
    df["Return_1"] = df["Return"]
    df["Return_2"] = df["Close"].pct_change(2)
    df["Return_5"] = df["Close"].pct_change(5)
    df["Return_10"] = df["Close"].pct_change(10)
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Price_SMA20_Ratio"] = df["Close"] / df["Close"].rolling(20).mean() - 1
    df["Price_SMA50_Ratio"] = df["Close"] / df["Close"].rolling(50).mean() - 1
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df


df = build_indicators(raw)

# ---------------------------------------------------------------------
# Company information
# ---------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_info(symbol):
    try:
        info = yf.Ticker(symbol).get_info()
        if not isinstance(info, dict):
            return {}
        return info
    except Exception:
        return {}


with st.expander("Company information", expanded=False):
    info = get_company_info(ticker)
    if info:
        c1, c2, c3 = st.columns(3)
        c1.write(f"**Company:** {info.get('longName', info.get('shortName', ticker))}")
        c2.write(f"**Sector:** {info.get('sector', 'Not available')}")
        c3.write(f"**Industry:** {info.get('industry', 'Not available')}")
        summary = info.get("longBusinessSummary")
        if summary:
            st.write(summary)
    else:
        st.write("Company profile information is not available for this ticker.")

# ---------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------

latest = df.iloc[-1]
previous = df["Close"].iloc[-2]
latest_price = float(latest["Close"])
latest_change = (latest_price / float(previous) - 1) * 100

st.subheader(f"{ticker} — Market overview")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Latest close", f"{latest_price:,.2f}", f"{latest_change:+.2f}%")
m2.metric("RSI (14)", f"{latest['RSI']:.1f}" if pd.notna(latest["RSI"]) else "N/A")
m3.metric(
    "20-day volatility",
    f"{latest['Volatility_20']:.1f}%" if pd.notna(latest["Volatility_20"]) else "N/A",
)
m4.metric(
    "20-day SMA",
    f"{latest['SMA_20']:,.2f}" if "SMA_20" in df and pd.notna(latest["SMA_20"]) else "N/A",
)
m5.metric("Latest volume", f"{latest['Volume']:,.0f}")

st.caption(
    f"Data range: {df.index.min().date()} to {df.index.max().date()} "
    f"· {len(df):,} observations · Adjusted prices"
)

# ---------------------------------------------------------------------
# Price and volume charts
# ---------------------------------------------------------------------

st.subheader("1. Historical price and moving averages")

price_fig = go.Figure()
price_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name="Close",
        line=dict(width=2),
    )
)

for col, label in [
    ("SMA_20", "20-day SMA"),
    ("SMA_50", "50-day SMA"),
    ("SMA_200", "200-day SMA"),
]:
    if col in df.columns:
        price_fig.add_trace(
            go.Scatter(x=df.index, y=df[col], mode="lines", name=label)
        )

price_fig.update_layout(
    title=f"{ticker}: adjusted closing price",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified",
    height=480,
    legend_title="Series",
)
st.plotly_chart(price_fig, use_container_width=True)

volume_fig = px.bar(
    df,
    x=df.index,
    y="Volume",
    title="Trading volume",
    labels={"x": "Date", "Volume": "Volume"},
)
volume_fig.update_layout(height=300)
st.plotly_chart(volume_fig, use_container_width=True)

# ---------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------

st.subheader("2. Technical indicators")

left, right = st.columns(2)

with left:
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI (14)"))
    rsi_fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought reference (70)")
    rsi_fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold reference (30)")
    rsi_fig.update_layout(
        title="Relative Strength Index (RSI)",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        height=350,
    )
    st.plotly_chart(rsi_fig, use_container_width=True)

with right:
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal"))
    macd_fig.add_trace(
        go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", opacity=0.35)
    )
    macd_fig.update_layout(title="MACD (12, 26, 9)", height=350)
    st.plotly_chart(macd_fig, use_container_width=True)

bb_fig = go.Figure()
bb_fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="Upper band", line=dict(dash="dot")))
bb_fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="Lower band", line=dict(dash="dot"), fill="tonexty"))
bb_fig.add_trace(go.Scatter(x=df.index, y=df["BB_Middle"], name="20-day average"))
bb_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
bb_fig.update_layout(title="Bollinger Bands (20-day, 2 standard deviations)", height=380)
st.plotly_chart(bb_fig, use_container_width=True)

# ---------------------------------------------------------------------
# Returns and risk analysis
# ---------------------------------------------------------------------

st.subheader("3. Returns and risk analysis")

returns = df["Return"].dropna()
if len(returns) > 1:
    annualized_volatility = returns.std() * np.sqrt(252) * 100
    annualized_return = ((1 + returns.mean()) ** 252 - 1) * 100
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max - 1) * 100
    max_drawdown = float(drawdown.min())
    var_95 = float(np.percentile(returns, 5) * 100)
    cvar_95 = float(returns[returns <= np.percentile(returns, 5)].mean() * 100)
    daily_risk_free = 0.06 / 252
    sharpe = (
        ((returns.mean() - daily_risk_free) / returns.std()) * np.sqrt(252)
        if returns.std() != 0
        else np.nan
    )

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Annualized volatility", f"{annualized_volatility:.2f}%")
    r2.metric("Annualized return*", f"{annualized_return:.2f}%")
    r3.metric("Sharpe ratio*", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
    r4.metric("Maximum drawdown", f"{max_drawdown:.2f}%")
    r5.metric("Historical VaR (95%)", f"{var_95:.2f}%")

    st.caption(
        "*Annualized return is a simple approximation based on the mean periodic "
        "return; the Sharpe ratio assumes a 6% annual risk-free rate and 252 "
        "trading days. VaR/CVaR are historical estimates, not guarantees."
    )

    hist_fig = px.histogram(
        x=returns * 100,
        nbins=50,
        title="Distribution of periodic returns",
        labels={"x": "Return (%)", "y": "Frequency"},
    )
    hist_fig.update_layout(height=330)
    st.plotly_chart(hist_fig, use_container_width=True)

    dd_fig = px.area(
        x=drawdown.index,
        y=drawdown.values,
        title="Drawdown over time",
        labels={"x": "Date", "y": "Drawdown (%)"},
    )
    dd_fig.update_layout(height=330)
    st.plotly_chart(dd_fig, use_container_width=True)

    st.metric("Conditional VaR (95%)", f"{cvar_95:.2f}%")
else:
    st.info("Not enough return observations for risk analysis.")

# ---------------------------------------------------------------------
# Monthly return calendar heatmap
# Uses PeriodIndex rather than resample('ME'), avoiding pandas frequency
# alias differences between older and newer pandas versions.
# ---------------------------------------------------------------------

st.subheader("4. Monthly returns heatmap")

monthly = (
    df["Close"]
    .groupby(df.index.to_period("M"))
    .last()
    .pct_change()
    .dropna()
    * 100
)

if not monthly.empty:
    cal = monthly.to_frame("MonthlyReturn")
    cal["Year"] = cal.index.year
    cal["Month"] = cal.index.month

    wide = (
        cal.pivot(index="Year", columns="Month", values="MonthlyReturn")
        .reindex(columns=range(1, 13))
    )

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    wide.columns = month_names

    heatmap_fig = px.imshow(
        wide,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Monthly return calendar heatmap (%)",
        labels={"x": "Month", "y": "Year", "color": "Return %"},
    )
    heatmap_fig.update_layout(height=max(300, 55 * len(wide.index)))
    st.plotly_chart(heatmap_fig, use_container_width=True)
else:
    st.info("There are not enough monthly observations to create the heatmap.")

# ---------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------

st.subheader("5. Feature correlation")

corr_cols = [
    c for c in [
        "Return", "Return_2", "Return_5", "Return_10",
        "Volume_Change", "RSI", "MACD", "ATR_14",
        "Volatility_20", "Price_SMA20_Ratio", "Price_SMA50_Ratio",
    ]
    if c in df.columns
]

corr = df[corr_cols].replace([np.inf, -np.inf], np.nan).dropna().corr()
corr_fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlation matrix of selected indicators",
)
corr_fig.update_layout(height=600)
st.plotly_chart(corr_fig, use_container_width=True)

# ---------------------------------------------------------------------
# Machine-learning direction prediction
# ---------------------------------------------------------------------

st.subheader("6. Machine-learning direction prediction")

st.write(
    "The model estimates whether the next observed period's closing price "
    "will be higher or lower than the current close. It does not predict an "
    "exact future price."
)

feature_cols = [
    "Return_1",
    "Return_2",
    "Return_5",
    "Return_10",
    "Volume_Change",
    "RSI",
    "MACD",
    "ATR_14",
    "Volatility_20",
    "Price_SMA20_Ratio",
    "Price_SMA50_Ratio",
]

model_data = df[feature_cols + ["Target"]].replace([np.inf, -np.inf], np.nan).dropna()

# The last row has no known next-period outcome; exclude it from training.
if len(model_data) > 1:
    model_data = model_data.iloc[:-1]

if len(model_data) < 50 or model_data["Target"].nunique() < 2:
    st.warning(
        "There is not enough clean data or target-class variation to train "
        "the selected model. Try a longer historical period."
    )
else:
    X = model_data[feature_cols]
    y = model_data["Target"].astype(int)

    split_at = int(len(model_data) * (1 - test_fraction))
    split_at = max(30, min(split_at, len(model_data) - 10))

    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        st.warning(
            "The time-ordered train/test split contains only one class in "
            "part of the data. Try a longer period or a different ticker."
        )
    else:
        if model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
            )
        else:
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    )),
                ]
            )

        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        test_probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, test_predictions)
        try:
            auc = roc_auc_score(y_test, test_probabilities)
        except Exception:
            auc = np.nan

        a, b, c = st.columns(3)
        a.metric("Test accuracy", f"{accuracy * 100:.1f}%")
        b.metric("ROC-AUC", f"{auc:.3f}" if pd.notna(auc) else "N/A")
        c.metric("Test observations", f"{len(y_test)}")

        st.caption(
            "The test set is later in time than the training set. Historical "
            "performance does not guarantee future performance."
        )

        cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Down", "Predicted Up"],
            y=["Actual Down", "Actual Up"],
            color_continuous_scale="Blues",
            title="Confusion matrix",
            labels={"x": "Prediction", "y": "Actual"},
        )
        cm_fig.update_layout(height=350)
        st.plotly_chart(cm_fig, use_container_width=True)

        report = classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Down", "Up"],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        st.write("Classification report")
        st.dataframe(report_df.round(3), use_container_width=True)

        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, test_probabilities)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve"))
            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random baseline",
                    line=dict(dash="dash"),
                )
            )
            roc_fig.update_layout(
                title="ROC curve",
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                height=350,
            )
            st.plotly_chart(roc_fig, use_container_width=True)

        # Feature importance
        if model_choice == "Random Forest":
            importance = pd.Series(
                model.feature_importances_,
                index=feature_cols,
            ).sort_values(ascending=True)
        else:
            importance = pd.Series(
                np.abs(model.named_steps["classifier"].coef_[0]),
                index=feature_cols,
            ).sort_values(ascending=True)

        importance_fig = px.bar(
            x=importance.values,
            y=importance.index,
            orientation="h",
            title="Feature importance / absolute model coefficients",
            labels={"x": "Importance", "y": "Feature"},
        )
        importance_fig.update_layout(height=430)
        st.plotly_chart(importance_fig, use_container_width=True)

        # Latest prediction: use the latest available feature row.
        latest_features = (
            df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if not latest_features.empty:
            latest_X = latest_features.iloc[[-1]]
            predicted_class = int(model.predict(latest_X)[0])
            probability_up = float(model.predict_proba(latest_X)[0, 1])
            probability_down = 1 - probability_up

            st.markdown("#### Latest model estimate")

            p1, p2, p3 = st.columns(3)
            p1.metric("Estimated direction", "UP" if predicted_class == 1 else "DOWN")
            p2.metric("Probability of UP", f"{probability_up * 100:.1f}%")
            p3.metric("Probability of DOWN", f"{probability_down * 100:.1f}%")

            if probability_up >= 0.60:
                educational_signal = "BUY bias"
                signal_explanation = "The model assigns at least 60% probability to an upward move."
            elif probability_up <= 0.40:
                educational_signal = "SELL bias"
                signal_explanation = "The model assigns at most 40% probability to an upward move."
            else:
                educational_signal = "HOLD / Neutral"
                signal_explanation = "The model's estimated direction is not sufficiently decisive."

            st.info(
                f"Illustrative model signal: **{educational_signal}**. "
                f"{signal_explanation} This is a classroom demonstration, "
                "not a recommendation to trade."
            )

# ---------------------------------------------------------------------
# Financial news sentiment (Yahoo Finance headlines)
# ---------------------------------------------------------------------

st.subheader("7. Financial-news sentiment")

POSITIVE_WORDS = {
    "gain", "gains", "growth", "surge", "surges", "rise", "rises", "rally",
    "profit", "profits", "beat", "beats", "strong", "upgrade", "upgraded",
    "bullish", "positive", "record", "high", "higher", "improve", "improved",
    "outperform", "buy", "expansion", "dividend",
}

NEGATIVE_WORDS = {
    "loss", "losses", "fall", "falls", "drop", "drops", "decline", "declines",
    "weak", "miss", "misses", "cut", "downgrade", "downgraded", "bearish",
    "negative", "risk", "risks", "lawsuit", "probe", "investigation", "debt",
    "lower", "warning", "slump", "crash", "sell",
}


def headline_sentiment(headline):
    words = {
        word.strip(".,:;!?()[]{}'\"").lower()
        for word in str(headline).split()
    }
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)

    if positive > negative:
        label = "Positive"
    elif negative > positive:
        label = "Negative"
    else:
        label = "Neutral"

    return label, positive - negative


@st.cache_data(ttl=900, show_spinner=False)
def get_news(symbol):
    try:
        news_items = yf.Ticker(symbol).news
        rows = []

        for item in news_items or []:
            # yfinance response structures can vary by version.
            content = item.get("content", item)
            title = content.get("title") or item.get("title")
            if not title:
                continue

            publisher = content.get("publisher") or item.get("publisher", "Unknown")
            link = content.get("link") or item.get("link", "")
            rows.append(
                {
                    "Headline": title,
                    "Publisher": publisher,
                    "Link": link,
                }
            )

        return rows
    except Exception:
        return []


news = get_news(ticker)

if news:
    news_df = pd.DataFrame(news)
    sentiment_results = news_df["Headline"].apply(headline_sentiment)
    news_df["Sentiment"] = sentiment_results.apply(lambda x: x[0])
    news_df["Score"] = sentiment_results.apply(lambda x: x[1])

    sentiment_counts = (
        news_df["Sentiment"]
        .value_counts()
        .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        .reset_index()
    )
    sentiment_counts.columns = ["Sentiment", "Headlines"]

    news_col, chart_col = st.columns([1.5, 1])

    with news_col:
        st.write("Recent headlines")
        for _, row in news_df.iterrows():
            st.markdown(f"**{row['Sentiment']}** — {row['Headline']}")
            st.caption(f"Source: {row['Publisher']}")
            if row["Link"]:
                st.markdown(f"[Read article]({row['Link']})")
            st.divider()

    with chart_col:
        sentiment_fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Headlines",
            title="Headline sentiment distribution",
            hole=0.35,
        )
        st.plotly_chart(sentiment_fig, use_container_width=True)

    st.caption(
        "Sentiment is estimated with a simple keyword dictionary. It does not "
        "understand context, sarcasm, company-specific meaning or the full article."
    )
else:
    st.info(
        "No recent news headlines were returned for this ticker. "
        "News availability depends on the data provider."
    )

# ---------------------------------------------------------------------
# Data table and download
# ---------------------------------------------------------------------

st.subheader("8. Historical data table")

display_cols = [
    c for c in [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "SMA_50", "RSI", "MACD", "ATR_14",
        "DailyReturnPct", "Volatility_20",
    ]
    if c in df.columns
]

st.dataframe(
    df[display_cols].sort_index(ascending=False).head(100).round(3),
    use_container_width=True,
)

csv_data = df.to_csv().encode("utf-8")
st.download_button(
    label="Download historical data as CSV",
    data=csv_data,
    file_name=f"{ticker.replace('.', '_')}_stock_analysis.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------
# Educational notes
# ---------------------------------------------------------------------

with st.expander("How to interpret this dashboard"):
    st.markdown(
        """
        - **Moving averages:** Smooth price history and help illustrate trends.
        - **RSI:** A momentum indicator commonly interpreted using 70 and 30
          reference levels. These levels are not automatic trading instructions.
        - **MACD:** Compares short- and longer-term exponential moving averages.
        - **Bollinger Bands:** Show a rolling average and bands based on recent
          price variability.
        - **Volatility:** Measures variation in returns; it does not predict
          the direction of the next move.
        - **Maximum drawdown:** The largest peak-to-trough decline in the
          selected historical period.
        - **Machine learning:** Uses past observations to classify the next
          period as Up or Down. Results can be unstable and may not generalize.
        - **News sentiment:** Uses a small keyword list and should be treated
          as a rough classroom illustration, not a sophisticated NLP model.
        """
    )

st.warning(
    "© 2026 Jaydip Sen. All rights reserved. | "
    "Disclaimer: This application is for teaching and research demonstrations "
    "only. It is not investment advice, does not account for your financial "
    "circumstances, and should not be used as the sole basis for trading decisions."
)


