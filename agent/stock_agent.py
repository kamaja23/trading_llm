"""
Stock Analysis Agent - analyzes stocks using the trained trading LLM.

Fetches live market data, computes technical indicators, runs model inference,
and returns BUY/SELL/HOLD predictions with confidence scores.
"""

import sys
from pathlib import Path
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils.indicators import add_all_indicators
from utils.market_data import MarketDataError, fetch_market_data
from utils.token_definitions import ACTION_TOKENS


BULLISH_INDICATORS = {
    "trend": ("ST_UpTrend",),
    "volume": ("Hi_Volume",),
    "heikin_ashi": ("HA_UpCross",),
    "stochastic": ("STO_Cross", "STO_Oversold"),
}

BEARISH_INDICATORS = {
    "trend": ("ST_DownTrend",),
    "volume": ("Lo_Volume",),
    "heikin_ashi": ("HA_DownCross",),
    "stochastic": ("STO_Cross", "STO_Overbought"),
}

INDICATOR_LABELS = {
    "trend": "Trend vs MA(20)",
    "volume": "Volume vs MA(20)",
    "heikin_ashi": "Heikin Ashi Signal",
    "stochastic": "Stochastic (14,3)",
}

INDICATOR_VALUES = {
    "ST_UpTrend": "Price > 2% above MA — Uptrend",
    "ST_DownTrend": "Price > 2% below MA — Downtrend",
    "ST_Flat": "Price within 2% of MA — Neutral",
    "Hi_Volume": "Volume > 20% above average",
    "Lo_Volume": "Volume > 20% below average",
    "Avg_Volume": "Volume near average",
    "HA_UpCross": "Bullish Heikin Ashi crossover",
    "HA_DownCross": "Bearish Heikin Ashi crossover",
    "HA_Neutral": "No Heikin Ashi signal",
    "STO_Cross": "Stochastic %K crossing %D",
    "STO_NoCross": "No stochastic crossover",
    "STO_Oversold": "Stochastic below 20 — Oversold",
    "STO_Overbought": "Stochastic above 80 — Overbought",
}


@dataclass
class AnalysisResult:
    ticker: str
    date: str
    price: float
    prediction: str
    confidence: float
    action_probabilities: Dict[str, float]
    prediction_source: str
    training_samples: int
    training_accuracy: Optional[float]
    indicators: Dict[str, str]
    top_k_tokens: List[Tuple[str, float]]
    indicator_score: float
    historical_action: str
    input_sequence: str
    price_history: Optional[Dict] = None


class StockAnalysisError(Exception):
    pass


class StockAnalysisAgent:

    MODEL_PATH = project_root / 'models' / 'trading_llm' / 'final_model'

    def __init__(self, model_path: Optional[str] = None):
        path = Path(model_path) if model_path else self.MODEL_PATH
        if not path.exists():
            raise StockAnalysisError(
                f"Model not found at {path}. Run training pipeline first:\n"
                f"  python src/01_generate_training_data.py\n"
                f"  python src/02_train_model.py"
            )
        self.tokenizer = GPT2Tokenizer.from_pretrained(str(path))
        self.model = GPT2LMHeadModel.from_pretrained(str(path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self._action_token_ids = self._get_action_token_ids()

    def _get_action_token_ids(self) -> Dict[str, int]:
        ids = {}
        for action in ACTION_TOKENS:
            encoded = self.tokenizer.encode(action, add_special_tokens=False)
            if encoded:
                ids[action] = encoded[0]
        return ids

    def _resolve_symbol_token(self, ticker: str) -> str:
        t = ticker.upper()
        if t in {"SPY", "QQQ", "DIA", "IWM"}:
            return f"<SYM_{t}>"
        return "<SYM_SPY>"

    def fetch_data(
        self,
        ticker: str,
        period: str = "1y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        csv_path = self._local_csv_path(ticker)
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = self._filter_date_range(df, start_date, end_date)
            if len(df) >= 30:
                return df

        provider_error = None
        try:
            df = fetch_market_data(
                ticker,
                period=period,
                start_date=start_date,
                end_date=end_date,
            )
        except MarketDataError as exc:
            provider_error = str(exc)
            df = pd.DataFrame()

        if df.empty or len(df) < 30:
            csv_path = self._local_csv_path(ticker)
            if csv_path and csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                df = self._filter_date_range(df, start_date, end_date)
                if len(df) >= 30:
                    return df
            detail = f" Provider error: {provider_error}" if provider_error else ""
            raise StockAnalysisError(
                f"Need at least 30 trading days for {ticker}, got {len(df)}. "
                f"Check the ticker symbol or your internet connection.{detail}"
            )
        return df

    @staticmethod
    def _filter_date_range(
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        return df

    @staticmethod
    def _local_csv_path(ticker: str) -> Optional[Path]:
        ticker = ticker.upper()
        p = project_root / 'data' / 'raw' / f'{ticker}_daily.csv'
        return p if p.exists() else None

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_all_indicators(df)
        return df.dropna()

    def _score_indicators(self, indicators: Dict[str, str]) -> float:
        bullish = 0
        bearish = 0
        total = 0
        for key, val in indicators.items():
            if key in BULLISH_INDICATORS and val in BULLISH_INDICATORS[key]:
                bullish += 1
            if key in BEARISH_INDICATORS and val in BEARISH_INDICATORS[key]:
                bearish += 1
            total += 1
        if total == 0:
            return 0.0
        return (bullish - bearish) / total

    def _predict(self, input_sequence: str) -> Dict:
        inputs = self.tokenizer(input_sequence + " ", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]

        action_logit_values = torch.tensor([
            logits[token_id].item() for token_id in self._action_token_ids.values()
        ])
        action_probs = torch.nn.functional.softmax(action_logit_values, dim=-1)

        action_probs_dict = {}
        for (action, _), prob in zip(self._action_token_ids.items(), action_probs):
            action_probs_dict[action] = float(prob)

        predicted = max(action_probs_dict, key=action_probs_dict.get)
        confidence = action_probs_dict[predicted]

        full_probs = torch.nn.functional.softmax(logits, dim=-1)
        top_k_ids = torch.topk(full_probs, 8).indices
        top_k = [
            (self.tokenizer.decode([i]).strip(), float(full_probs[i].cpu()))
            for i in top_k_ids
        ]

        return {
            "prediction": predicted,
            "confidence": confidence,
            "action_probabilities": action_probs_dict,
            "top_k": sorted(top_k, key=lambda x: -x[1]),
        }

    def _train_stock_classifier(
        self,
        df_indicators: pd.DataFrame,
        latest: pd.Series,
        future_window: int = 10,
    ) -> Optional[Dict]:
        train_df = df_indicators.iloc[:-future_window].copy()
        if len(train_df) < 15:
            train_df = df_indicators.iloc[:-1].copy()
        if len(train_df) < 10:
            return None

        feature_cols = ["trend_token", "volume_token", "ha_token", "sto_token"]
        try:
            from sklearn.dummy import DummyClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.metrics import accuracy_score
        except ImportError:
            return None

        vectorizer = DictVectorizer(sparse=False)
        train_features = train_df[feature_cols].to_dict(orient="records")
        x_train = vectorizer.fit_transform(train_features)
        y_train = train_df["action_token"].tolist()

        if len(set(y_train)) == 1:
            classifier = DummyClassifier(strategy="most_frequent")
        else:
            classifier = RandomForestClassifier(
                n_estimators=80,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42,
            )
        classifier.fit(x_train, y_train)

        latest_features = {
            "trend_token": latest["trend_token"],
            "volume_token": latest["volume_token"],
            "ha_token": latest["ha_token"],
            "sto_token": latest["sto_token"],
        }
        x_latest = vectorizer.transform([latest_features])
        probs_raw = classifier.predict_proba(x_latest)[0]
        action_probs = {action: 0.0 for action in ACTION_TOKENS}
        for action, prob in zip(classifier.classes_, probs_raw):
            action_probs[action] = float(prob)

        predicted = max(action_probs, key=action_probs.get)
        train_preds = classifier.predict(x_train)
        train_accuracy = float(accuracy_score(y_train, train_preds))

        return {
            "prediction": predicted,
            "confidence": action_probs[predicted],
            "action_probabilities": action_probs,
            "training_samples": len(train_df),
            "training_accuracy": train_accuracy,
        }

    def analyze(
        self,
        ticker: str,
        period: str = "1y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_chart_data: bool = True,
    ) -> AnalysisResult:
        ticker = ticker.upper().strip()
        symbol_token = self._resolve_symbol_token(ticker)

        df = self.fetch_data(
            ticker,
            period=period,
            start_date=start_date,
            end_date=end_date,
        )
        df_indicators = self.compute_indicators(df)
        latest = df_indicators.iloc[-1]

        indicators = {
            "trend": latest["trend_token"],
            "volume": latest["volume_token"],
            "heikin_ashi": latest["ha_token"],
            "stochastic": latest["sto_token"],
        }

        input_sequence = (
            f"{symbol_token} <TF_DAILY> "
            f"{indicators['trend']} {indicators['volume']} "
            f"{indicators['heikin_ashi']} {indicators['stochastic']}"
        )

        prediction = self._predict(input_sequence)
        stock_classifier = self._train_stock_classifier(df_indicators, latest)
        if stock_classifier:
            displayed_prediction = stock_classifier
            prediction_source = "On-the-spot stock classifier"
            training_samples = stock_classifier["training_samples"]
            training_accuracy = stock_classifier["training_accuracy"]
        else:
            displayed_prediction = prediction
            prediction_source = "Trading LLM next-token model"
            training_samples = 0
            training_accuracy = None

        indicator_score = self._score_indicators(indicators)

        price_history = None
        if include_chart_data:
            chart_df = df_indicators[["Open", "High", "Low", "Close", "Volume"]].copy()
            price_history = {
                "dates": [str(d.date()) for d in chart_df.index],
                "Open": chart_df["Open"].tolist(),
                "High": chart_df["High"].tolist(),
                "Low": chart_df["Low"].tolist(),
                "Close": chart_df["Close"].tolist(),
                "Volume": chart_df["Volume"].tolist(),
            }

        return AnalysisResult(
            ticker=ticker,
            date=str(latest.name.date()),
            price=float(latest["Close"]),
            prediction=displayed_prediction["prediction"],
            confidence=displayed_prediction["confidence"],
            action_probabilities=displayed_prediction["action_probabilities"],
            prediction_source=prediction_source,
            training_samples=training_samples,
            training_accuracy=training_accuracy,
            indicators=indicators,
            top_k_tokens=prediction["top_k"],
            indicator_score=indicator_score,
            historical_action=latest["action_token"],
            input_sequence=input_sequence,
            price_history=price_history,
        )

    def analyze_multiple(
        self,
        tickers: List[str],
        period: str = "1y",
    ) -> Dict[str, AnalysisResult]:
        results = {}
        for t in tickers:
            try:
                results[t] = self.analyze(t, period=period)
            except StockAnalysisError as e:
                results[t] = StockAnalysisError(str(e))
        return results

    @staticmethod
    def recommendation_text(prediction: str, confidence: float) -> Tuple[str, str]:
        if prediction == "BUY":
            if confidence >= 0.6:
                return "Strong Buy — Consider investing", "green"
            elif confidence >= 0.4:
                return "Buy — Cautious opportunity", "lightgreen"
            return "Lean Buy — Weak signal", "olive"
        elif prediction == "SELL":
            if confidence >= 0.6:
                return "Strong Sell — Consider divesting", "red"
            elif confidence >= 0.4:
                return "Sell — Cautious signal", "lightcoral"
            return "Lean Sell — Weak signal", "sienna"
        else:
            if confidence >= 0.6:
                return "Hold — Neither invest nor divest", "gray"
            elif confidence >= 0.4:
                return "Hold — Uncertain market", "darkgray"
            return "Hold — Insufficient signal", "silver"

    @staticmethod
    def indicator_meaning(key: str, value: str) -> str:
        return INDICATOR_VALUES.get(value, value)

    @staticmethod
    def indicator_label(key: str) -> str:
        return INDICATOR_LABELS.get(key, key)


if __name__ == "__main__":
    agent = StockAnalysisAgent()
    result = agent.analyze("SPY")
    print(f"\n{result.ticker} Analysis ({result.date}):")
    print(f"  Price: ${result.price:.2f}")
    print(f"  Prediction: {result.prediction} ({result.confidence:.1%})")
    print(f"  Indicator Score: {result.indicator_score:+.2f}")
    print(f"  Indicators: {result.indicators}")
    print(f"  Top-k: {result.top_k_tokens[:3]}")
