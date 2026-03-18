"""
ML-based signal scoring module.
Trains a model on historical trades to predict signal quality.
Uses scikit-learn for lightweight, fast inference.
"""

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Position, Signal, TradeResult

logger = logging.getLogger(__name__)

MODEL_PATH = Path("ml_model.pkl")


class MLSignalScorer:
    """
    Scores trading signals based on historical trade outcomes.
    Features: EMA spread, RSI, ATR, spread, hour-of-day, day-of-week.
    Target: WIN (1) or LOSS (0).
    """

    def __init__(self, enabled: bool = True, min_samples: int = 30) -> None:
        self.enabled = enabled
        self.min_samples = min_samples
        self._model: Optional[object] = None
        self._feature_names = [
            "ema_spread", "rsi", "atr_norm", "spread",
            "hour", "day_of_week", "direction_buy",
        ]
        self._is_trained = False
        self._load_model()

    def _load_model(self) -> None:
        """Load a previously saved model if it exists."""
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                    self._model = data["model"]
                    self._is_trained = True
                    logger.info("Loaded ML model from %s", MODEL_PATH)
            except Exception as e:
                logger.warning("Failed to load ML model: %s", e)

    def _save_model(self) -> None:
        """Save the trained model to disk."""
        if self._model is not None:
            try:
                with open(MODEL_PATH, "wb") as f:
                    pickle.dump({"model": self._model}, f)
                logger.info("Saved ML model to %s", MODEL_PATH)
            except Exception as e:
                logger.warning("Failed to save ML model: %s", e)

    async def train(self, session: AsyncSession) -> dict:
        """
        Train the model on historical closed trades.
        Returns training summary.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        # Fetch closed positions with signal data
        result = await session.execute(
            select(Position).where(
                Position.is_open.is_(False),
                Position.result.in_([TradeResult.WIN, TradeResult.LOSS]),
            )
        )
        positions = result.scalars().all()

        if len(positions) < self.min_samples:
            return {
                "status": "insufficient_data",
                "samples": len(positions),
                "required": self.min_samples,
            }

        # Build feature matrix
        X, y = self._build_features(positions)

        if len(X) < self.min_samples:
            return {
                "status": "insufficient_features",
                "samples": len(X),
                "required": self.min_samples,
            }

        # Train model
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42,
        )

        # Cross-validation
        scores = cross_val_score(model, X, y, cv=min(5, len(X) // 5), scoring="accuracy")

        # Train on full data
        model.fit(X, y)
        self._model = model
        self._is_trained = True
        self._save_model()

        # Feature importance
        importances = dict(zip(self._feature_names, model.feature_importances_))

        summary = {
            "status": "trained",
            "samples": len(X),
            "accuracy_cv": round(float(scores.mean()), 4),
            "accuracy_std": round(float(scores.std()), 4),
            "feature_importances": {k: round(v, 4) for k, v in importances.items()},
            "win_rate": round(float(y.mean()), 4),
        }
        logger.info("ML model trained: %s", summary)
        return summary

    def score_signal(
        self,
        ema_fast: float,
        ema_slow: float,
        rsi: float,
        atr: float,
        spread: float,
        direction: str,
        close_price: float,
        timestamp: Optional[datetime] = None,
    ) -> dict:
        """
        Score a signal using the trained model.
        Returns: {"score": float, "recommendation": str, "is_trained": bool}
        """
        if not self.enabled or not self._is_trained or self._model is None:
            return {
                "score": 0.5,
                "recommendation": "neutral",
                "is_trained": self._is_trained,
            }

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        features = self._extract_features(
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi=rsi,
            atr=atr,
            spread=spread,
            direction=direction,
            close_price=close_price,
            timestamp=timestamp,
        )

        try:
            X = np.array([features])
            proba = self._model.predict_proba(X)[0]
            # proba[1] = probability of WIN
            win_prob = float(proba[1]) if len(proba) > 1 else 0.5

            recommendation = "strong_buy" if win_prob > 0.7 else (
                "buy" if win_prob > 0.55 else (
                    "neutral" if win_prob > 0.45 else (
                        "avoid" if win_prob > 0.3 else "strong_avoid"
                    )
                )
            )

            return {
                "score": round(win_prob, 4),
                "recommendation": recommendation,
                "is_trained": True,
            }
        except Exception as e:
            logger.warning("ML scoring failed: %s", e)
            return {"score": 0.5, "recommendation": "neutral", "is_trained": True}

    def _build_features(self, positions: list[Position]) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix and target vector from historical positions."""
        features_list: list[list[float]] = []
        targets: list[int] = []

        for pos in positions:
            if pos.entry_price is None or pos.entry_price == 0:
                continue

            opened = pos.opened_at or datetime.now(timezone.utc)
            # Approximate indicators from position data
            features = [
                0.0,  # ema_spread (not available, default)
                50.0,  # rsi (not available, default)
                0.0,  # atr_norm
                0.0,  # spread
                float(opened.hour),
                float(opened.weekday()),
                1.0 if pos.direction == "BUY" else 0.0,
            ]

            # Use stop_loss distance as ATR proxy
            if pos.stop_loss and pos.entry_price:
                sl_distance = abs(pos.entry_price - pos.stop_loss)
                features[2] = sl_distance / pos.entry_price  # atr_norm

            features_list.append(features)
            targets.append(1 if pos.result == TradeResult.WIN else 0)

        return np.array(features_list), np.array(targets)

    def _extract_features(
        self,
        ema_fast: float,
        ema_slow: float,
        rsi: float,
        atr: float,
        spread: float,
        direction: str,
        close_price: float,
        timestamp: datetime,
    ) -> list[float]:
        """Extract feature vector from signal data."""
        ema_spread = (ema_fast - ema_slow) / ema_slow if ema_slow != 0 else 0
        atr_norm = atr / close_price if close_price != 0 else 0

        return [
            ema_spread,
            rsi,
            atr_norm,
            spread,
            float(timestamp.hour),
            float(timestamp.weekday()),
            1.0 if direction == "BUY" else 0.0,
        ]
