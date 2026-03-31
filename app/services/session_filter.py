"""
Session filter module.
Restricts trading to specific market sessions (London, New York, etc.)
for higher volume and better execution.
"""

import logging
from datetime import datetime, timezone, time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TradingSession:
    """Defines a trading session with start and end times (UTC)."""
    name: str
    start: time
    end: time
    days: list[int]  # 0=Monday, 6=Sunday


# Major trading sessions in UTC
SESSIONS = {
    "london": TradingSession(
        name="London",
        start=time(7, 0),
        end=time(16, 0),
        days=[0, 1, 2, 3, 4],  # Mon-Fri
    ),
    "new_york": TradingSession(
        name="New York",
        start=time(12, 0),
        end=time(21, 0),
        days=[0, 1, 2, 3, 4],
    ),
    "tokyo": TradingSession(
        name="Tokyo",
        start=time(0, 0),
        end=time(9, 0),
        days=[0, 1, 2, 3, 4],
    ),
    "sydney": TradingSession(
        name="Sydney",
        start=time(21, 0),
        end=time(6, 0),  # Crosses midnight
        days=[0, 1, 2, 3, 4],
    ),
    "london_ny_overlap": TradingSession(
        name="London/NY Overlap",
        start=time(12, 0),
        end=time(16, 0),
        days=[0, 1, 2, 3, 4],
    ),
}


class SessionFilter:
    """
    Filters trades based on active trading sessions.
    Only allows trading during configured sessions for better execution.
    """

    def __init__(self, allowed_sessions: list[str] | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        if allowed_sessions:
            self.allowed_sessions = [
                SESSIONS[s] for s in allowed_sessions if s in SESSIONS
            ]
        else:
            # Default: London and New York sessions
            self.allowed_sessions = [SESSIONS["london"], SESSIONS["new_york"]]

    def _is_in_session(self, session: TradingSession, now: datetime) -> bool:
        """Check if current time falls within a session."""
        if now.weekday() not in session.days:
            return False

        current_time = now.time()

        # Handle sessions that cross midnight (e.g., Sydney)
        if session.start > session.end:
            return current_time >= session.start or current_time <= session.end
        else:
            return session.start <= current_time <= session.end

    def is_trading_allowed(self, now: datetime | None = None) -> tuple[bool, str]:
        """
        Check if trading is allowed based on current time and sessions.
        Returns (allowed, reason).
        """
        if not self.enabled:
            return True, "Session filter disabled"

        if now is None:
            now = datetime.now(timezone.utc)

        # Check weekend
        if now.weekday() >= 5:
            return False, f"Weekend (day={now.strftime('%A')}), markets closed"

        active_sessions = []
        for session in self.allowed_sessions:
            if self._is_in_session(session, now):
                active_sessions.append(session.name)

        if active_sessions:
            names = ", ".join(active_sessions)
            return True, f"Active sessions: {names}"

        next_session = self._next_session_start(now)
        return False, f"Outside trading sessions. Next session: {next_session}"

    def _next_session_start(self, now: datetime) -> str:
        """Find when the next trading session starts."""
        min_delta = None
        next_name = ""

        for session in self.allowed_sessions:
            for day_offset in range(7):
                candidate = now.replace(
                    hour=session.start.hour,
                    minute=session.start.minute,
                    second=0,
                    microsecond=0,
                )
                # Add day offset
                from datetime import timedelta
                candidate = candidate + timedelta(days=day_offset)

                if candidate <= now:
                    continue
                if candidate.weekday() not in session.days:
                    continue

                delta = candidate - now
                if min_delta is None or delta < min_delta:
                    min_delta = delta
                    hours = int(delta.total_seconds() / 3600)
                    minutes = int((delta.total_seconds() % 3600) / 60)
                    next_name = f"{session.name} in {hours}h {minutes}m"
                break

        return next_name or "Unknown"

    def get_active_sessions(self, now: datetime | None = None) -> list[str]:
        """Get list of currently active session names."""
        if now is None:
            now = datetime.now(timezone.utc)
        return [s.name for s in self.allowed_sessions if self._is_in_session(s, now)]
