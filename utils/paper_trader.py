from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TradeLog:
    ticker: str
    action: str
    shares: int
    price: float
    total: float
    date: str
    confidence: float


class PaperTrader:
    def __init__(self, initial_cash: float = 10_000.0):
        self.initial_cash = round(initial_cash, 2)
        self.cash = round(initial_cash, 2)
        self.holdings: dict[str, int] = {}
        self.trades: list[TradeLog] = []

    def buy(
        self,
        ticker: str,
        price: float,
        date: str,
        confidence: float,
        max_fraction: float = 0.3,
    ) -> TradeLog | None:
        if confidence >= 0.6:
            fraction = max_fraction
        elif confidence >= 0.4:
            fraction = max_fraction * 0.5
        else:
            fraction = max_fraction * 0.15

        available = self.cash * fraction
        shares = int(available / price)
        if shares <= 0:
            return None
        cost = round(shares * price, 2)
        self.cash = round(self.cash - cost, 2)
        self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
        trade = TradeLog(ticker, "BUY", shares, price, cost, date, confidence)
        self.trades.append(trade)
        return trade

    def sell(self, ticker: str, price: float, date: str, confidence: float) -> TradeLog | None:
        shares = self.holdings.get(ticker, 0)
        if shares <= 0:
            return None
        proceeds = round(shares * price, 2)
        self.cash = round(self.cash + proceeds, 2)
        del self.holdings[ticker]
        trade = TradeLog(ticker, "SELL", shares, price, proceeds, date, confidence)
        self.trades.append(trade)
        return trade

    def portfolio_value(self, prices: dict[str, float]) -> float:
        holdings_value = sum(
            shares * prices.get(t, 0) for t, shares in self.holdings.items()
        )
        return round(self.cash + holdings_value, 2)

    @property
    def pnl(self) -> float:
        return round(self.cash - self.initial_cash, 2)

    @property
    def total_invested(self) -> float:
        return round(sum(t.total for t in self.trades if t.action == "BUY"), 2)

    @property
    def total_realized(self) -> float:
        return round(sum(t.total for t in self.trades if t.action == "SELL"), 2)

    @property
    def position_count(self) -> int:
        return len(self.holdings)

    def cost_basis(self, ticker: str) -> float:
        buys = [t for t in self.trades if t.ticker == ticker and t.action == "BUY"]
        if not buys:
            return 0.0
        total_shares = sum(t.shares for t in buys)
        total_cost = sum(t.total for t in buys)
        return round(total_cost / total_shares, 2) if total_shares else 0.0

    def to_dict(self) -> dict:
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "holdings": dict(self.holdings),
            "trades": [asdict(t) for t in self.trades],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PaperTrader:
        trader = cls(data["initial_cash"])
        trader.cash = data["cash"]
        trader.holdings = {k: int(v) for k, v in data.get("holdings", {}).items()}
        trader.trades = [TradeLog(**t) for t in data.get("trades", [])]
        return trader
