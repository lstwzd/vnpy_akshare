from datetime import datetime

from vnpy.trader.constant import Exchange
from vnpy.trader.object import HistoryRequest

from vnpy_akshare import get_datafeed


def check_daily(name: str, symbol: str = "000001", exchange: Exchange = Exchange.SSE,
               start: datetime | None = None, end: datetime | None = None):
    if start is None:
        start = datetime(2024, 1, 2)
    if end is None:
        end = datetime(2024, 1, 5)

    feed = get_datafeed(name)
    if feed is None:
        print(f"[{name}] unsupported")
        return False

    req = HistoryRequest(symbol=symbol, exchange=exchange, start=start, end=end)
    try:
        bars = feed.query_bar_history(req, output=lambda *args, **kwargs: None)
    except Exception as exc:
        print(f"[{name}] ERROR {type(exc).__name__}: {exc}")
        return False

    if not bars:
        print(f"[{name}] no daily data returned")
        return False

    first = bars[0]
    print(f"[{name}] count={len(bars)} first={first.datetime} close={first.close_price} volume={first.volume}")
    return True


def compare_sources():
    start = datetime(2024, 1, 2)
    end = datetime(2024, 1, 5)
    samples = {}
    for name in ["akshare", "baostock"]:
        feed = get_datafeed(name)
        if feed is None:
            continue
        req = HistoryRequest(symbol="000001", exchange=Exchange.SSE, start=start, end=end)
        bars = feed.query_bar_history(req, output=lambda *args, **kwargs: None)
        samples[name] = bars

    if not samples:
        print("no comparable source data available")
        return

    baseline = samples["akshare"][0]
    print("\nsource compare (baseline=akshare)")
    for name, bars in samples.items():
        if not bars:
            print(f"[{name}] empty")
            continue
        first = bars[0]
        ratio = first.volume / baseline.volume if baseline.volume else 0
        print(f"[{name}] datetime={first.datetime} volume={first.volume} ratio_to_akshare={ratio:.4f} close={first.close_price}")


if __name__ == "__main__":
    print("daily bar smoke test")
    print("=" * 40)
    for source in ["akshare", "baostock", "mootdx", "efinance"]:
        check_daily(source)
    compare_sources()
