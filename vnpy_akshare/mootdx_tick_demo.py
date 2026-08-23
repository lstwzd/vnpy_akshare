from datetime import datetime

from vnpy.trader.constant import Exchange
from vnpy.trader.object import HistoryRequest

from vnpy_akshare import get_datafeed


if __name__ == "__main__":
    feed = get_datafeed('mootdx')
    req = HistoryRequest(
        symbol='000001',
        exchange=Exchange.SSE,
        start=datetime(2024, 1, 2),
        end=datetime(2024, 1, 2),
    )
    ticks = feed.query_tick_history(req, output=print)
    print(f"count={len(ticks)}")
    if ticks:
        t = ticks[0]
        print(t.datetime, t.last_price, t.last_volume, t.turnover)
