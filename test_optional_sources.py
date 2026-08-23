from __future__ import annotations

from datetime import datetime

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest

from vnpy_akshare import get_datafeed


def check_tick_source(name: str, symbol: str, exchange: Exchange, start: datetime, end: datetime):
    datafeed = get_datafeed(name)
    if datafeed is None:
        print(f"[{name}] 不支持或未注册")
        return False

    req = HistoryRequest(symbol=symbol, exchange=exchange, start=start, end=end)
    try:
        ticks = datafeed.query_tick_history(req, output=print)
    except Exception as exc:
        print(f"[{name}] tick抓数异常: {type(exc).__name__}: {exc}")
        return False

    print(f"[{name}] ticks={len(ticks)}")
    if ticks:
        first = ticks[0]
        print(f"[{name}] first={first.datetime} last_price={first.last_price} volume={first.volume}")
        return True

    print(f"[{name}] tick未返回有效数据（环境未配置/代理不可用/数据为空）")
    return False


def check_source(name: str, symbol: str, exchange: Exchange, start: datetime, end: datetime, interval: Interval = Interval.DAILY):
    datafeed = get_datafeed(name)
    if datafeed is None:
        print(f"[{name}] 不支持或未注册")
        return False

    req = HistoryRequest(symbol=symbol, exchange=exchange, start=start, end=end, interval=interval)
    try:
        bars = datafeed.query_bar_history(req, output=print)
    except Exception as exc:
        print(f"[{name}] 抓数异常: {type(exc).__name__}: {exc}")
        return False

    print(f"[{name}] bars={len(bars)}")
    if bars:
        first = bars[0]
        print(f"[{name}] first={first.datetime} open={first.open_price} close={first.close_price} volume={first.volume}")
        return True

    print(f"[{name}] 未返回有效数据（环境未配置/代理不可用/数据为空）")
    return False


def check_imports():
    import importlib

    for name in ["vnpy_akshare", "vnpy_mootdx", "vnpy_baostock", "vnpy_efinance"]:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:
            print(f"[{name}] import failed: {type(exc).__name__}: {exc}")
            return False

        datafeed = getattr(mod, "Datafeed", None)
        print(f"[{name}] import ok, Datafeed={datafeed}")

    from vnpy_akshare import get_datafeed
    print(f"[akshare_factory] {get_datafeed('akshare')!r}")
    print(f"[mootdx_factory] {get_datafeed('mootdx')!r}")
    return True


if __name__ == "__main__":
    now = datetime.now()
    start = datetime(2024, 1, 2)
    end = datetime(2024, 1, 5)

    import_ok = check_imports()
    print(f"module_import_status={'ok' if import_ok else 'fail'}")

    results = {
        "akshare": check_source("akshare", "000001", Exchange.SSE, start, end),
        "baostock": check_source("baostock", "000001", Exchange.SSE, start, end),
        "mootdx": check_source("mootdx", "000001", Exchange.SSE, start, end),
        "efinance": check_source("efinance", "000001", Exchange.SSE, start, end),
    }

    tick_results = {
        "mootdx": check_tick_source("mootdx", "000001", Exchange.SSE, start, end),
        "baostock": check_tick_source("baostock", "000001", Exchange.SSE, start, end),
        "efinance": check_tick_source("efinance", "000001", Exchange.SSE, start, end),
    }

    for name, ok in results.items():
        print(f"source={name} status={'ok' if ok else 'fail'}")
    for name, ok in tick_results.items():
        print(f"tick_source={name} status={'ok' if ok else 'fail'}")

    print("测试完成")
