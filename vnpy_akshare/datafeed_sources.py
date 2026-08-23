import importlib
from datetime import datetime
from typing import Any, Callable, Optional

import pandas as pd

from pytz import timezone

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.datafeed import BaseDatafeed
from vnpy.trader.object import BarData, HistoryRequest, TickData

CHINA_TZ = timezone("Asia/Shanghai")


class OptionalSourceDatafeed(BaseDatafeed):
    """Optional data source adapter for third-party market data libraries.

    The concrete data source may or may not be installed in the current Python
    environment. These classes intentionally fail gracefully so that the package
    remains importable and can still expose a consistent VeighNa datafeed API.
    """

    package_name: str = ""
    source_name: str = ""

    def __init__(self):
        self.inited = False
        self.module: Optional[Any] = None

    def init(self, output: Callable = print) -> bool:
        if not self.package_name:
            self.inited = False
            output(f"{self.source_name}数据源未配置依赖包")
            return False

        try:
            self.module = importlib.import_module(self.package_name)
            self.inited = True
            return True
        except ModuleNotFoundError:
            self.inited = False
            output(f"未安装 {self.source_name} 数据源依赖: pip install {self.package_name}")
            return False
        except Exception as exc:
            self.inited = False
            output(f"{self.source_name}数据源初始化失败: {exc!r}")
            return False

    def _normalize_source_dataframe(self, df: Any) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()

        frame = df.copy()
        if not isinstance(frame, pd.DataFrame):
            return pd.DataFrame()

        frame.columns = [str(col).strip().lower() for col in frame.columns]

        rename_map = {
            "tradingday": "datetime",
            "trade_date": "datetime",
            "date": "datetime",
            "day": "datetime",
            "time": "datetime",
            "datetime": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "vol": "volume",
            "turnover": "turnover",
            "amount": "turnover",
            "amt": "turnover",
            "price": "close",
        }
        frame = frame.rename(columns=rename_map)

        if "close" not in frame.columns and "price" in df.columns:
            frame["close"] = frame.get("price", 0)

        required = {"datetime", "open", "high", "low", "close", "volume"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"数据列缺失: {missing}")

        if "turnover" not in frame.columns:
            frame["turnover"] = 0.0

        try:
            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        except Exception:
            frame["datetime"] = pd.to_datetime(frame["datetime"].astype(str), errors="coerce")

        frame = frame.dropna(subset=["datetime"])
        frame = frame.sort_values("datetime")
        frame = frame.reset_index(drop=True)
        return frame

    def _convert_df_to_bars(self, req: HistoryRequest, df: Any, output: Callable = print) -> list[BarData]:
        try:
            frame = self._normalize_source_dataframe(df)
        except Exception as exc:
            output(f"{self.source_name}数据源返回数据格式异常: {exc!r}")
            return []

        if frame.empty:
            return []

        interval = req.interval or Interval.DAILY
        bars: list[BarData] = []
        for row in frame.itertuples(index=False):
            dt = getattr(row, "datetime", None)
            if dt is None:
                continue

            if hasattr(dt, "to_pydatetime"):
                dt = dt.to_pydatetime()
            else:
                try:
                    dt = pd.Timestamp(str(dt)).to_pydatetime()
                except Exception:
                    continue

            if interval == Interval.DAILY and dt.time() == datetime.min.time():
                dt = dt.replace(hour=15, minute=0, second=0, microsecond=0)
                if dt.tzinfo is None:
                    dt = CHINA_TZ.localize(dt)

            if dt.tzinfo is None:
                dt = CHINA_TZ.localize(dt)

            volume = float(getattr(row, "volume", 0) or 0)
            if interval == Interval.DAILY and req.exchange in {Exchange.SSE, Exchange.SZSE, Exchange.BSE}:
                if volume >= 10_000_000_000 and volume % 100 == 0:
                    volume /= 100.0

            try:
                bar = BarData(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=req.interval,
                    datetime=dt,
                    open_price=float(getattr(row, "open", 0) or 0),
                    high_price=float(getattr(row, "high", 0) or 0),
                    low_price=float(getattr(row, "low", 0) or 0),
                    close_price=float(getattr(row, "close", 0) or 0),
                    volume=volume,
                    turnover=float(getattr(row, "turnover", 0) or 0),
                    gateway_name=self.source_name.upper(),
                )
                bars.append(bar)
            except Exception as exc:
                output(f"{self.source_name}转换BarData失败: {exc!r}")

        return bars

    def query_bar_history(self, req: HistoryRequest, output: Callable = print) -> list[BarData]:
        if not self.inited and not self.init(output):
            return []

        output(f"{self.source_name}数据源接口已注册，但当前环境未提供可用的历史数据实现")
        return []

    def query_tick_history(self, req: HistoryRequest, output: Callable = print) -> list[TickData]:
        if not self.inited and not self.init(output):
            return []

        output(f"{self.source_name}数据源接口已注册，但当前环境未提供可用的Tick数据实现")
        return []


class MootdxDataFeed(OptionalSourceDatafeed):
    package_name = "mootdx"
    source_name = "Mootdx"

    def _fetch_daily_bars(self, quote_client: Any, symbol: str, begin: str, end: str) -> pd.DataFrame:
        """Fetch daily bars with a compatibility fallback for current mootdx releases."""
        try:
            return quote_client.k(symbol=symbol, begin=begin, end=end)
        except KeyError as exc:
            if "datetime" not in str(exc):
                raise

            raw_client = getattr(quote_client, "client", None)
            if raw_client is None:
                raise

            from mootdx.utils import get_stock_market
            market = get_stock_market(symbol, string=False)
            raw_data = raw_client.get_security_bars(9, market, symbol, 0, 800)
            if raw_data is None:
                raise ValueError("Mootdx raw TDX bar API returned no data")

            df = raw_client.to_df(raw_data)
            if df is None or getattr(df, "empty", False):
                raise ValueError("Mootdx raw TDX bar API returned empty data")
            return df

    def query_bar_history(self, req: HistoryRequest, output: Callable = print) -> list[BarData]:
        if not self.inited and not self.init(output):
            return []

        try:
            quotes_mod = getattr(self.module, "quotes", None)
            if quotes_mod is None:
                quotes_mod = importlib.import_module(f"{self.package_name}.quotes")
            quote_cls = getattr(quotes_mod, "StdQuotes", None)
            if quote_cls is None:
                raise AttributeError("mootdx.StdQuotes not found")

            quote_client = quote_cls()
            start_str = req.start.strftime("%Y%m%d")
            end_str = (req.end or datetime.now()).strftime("%Y%m%d")
            df = self._fetch_daily_bars(quote_client, req.symbol, start_str, end_str)
            if df is None:
                output(f"Mootdx查询{req.symbol}返回空结果，当前环境可能未配置有效的 TDX 服务器或股票代码")
                return []
            if getattr(df, "empty", False):
                output(f"Mootdx查询{req.symbol}返回空数据，当前环境可能未配置有效的 TDX 服务器或股票代码")
                return []
            if not hasattr(df, "columns"):
                output(f"Mootdx查询{req.symbol}返回非表格对象: {type(df)!r}")
                return []
            if "datetime" not in df.columns and "date" not in df.columns:
                output(f"Mootdx查询{req.symbol}返回列不符合预期: {list(df.columns)}")
                return []
            return self._convert_df_to_bars(req, df, output)
        except Exception as exc:
            output(f"Mootdx查询{req.symbol}失败: 当前环境未配置有效的 TDX 服务器或该股票代码无数据 ({type(exc).__name__}: {exc})")
            return []

    def query_tick_history(self, req: HistoryRequest, output: Callable = print) -> list[TickData]:
        if not self.inited and not self.init(output):
            return []

        try:
            quotes_mod = getattr(self.module, "quotes", None)
            if quotes_mod is None:
                quotes_mod = importlib.import_module(f"{self.package_name}.quotes")
            quote_cls = getattr(quotes_mod, "StdQuotes", None)
            if quote_cls is None:
                raise AttributeError("mootdx.StdQuotes not found")

            end_dt = req.end or datetime.now()
            days = pd.date_range(req.start.date(), end_dt.date(), freq="B")
            quote_client = quote_cls()
            ticks: list[TickData] = []

            try:
                for day in days:
                    day_str = day.strftime("%Y%m%d")
                    df = quote_client.transactions(symbol=req.symbol, date=day_str, start=0, offset=5000)
                    if df is None or getattr(df, "empty", False):
                        continue
                    if "time" not in df.columns:
                        continue

                    for row in df.itertuples(index=False):
                        time_str = getattr(row, "time", None)
                        if not time_str:
                            continue
                        try:
                            dt = datetime.strptime(f"{day_str} {time_str}", "%Y%m%d %H:%M")
                        except ValueError:
                            continue

                        price = float(getattr(row, "price", 0) or 0)
                        vol = float(getattr(row, "vol", 0) or 0)
                        volume = float(getattr(row, "volume", 0) or 0)
                        tick = TickData(
                            symbol=req.symbol,
                            exchange=req.exchange,
                            datetime=CHINA_TZ.localize(dt),
                            volume=volume,
                            turnover=price * vol,
                            open_interest=0,
                            last_price=price,
                            last_volume=vol,
                            bid_price_1=price,
                            bid_price_2=price,
                            bid_price_3=price,
                            bid_price_4=price,
                            bid_price_5=price,
                            ask_price_1=price,
                            ask_price_2=price,
                            ask_price_3=price,
                            ask_price_4=price,
                            ask_price_5=price,
                            bid_volume_1=int(vol),
                            bid_volume_2=0,
                            bid_volume_3=0,
                            bid_volume_4=0,
                            bid_volume_5=0,
                            ask_volume_1=int(vol),
                            ask_volume_2=0,
                            ask_volume_3=0,
                            ask_volume_4=0,
                            ask_volume_5=0,
                            gateway_name="MOOTDX",
                        )
                        ticks.append(tick)
            finally:
                try:
                    quote_client.close()
                except Exception:
                    pass

            return ticks
        except Exception as exc:
            output(f"Mootdx Tick查询{req.symbol}失败: {exc!r}")
            return []


class BaostockDataFeed(OptionalSourceDatafeed):
    package_name = "baostock"
    source_name = "Baostock"

    def query_bar_history(self, req: HistoryRequest, output: Callable = print) -> list[BarData]:
        if not self.inited and not self.init(output):
            return []

        try:
            frequency = self._map_interval(req.interval)
            code = req.symbol
            if req.exchange in {Exchange.SSE, Exchange.SZSE, Exchange.BSE}:
                prefix = {Exchange.SSE: "sh.", Exchange.SZSE: "sz.", Exchange.BSE: "bj."}.get(req.exchange, "")
                code = f"{prefix}{req.symbol}"

            start_str = req.start.strftime("%Y-%m-%d")
            end_str = (req.end or datetime.now()).strftime("%Y-%m-%d")
            fields = "date,open,high,low,close,volume,amount"
            login_res = self.module.login() if hasattr(self.module, "login") else None
            if login_res is not None:
                error_code = getattr(login_res, "error_code", None)
                if error_code not in (None, "0", 0):
                    output(f"Baostock登录失败: {login_res!r}")
                    return []

            result = self.module.query_history_k_data_plus(
                code=code,
                fields=fields,
                start_date=start_str,
                end_date=end_str,
                frequency=frequency,
                adjustflag="3",
            )
            if hasattr(result, "get_data"):
                df = result.get_data()
            else:
                df = result

            if hasattr(result, "error_code") and str(getattr(result, "error_code", "0")) not in {"0", "00"}:
                output(f"Baostock查询{req.symbol}失败: {result.error_code} {getattr(result, 'error_msg', '')}")
                return []

            if hasattr(self.module, "logout"):
                self.module.logout()
            return self._convert_df_to_bars(req, df, output)
        except Exception as exc:
            if hasattr(self.module, "logout"):
                try:
                    self.module.logout()
                except Exception:
                    pass
            output(f"Baostock查询{req.symbol}失败: {exc!r}")
            return []

    @staticmethod
    def _map_interval(interval: Optional[Interval]) -> str:
        if interval == Interval.MINUTE:
            return "5"
        if interval == Interval.HOUR:
            return "60"
        if interval == Interval.WEEKLY:
            return "w"
        return "d"


class EfinanceDataFeed(OptionalSourceDatafeed):
    package_name = "efinance"
    source_name = "eFinance"

    def query_bar_history(self, req: HistoryRequest, output: Callable = print) -> list[BarData]:
        if not self.inited and not self.init(output):
            return []

        try:
            if req.exchange in {Exchange.CFFEX, Exchange.SHFE, Exchange.CZCE, Exchange.DCE, Exchange.INE, Exchange.CFETS, Exchange.SGE, Exchange.WXE}:
                fut = getattr(self.module, "futures", None)
                if fut is None:
                    fut = importlib.import_module(f"{self.package_name}.futures")
                df = fut.get_quote_history(
                    req.symbol,
                    beg=req.start.strftime("%Y%m%d"),
                    end=(req.end or datetime.now()).strftime("%Y%m%d"),
                    klt=self._map_interval(req.interval),
                )
            else:
                stock = getattr(self.module, "stock", None)
                if stock is None:
                    stock = importlib.import_module(f"{self.package_name}.stock")
                df = stock.get_quote_history(
                    req.symbol,
                    beg=req.start.strftime("%Y%m%d"),
                    end=(req.end or datetime.now()).strftime("%Y%m%d"),
                    klt=self._map_interval(req.interval),
                    fqt=0,
                )

            if df is None:
                output(f"eFinance查询{req.symbol}返回空结果：当前网络/代理状态不满足或调用失败")
                return []
            if getattr(df, "empty", False):
                output(f"eFinance查询{req.symbol}返回空数据：当前网络/代理状态不满足或接口限制")
                return []
            return self._convert_df_to_bars(req, df, output)
        except Exception as exc:
            output(f"eFinance查询{req.symbol}失败: {exc!r}")
            return []

    @staticmethod
    def _map_interval(interval: Optional[Interval]) -> int:
        if interval == Interval.MINUTE:
            return 1
        if interval == Interval.HOUR:
            return 60
        if interval == Interval.WEEKLY:
            return 101
        return 101


MootdxDatafeed = MootdxDataFeed
BaostockDatafeed = BaostockDataFeed
EfinanceDatafeed = EfinanceDataFeed


DATAFEED_MAP = {
    "mootdx": MootdxDataFeed,
    "baostock": BaostockDataFeed,
    "efinance": EfinanceDataFeed,
}


def get_datafeed(name: str = "akshare") -> Optional[BaseDatafeed]:
    """Return a concrete source adapter by name.

    The default value keeps legacy AKShare behavior. Other names select the
    optional adapters without forcing their dependencies to be installed.
    """
    source_name = (name or "akshare").lower()
    if source_name == "akshare":
        from .akshare_datafeed import AKShareDataFeed
        return AKShareDataFeed()

    clazz = DATAFEED_MAP.get(source_name)
    if clazz is None:
        return None
    return clazz()


__all__ = [
    "OptionalSourceDatafeed",
    "MootdxDataFeed",
    "MootdxDatafeed",
    "BaostockDataFeed",
    "BaostockDatafeed",
    "EfinanceDataFeed",
    "EfinanceDatafeed",
    "get_datafeed",
    "DATAFEED_MAP",
]
