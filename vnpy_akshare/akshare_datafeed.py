import dataclasses
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable

import pandas as pd
from pytz import timezone

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData, HistoryRequest
from vnpy.trader.utility import round_to
from vnpy.trader.datafeed import BaseDatafeed

import akshare as ak

INTERVAL_VT2RQ: Dict[Interval, str] = {
    Interval.MINUTE: "1",
    Interval.DAILY: "daily",
    Interval.WEEKLY: "weekly",
}

INTERVAL_ADJUSTMENT_MAP: Dict[Interval, timedelta] = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(hours=-15)         # no need to adjust for daily bar
}

CHINA_TZ = timezone("Asia/Shanghai")

def date_to_datetime(dt: date) -> datetime:
    return datetime.strptime(str(dt), "%Y-%m-%d")


def string_to_date(ds: str) -> datetime:
    if isinstance(ds, date):
        return date_to_datetime(ds) 
    else:
        try:
            return datetime.strptime(ds, "%Y-%m-%d")
        except:
            return datetime.strptime(ds, "%Y%m%d")

def string_to_datetime(ds: str) -> datetime:
    if isinstance(ds, datetime):
        return datetime.strptime(str(ds), "%Y-%m-%d %H:%M:%S")
    
    else:
        try:
            return datetime.strptime(ds, "%Y-%m-%d %H:%M:%S")
        except:
            return datetime.strptime(ds, "%Y%m%d %H:%M:%S")


def date_to_string(dd: datetime) -> str:
    if dd is None:
        return None
    return dd.strftime("%Y%m%d")


@dataclasses.dataclass
class TradeDate:
    start:datetime
    end: datetime
    date_list: list[datetime]


class Country(Enum):
    China = "china"
    US = "us"
    UK = "uk"


country_trade_date: Dict[Country, TradeDate or None] = {
    Country.China: None,
    Country.US: None,
    Country.UK: None,
}


EXCHANGE_COUNTRY = {
    Country.China: {
        Exchange.CFFEX,
        Exchange.SHFE,
        Exchange.CZCE,
        Exchange.DCE,
        Exchange.INE,
        Exchange.SSE,
        Exchange.SZSE,
        Exchange.BSE,
        Exchange.SGE,
        Exchange.WXE,
        Exchange.CFETS,
        Exchange.XBOND,
    },
}


def get_country(exchange: Exchange):
    for country, exchange_set in EXCHANGE_COUNTRY.items():
        if exchange in exchange_set:
            return country

    return None


def get_zh_a_trader_date():
    #date_list = list(ak.stock_zh_index_daily_tx("sh000919").date)
    date_list = list(ak.tool_trade_date_hist_sina().trade_date)
    #date_list = [string_to_date(d) for d in date_list]
    date_list = [ date_to_datetime(d) for d in date_list]
    start = date_list[0]
    end = date_list[-1]
    return TradeDate(start, end, date_list)


def get_trade_date(exchange, start: datetime, end: datetime)-> List[datetime]:
    country = get_country(exchange)
    td = country_trade_date[country]
    if td is None:
        if country == Country.China:
            td = get_zh_a_trader_date()
        country_trade_date[country] = td

    return [d for d in td.date_list if end >= d >= start]


class BaseFeed:
    def query_bar_history(self, req: HistoryRequest, output: Callable = print) -> pd.DataFrame:
        pass

    def query_tick_history(self, req: HistoryRequest, output: Callable = print) -> pd.DataFrame:
        pass


class ZhADataFeed(BaseFeed):
    '''
    股票数据
    '''
    def query_bar_history(self, req: HistoryRequest, output: Callable = print) -> pd.DataFrame:
        symbol: str = req.symbol
        interval: Interval = req.interval
        start: datetime = req.start
        end: datetime = req.end

        if interval is None:
            interval = Interval.DAILY

        if interval == Interval.MINUTE:
            period = INTERVAL_VT2RQ[interval]  # 返回5分钟数据
            df = ak.stock_zh_a_hist_min_em(symbol, date_to_string(start), date_to_string(end), period, "hfq")
            df.rename(columns={
                '时间': "datetime",
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'turnover',
            }, inplace=True)
            return df
        
        else:
            period = INTERVAL_VT2RQ[interval]
            df = ak.stock_zh_a_hist(symbol, period, date_to_string(start), date_to_string(end), "hfq")
            df.rename(columns={
                '日期': "datetime",
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'turnover',
            }, inplace=True)

            return df

    def query_tick_history(self, req: HistoryRequest, output: Callable = print) -> pd.DataFrame:
        symbol: str = req.symbol
        start: datetime = req.start
        end: datetime = req.end

        if end is None:
            end = datetime.now()

        date_list = get_trade_date(req.exchange, start, end)
        ret = []
        for d in date_list:
            ret.append(ak.stock_zh_a_tick_tx(symbol, date_to_string(d)))

        return pd.concat(ret)

class ZhFutureDataFeed(BaseFeed):
    '''
    期货数据
    '''
    def query_bar_history(self, req: HistoryRequest, output: Callable = print) -> pd.DataFrame:
        symbol: str = req.symbol

        start: datetime = req.start
        end: datetime = req.end
        exchange = req.exchange

        df = ak.get_futures_daily(date_to_string(start), date_to_string(end), exchange.value)

        df.rename(columns={
            'date': "datetime",
        }, inplace=True)

        return df

    def query_tick_history(self, req: HistoryRequest, output: Callable = print) -> pd.DataFrame:
        symbol: str = req.symbol
        start: datetime = req.start
        end: datetime = req.end

        if end is None:
            end = datetime.now()

        date_list = get_trade_date(req.exchange, start, end)
        ret = []
        for d in date_list:
            ret.append(ak.stock_zh_a_tick_tx(symbol, date_to_string(d)))
        if ret:
            return pd.concat(ret) 
        else:
            return None


FEEDS = {
    Exchange.CFFEX: ZhFutureDataFeed,
    Exchange.SHFE: ZhFutureDataFeed,
    Exchange.CZCE: ZhFutureDataFeed,
    Exchange.DCE: ZhFutureDataFeed,
    Exchange.INE: ZhFutureDataFeed,

    Exchange.SSE: ZhADataFeed,
    Exchange.SZSE: ZhADataFeed,
    Exchange.BSE: ZhADataFeed,
}


class AKShareDataFeed(BaseDatafeed):
    """AKData数据服务接口"""

    def __init__(self):
        self.inited = False

    def init(self, output: Callable = print) -> bool:
        self.inited = True
        return True

    def convert_df_to_bar(self, req: HistoryRequest, df: DataFrame, output: Callable = print) -> Optional[List[BarData]]:

        data: List[BarData] = []

        interval: Interval = req.interval if req.interval is not None else Interval.DAILY

        # 为了将时间戳（K线结束时点）转换为VeighNa时间戳（K线开始时点）
        adjustment: timedelta = INTERVAL_ADJUSTMENT_MAP[interval]

        if df is not None:

            # 填充空串为NaN
            df.replace(to_replace=r'^\s*$',value=np.nan,regex=True,inplace=True)

            # 填充NaN为0
            df.fillna(0, inplace=True)
            
            for row in df.itertuples():

                if row.datetime == 0:
                    continue
                
                try:
                    dt: datetime = string_to_date(row.datetime)
                except:
                    dt: datetime = string_to_datetime(row.datetime)

                dt: datetime = dt - adjustment
                dt: datetime = CHINA_TZ.localize(dt)

                bar: BarData = BarData(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=interval,
                    datetime=dt,
                    open_price=round_to(row.open, 0.000001),
                    high_price=round_to(row.high, 0.000001),
                    low_price=round_to(row.low, 0.000001),
                    close_price=round_to(row.close, 0.000001),
                    volume=row.volume,
                    turnover=row.turnover,
                    open_interest=getattr(row, "open_interest", 0),
                    gateway_name="AK"
                )

                data.append(bar)

        return data

    #def convert_df_to_tick(self, req: HistoryRequest,  df: DataFrame, output: Callable = print) -> Optional[List[TickData]]:
    #     #return df

    #     data: List[TickData] = []

    #     if df is not None:

    #         # 填充空串为NaN
    #         df.replace(to_replace=r'^\s*$',value=np.nan,regex=True,inplace=True)

    #         # 填充NaN为0
    #         df.fillna(0, inplace=True)

    #         df.rename(columns={
    #             '成交时间': "datetime",
    #             '成交价格': 'open',
    #             '价格变动': 'close',
    #             '成交量': 'high',
    #             '成交额': 'low',
    #             '性质': 'volume'
    #         }, inplace=True)

    #         for row in df.itertuples():
    #             dt: datetime = string_to_date(row.成交时间)
    #             dt: datetime = CHINA_TZ.localize(dt)

    #             tick: TickData = TickData(
    #                 symbol=req.symbol,
    #                 exchange=req.exc,
    #                 datetime=dt,
    #                 open_price=row.open,
    #                 high_price=row.high,
    #                 low_price=row.low,
    #                 pre_close=row.prev_close,
    #                 last_price=row.last,
    #                 volume=row.volume,
    #                 turnover=row.total_turnover,
    #                 open_interest=getattr(row, "open_interest", 0),
    #                 limit_up=row.limit_up,
    #                 limit_down=row.limit_down,
    #                 bid_price_1=row.b1,
    #                 bid_price_2=row.b2,
    #                 bid_price_3=row.b3,
    #                 bid_price_4=row.b4,
    #                 bid_price_5=row.b5,
    #                 ask_price_1=row.a1,
    #                 ask_price_2=row.a2,
    #                 ask_price_3=row.a3,
    #                 ask_price_4=row.a4,
    #                 ask_price_5=row.a5,
    #                 bid_volume_1=row.b1_v,
    #                 bid_volume_2=row.b2_v,
    #                 bid_volume_3=row.b3_v,
    #                 bid_volume_4=row.b4_v,
    #                 bid_volume_5=row.b5_v,
    #                 ask_volume_1=row.a1_v,
    #                 ask_volume_2=row.a2_v,
    #                 ask_volume_3=row.a3_v,
    #                 ask_volume_4=row.a4_v,
    #                 ask_volume_5=row.a5_v,
    #                 gateway_name="RQ"
    #             )

    #             data.append(tick)

    #             data.append(tick)

    #     return data

    def query_bar_history(self, req: HistoryRequest, output: Callable = print) -> Optional[List[BarData]]:
        """查询K线数据"""
        if not self.inited:
            n: bool = self.init()
            if not n:
                return []

        exchange: Exchange = req.exchange
        if exchange not in FEEDS:
            return []

        clazz = FEEDS[exchange]
        df = clazz().query_bar_history(req, output)

        return self.convert_df_to_bar(req, df, output)
    
    # akshare tick数据不够详细无法使用，该接口暂不提供
    # def query_tick_history(self, req: HistoryRequest, output: Callable = print) -> Optional[List[TickData]]:
    #     exchange: Exchange = req.exchange
    #     if exchange not in FEEDS:
    #         return []

    #     clazz = FEEDS[exchange]

    #     df = clazz().query_tick_history(req, output)
    #     return self.convert_df_to_tick(req, df, output)

