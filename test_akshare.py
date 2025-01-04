from datetime import datetime

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest

from vnpy_akshare.akshare_datafeed import AKShareDataFeed

req = HistoryRequest('000001', Exchange.SZSE, datetime.strptime("2018-01-01", "%Y-%m-%d"), datetime.strptime("2018-02-01", "%Y-%m-%d"))
print(AKShareDataFeed().query_bar_history(req))

# req = HistoryRequest('600276', Exchange.SSE, datetime.strptime("2018-01-01", "%Y-%m-%d"), datetime.strptime("2018-02-01", "%Y-%m-%d"))
# print(AKShareDataFeed().query_bar_history(req))

req = HistoryRequest('600276', Exchange.SZSE, datetime.strptime("2020-01-01", "%Y-%m-%d"), datetime.strptime("2020-02-01", "%Y-%m-%d"))
print(AKShareDataFeed().query_bar_history(req))

req = HistoryRequest('600276', Exchange.SZSE, datetime.strptime("2024-11-20 9:30:00", "%Y-%m-%d %H:%M:%S"), datetime.strptime("2024-11-20 15:00:00", "%Y-%m-%d %H:%M:%S"), interval=Interval.MINUTE)
print(AKShareDataFeed().query_bar_history(req))

# print('---------------------------------tick---------------------------------')

# req = HistoryRequest('sh600848', Exchange.SHFE, datetime.strptime("2020-01-01", "%Y-%m-%d"), datetime.strptime("2020-01-03", "%Y-%m-%d"))
# print(AKShareDataFeed().query_tick_history(req))
