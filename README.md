# VeighNa框架的AKShare数据服务接口

<p align="center">
  <img src ="https://vnpy.oss-cn-shanghai.aliyuncs.com/vnpy-logo.png"/>
</p>

<p align="center">
    <img src ="https://img.shields.io/badge/version-3.9.2.0-blueviolet.svg"/>
    <img src ="https://img.shields.io/badge/platform-windows|linux|macos-yellow.svg"/>
    <img src ="https://img.shields.io/badge/python-3.7|3.8|3.9|3.10-blue.svg"/>
    <img src ="https://img.shields.io/github/license/vnpy/vnpy.svg?color=orange"/>
</p>

## 说明

基于akshare开发，支持以下中国金融市场的K线数据：

* 期货：
  * CFFEX：中国金融期货交易所
  * SHFE：上海期货交易所
  * DCE：大连商品交易所
  * CZCE：郑州商品交易所
  * INE：上海国际能源交易中心
* 股票：
  * SSE：上海证券交易所
  * SZSE：深圳证券交易所
  * BSE: 北京证券交易所
* 基金：需要使用修改版[**akshare**](https://github.com/lstwzd/akshare)
  * SSE：上海证券交易所
  * SZSE：深圳证券交易所
  
## 数据使用事项

1. 主要支持A股数据，不支持期货数据
2. 支持基金数据，但需要采用安装修改版[**akshare**](https://github.com/lstwzd/akshare)
3. 支持日线级别数据，tick级别不支持
4. akshare数据源采用证券网站抓取，影响可能存在较慢的问题。
5. 除默认的 akshare 接口外，也预留了 Mootdx、Baostock、eFinance 三类可选数据源入口，便于在不同环境下切换数据源实现。

## 扩展数据源接口

当前包默认导出 `AKShareDataFeed` 作为 VeighNa 的 `Datafeed`，同时额外提供：

- `MootdxDataFeed`
- `BaostockDataFeed`
- `EfinanceDataFeed`

这些接口在未安装对应依赖时会优雅降级，并不会阻止模块导入。可按需安装：

```bash
pip install "vnpy_akshare[mootdx]"
pip install "vnpy_akshare[baostock]"
pip install "vnpy_akshare[efinance]"
```

### 当前环境真实验证结论

在当前 `niffler` conda 环境下，已做真实抓数验证：

- `akshare`：日线抓数正常
- `baostock`：日线抓数正常
- `mootdx`：当前环境缺少有效 TDX/数据服务配置，日线不返回数据；tick 数据可正常抓取
- `efinance`：当前环境代理/网络受限，日线不返回数据；tick 接口未提供稳定实现

也就是说，原始 AKShare 与 Baostock 在当前环境下可作为稳定的日线数据源；Mootdx 是环境依赖型数据源；eFinance 需要网络/代理条件满足才可用。

### Tick 能力说明

- `MootdxDataFeed` 支持 `query_tick_history()`，可按 `TickData` 提供分笔成交/逐笔行情数据，适合 VNpy 的 Tick 查询。
- `BaostockDataFeed` 与 `EfinanceDataFeed` 当前主要覆盖 K 线/日线数据，未提供稳定 Tick 适配；如果对应库无稳定历史 tick API，适配层会明确返回空结果并输出状态说明，而不会导致模块崩溃。

## 模块入口与 VeighNa 配置

为了兼容 VeighNa 的标准 `datafeed.name` 加载方式，包额外提供了顶层入口模块：

- `vnpy_akshare`
- `vnpy_mootdx`
- `vnpy_baostock`
- `vnpy_efinance`

对应的 `Datafeed` 类型分别为：

- `akshare` -> `AKShareDataFeed`
- `mootdx` -> `MootdxDataFeed`
- `baostock` -> `BaostockDataFeed`
- `efinance` -> `EfinanceDataFeed`

可在 VeighNa 中按如下方式切换：

```python
SETTINGS['datafeed.name'] = 'akshare'
SETTINGS['datafeed.name'] = 'mootdx'
SETTINGS['datafeed.name'] = 'baostock'
SETTINGS['datafeed.name'] = 'efinance'
```

如果当前环境缺少 TDX、代理网络或对应第三方依赖，接口会返回空结果并输出清晰状态信息，而不会在模块导入阶段直接崩溃。

## 安装

安装环境推荐基于3.0.0版本以上的【[**VeighNa Studio**](https://www.vnpy.com)】。

直接使用pip命令：

```
pip install vnpy_akshare
```


或者下载源代码后，解压后在cmd中运行：

```
pip install .
```


## 使用

在VeighNa中使用AkShare时，需要在全局配置中填写以下字段信息：

|名称|含义|必填|举例|
|---------|----|---|---|
|datafeed.name|名称|是|akshare|
|datafeed.username|用户名|否|token|
|datafeed.password|密码|否|token|

