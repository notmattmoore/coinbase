# CoinBase

A Python authentication library and API for CoinBase, geared towards automated
trading. See also the
[crypto-trading](https://github.com/notmattmoore/crypto-trading) repository

The authentication class is `CoinBaseAuth`, which extends `AuthBase` from the
[`requests.auth`](https://requests.readthedocs.io/en/latest/user/authentication/)
module. The API class is `CoinBase`.

#### Features
- account balance querying
- candle data retrieval (historical OHLCV)
- market, limit, stop-limit, and bracket order support
- order placement, status checking, and cancellation

Usage instructions and documentation is included in the comments in the source
file. Here is an example of how it might be used:

```
CB = CoinBase(CB_API_KEY, CB_API_PRIVATE_KEY)
CB.balances(["BTC", "ETH"])  # get balances from CoinBase
CB.candles("BTC-USD")        # get minute candles (configurable)
```

Also included is the script `data_hist.py`, which downloads or updates a
CoinBase historical candle data file for a specified trading pair.

Usage: `data_hist.py [-h] [-s PAIR] [-i {60,300,900,3600,21600,86400}] [--max-age YEARS] [-v] [--version] [FILE]`

Both `coinbase.py` and `data-hist.py` require `mylibs` from
[python-mylibs](https://github.com/notmattmoore/python-mylibs).
