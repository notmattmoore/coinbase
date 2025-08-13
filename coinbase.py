# imports {{{1
# general
from   functools     import partial
from   requests.auth import AuthBase
from   time          import sleep, time
from   uuid          import uuid4
import jwt
import requests
import secrets

# ML
import numpy  as np
import pandas as pd

# custom modules
from mylibs.utils import datetime_iso, datetime_iso_gm, print_dyn_line, map_dict, str_dict, func_name
#----------------------------------------------------------------------------}}}1

class CoinBaseAuth(AuthBase):  # {{{1
  """
  A request.auth authenticator for CoinBase. Usage:
    api_url = ".../"
    auth = CoinBaseAuth(api_key, api_private_key)
    r = requests.post(api_url + "accounts", auth=auth)
  Parameters:
  - api_key
  - api_private_key
  """
  api_host = "api.coinbase.com"

  def __init__(self, api_key, api_private_key):  # {{{
    self.api_key         = api_key
    self.api_private_key = api_private_key
  #--------------------------------------------------------------------------}}}
  def __call__(self, request):  # {{{
    """
    Generate a JWT for authentication wtih coinbase API v3 and add it to the headers.

    Reference: https://docs.cdp.coinbase.com/advanced-trade/docs/rest-api-auth
    """
    jwt_payload = {
      'sub': self.api_key,
      'iss': "cdp",
      'nbf': (timestamp := int(time())),
      'exp': timestamp + 120,
      'uri': f"{request.method} {self.api_host}{request.path_url.split('?')[0]}",
    }
    jwt_token = jwt.encode(
        jwt_payload,
        # Docs say that private_key below should be
        #   from cryptography.hazmat.primitives import serialization
        #   serialization.load_pem_private_key(self.api_private_key.encode("UTF-8"), password=None)
        # but this seems to work too.
        self.api_private_key,
        algorithm = "ES256",
        headers = {"kid": self.api_key, "nonce": secrets.token_hex()},
    )

    request.headers.update({"Authorization": f"Bearer {jwt_token}"})
    return request
  #--------------------------------------------------------------------------}}}
#----------------------------------------------------------------------------}}}1

class CoinBase:   # {{{1
  """
  Wrapper for the CoinBase API. Tries to implement a stable API that a trading
  algorithm can interact with.
  """
  api_url = "https://api.coinbase.com/api/v3/brokerage/"

  def __init__(  # {{{
    self, api_key, api_private_key, api_timeout=10, api_retry=10, rate_limit_time=0.25, debug=False,
  ):
    self.api_key         = api_key
    self.api_private_key = api_private_key
    self.api_timeout     = api_timeout
    self.api_retry       = api_retry

    self.rate_limit_time    = rate_limit_time
    self._time_last_message = 0

    self.debug = debug

    self.session = requests.Session()
    self.session.auth = CoinBaseAuth(self.api_key, self.api_private_key)
  #--------------------------------------------------------------------------}}}

  def _send_message(self, method, endpoint, json=None, params=None, retry_count=1, retry_err_msgs=''):  # {{{
    """
    Send an API request and return the result (as json dictionary). Raises an
    exception on network error.
    Parameters:
    - method: HTTP method (get, post, delete, etc.)
    - endpoint: API Endpoint (to be added to api_url)
    - json: JSON payload for POST
    - params: HTTP request parameters
    """
    # To limit the rate of requests, if the last message was sent less than
    # $rate_limit_time ago, then sleep until $rate_limit_time will have passed
    # between sending messages.
    sleep_time = self.rate_limit_time - (time() - self._time_last_message)
    if self.debug and sleep_time > 0:
      print(f"DEBUG: sleeping {max(0, sleep_time)}s in _send_message() to avoid rate limiting.")
    sleep(max(0, sleep_time))
    self._time_last_message = time()

    url = self.api_url + endpoint
    try:
      reply = self.session.request(method, url, json=json, params=params, timeout=self.api_timeout)
      reply.raise_for_status()  # raise exception if response status isn't 200 (success)
    except Exception as err:
      if retry_count < self.api_retry:  # retry the message
        return self._send_message(
          method, endpoint, json=json, params=params, retry_count=retry_count+1,
          retry_err_msgs=retry_err_msgs + f"\n  - {err}"
        )
      else:
        print(
          f"ERROR {datetime_iso()}, {func_name()}: network error with arguments "
          f"{method=}, {endpoint=}, {json=}, {params=}.\n Tried {self.api_retry} "
          f"times and caught exceptions{retry_err_msgs}."
        )
        raise

    if self.debug:
      print(f"DEBUG: {method=}, {endpoint=}, {json=}, {params=}, {reply.json()=}")

    return self._type_inference(reply.json())
  #--------------------------------------------------------------------------}}}
  def _type_inference(self, x):  # {{{
    """
    Recursively try to infer all the types of x as bool, int, or float. If type
    inference fails, then just return x.
    """
    if isinstance(x, tuple):
      return tuple(map(self._type_inference, x))
    if isinstance(x, list):
      return list(map(self._type_inference, x))
    if isinstance(x, dict):
      return map_dict(self._type_inference, x)

    if isinstance(x, bool):
      return x
    try:
      return int(x)
    except (ValueError, TypeError):
      try:
        return float(x)
      except (ValueError, TypeError):
        return x
  #--------------------------------------------------------------------------}}}
  def _reply_is_error(self, reply):  # {{{
    """
    Return a bool indicating whether $reply is an error response. We expect
    $reply to be a dictionary, and recursively go through all subdictionaries
    looking for an error response.
    """
    if not isinstance(reply, dict):  # reply must be a dict
      return False

    for (key, value) in reply.items():
      if key.lower().startswith("error"):
        return True
      if isinstance(value, dict) and self._reply_is_error(value):
        return True
    return False
  #--------------------------------------------------------------------------}}}

  def balances(self, symbols):   # {{{
    """
    Return the account balances of an array of symbols as a dictionary with those
    symbols as keys. The value of a symbol will be a dictionary with keys "balance",
    "hold", "available". Should only raise an exception for user error.

    Reference: https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getaccounts
    """
    if isinstance(symbols, str):
      symbols = [symbols]

    # sanitize symbols
    symbols = [s.upper() for s in symbols]

    try:
      reply = self._send_message("get", "accounts")
      accounts = [acct for acct in reply["accounts"] if acct.get("currency") in symbols]
      while reply.get("has_next", False) and len(accounts) != len(symbols):
        reply = self._send_message("get", f"accounts?cursor={reply['cursor']}")
        accounts.extend([acct for acct in reply["accounts"] if acct.get("currency") in symbols])
    except Exception as err:
      print(
        f"ERROR {datetime_iso()}, {func_name()}: failed to get balances due to "
        f"exception\n  {err}."
      )
      accounts = []

    bals = {s:{"balance": 0, "hold": 0, "available": 0} for s in symbols}
    for acct in accounts:
      cur = acct["currency"]
      bals[cur]["available"] = acct["available_balance"]["value"]
      bals[cur]["hold"] = acct["hold"]["value"]
      bals[cur]["balance"] = bals[cur]["available"] + bals[cur]["hold"]

    return bals
  #--------------------------------------------------------------------------}}}
  def symbol_pair_info(self, symbol_pair, use_cache=True):   # {{{
    """
    Return information about the symbol pair, in particular the precision of prices
    that the exchange supports for orders. On network or server error returns a
    default reply. Should only raise an exception for user error. The response is of
    the form
      {
        "product_id": "BTC-USD",
        "price": 23512.06,
        "price_percentage_change_24h": -1.31109172857094,
        "volume_24h": 16033.18555722,
        "volume_percentage_change_24h": -20.08377199880537,
        "base_increment": 1e-08,
        "quote_increment": 0.01,
        "quote_min_size": 1,
        "quote_max_size": 50000000,
        "base_min_size": 1.6e-05,
        "base_max_size": 2600,
        "base_name": "Bitcoin",
        "quote_name": "US Dollar",
        "watched": False,
        "is_disabled": False,
        "new": False,
        "status": "online",
        "cancel_only": False,
        "limit_only": False,
        "post_only": False,
        "trading_disabled": False,
        "auction_mode": False,
        "product_type": "SPOT",
        "quote_currency_id": "USD",
        "base_currency_id": "BTC",
        "fcm_trading_session_details": None,
        "mid_market_price": '',
        "alias": '',
        "alias_to": [],
        "base_display_symbol": "BTC",
        "quote_display_symbol": "USD"
      }

    Reference: https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getproduct
    """
    symbol_pair = symbol_pair.upper()

    if use_cache:
      # initialize the cache if it hasn't already been
      if getattr(self, "_symbol_pair_info_cache", None) is None:
        self._symbol_pair_info_cache = {}
      if symbol_pair in self._symbol_pair_info_cache:
        return self._symbol_pair_info_cache[symbol_pair]

    try:
      reply = self._send_message("get", f"products/{symbol_pair}")
      if self._reply_is_error(reply):
        raise ValueError(f"received reply error {reply=}")
      if use_cache:
        self._symbol_pair_info_cache[symbol_pair] = reply
    except Exception as err:
      print(
        f"ERROR {datetime_iso()}, {func_name()}: failed to get symbol pair "
        f"information due to exception\n  {err}."
      )
      reply = {   # fallback
          "fallback"             : True,
          "product_id"           : symbol_pair,
          "base_increment"       : 1e-08,
          "base_min_size"        : 1.6e-05,
          "base_max_size"        : 2600,
          "base_name"            : (base_symbol := symbol_pair.split('-')[0]),
          "base_currency_id"     : base_symbol,
          "base_display_symbol"  : base_symbol,
          "quote_increment"      : 0.01,
          "quote_min_size"       : 1,
          "quote_max_size"       : 50000000,
          "quote_name"           : (quote_symbol := symbol_pair.split('-')[1]),
          "quote_currency_id"    : quote_symbol,
          "quote_display_symbol" : quote_symbol,
        }

    return reply
  #--------------------------------------------------------------------------}}}
  def format_base_quote(self, symbol_pair, signed=False, use_cache=True):  # {{{
    """
    Returns a pair of string formatting function for symbol_pair, appropriate
    for use when submitting orders.
    """
    symbol_pair = symbol_pair.upper()
    sign = '+' if signed else ''

    if use_cache:
      cache_key = f"{symbol_pair}_{signed=}"
      # initialize the cache if it hasn't already been
      if getattr(self, "_format_cache", None) is None:
        self._format_cache = {}
      if cache_key in self._format_cache:
        return self._format_cache[cache_key]

    # Get the information on the symbol pair. Define functions to format the
    # specified quantities of base/quote currencies as multiples of their respective
    # increments with the appropriate precision.
    symbol_pair_info = self.symbol_pair_info(symbol_pair)
    num_decimals = lambda x: -1 * min(int(f"{x:.0e}".split('e')[1]), 0)
    mult_of_inc = lambda x, inc: np.floor(x / inc) * inc
    format_prim = lambda x, inc=1: f"{mult_of_inc(x, inc):{sign}.{num_decimals(inc)}f}"
    format_base = partial(format_prim, inc=symbol_pair_info['base_increment'])
    format_quote = partial(format_prim, inc=symbol_pair_info['quote_increment'])

    if use_cache:
      self._format_cache[cache_key] = (format_base, format_quote)

    return format_base, format_quote
  #--------------------------------------------------------------------------}}}

  def candles(self, symbol_pair, granularity=None, interval=None, start=None, end=None, verbose=False):  # {{{
    """
    Return a DataFrame of $granularity candle data from time $start to $stop.
    Parameters:
    - symbol_pair: the trading pair
    - granularity: candle granularity, valid choices are "ONE_MINUTE",
      "FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE", "ONE_HOUR", "TWO_HOUR",
      "SIX_HOUR", and "ONE_DAY". One of granularity or interval must be set.
    - interval: candle interval, must be one of 60, 300, 900, 1800, 3600, 7200,
      21600, 86400. One of granularity or interval must be set.
    - start, end: epoch times for the first and last candle.
    - verbose: 0-2 how much status information to print.
    If start or stop are omitted, then return the previous 300 candles. Should only
    raise an exception for user error.
    """
    # sanitize inputs
    symbol_pair = symbol_pair.upper()
    granularity, interval = self._granularity_interval(granularity=granularity, interval=interval)

    if end is None:
      end = int(time())
    if start is None:
      start = end - 299 * interval
    end = int(np.floor(end / interval)) * interval  # start and end should be multiples of $interval
    start = int(np.ceil(start / interval)) * interval
    if verbose:
      print(f"Downloading {symbol_pair} with granularity {granularity} from epoch")
      print(f"  {start} ({pd.to_datetime(start, unit='s')})")
      print("to epoch")
      print(f"  {end} ({pd.to_datetime(end, unit='s')}).")

    # We can only request 300 candles at a time, so we iterate.
    R = pd.DataFrame(columns=["unix", "date", "open", "high", "low", "close", "volume"])
    for end_loop in range(end, start - 1, -300*interval):
      start_loop = max(start, end_loop - 299*interval)
      R_loop = self._candles_single_page(
        symbol_pair, granularity=granularity, interval=interval,
        start=start_loop, end=end_loop
      )
      if len(R_loop) == 0:
        break
      R = pd.concat([R, R_loop], ignore_index=True)
      if verbose:
        print_dyn_line("... at", R_loop.iloc[[0]].to_string(header=False, index=False))

    if verbose:   # print a final newline
      print()
    return R
  #--------------------------------------------------------------------------}}}
  def candle_latest(self, symbol_pair, granularity=None, interval=None, window_mult=2):  # {{{
    """
    Get the latest candle with $granularity or $interval precision (see documentation
    for candles()). If neither are provided, default to 1 minute precision. Should
    only raise an exception for user error.
    """
    if granularity is None and interval is None:
      granularity, interval = "ONE_MINUTE", 60
    granularity, interval = self._granularity_interval(granularity=granularity, interval=interval)
    time_now = int(time())
    candles = self.candles(
      symbol_pair, granularity=granularity, interval=interval,
      start=time_now - window_mult * interval, end=time_now
    )
    if len(candles) == 0:
      candles = candles.reindex(index=[0])
    return candles.iloc[0]
#----------------------------------------------------------------------------}}}
  def _candles_single_page(self, symbol_pair, granularity=None, interval=None, start=None, end=None):   # {{{
    """
    Return a DataFrame of $granularity candle data from time $start to $stop.
    Parameters:
    - symbol_pair: the trading pair
    - granularity: candle granularity, valid choices are "ONE_MINUTE",
      "FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE", "ONE_HOUR", "TWO_HOUR",
      "SIX_HOUR", and "ONE_DAY". One of granularity or interval must be set.
    - interval: candle interval, must be one of 60, 300, 900, 1800, 3600, 7200,
      21600, 86400. One of granularity or interval must be set.
     "SIX_HOUR", "ONE_DAY"
    - start, end: epoch times for the first and last candle.
    NB: (end-start)/interval must be < 300 or the request will be rejected (set
    to 299 to get the maximum of 300 candles).

    If start or stop are omitted, then return the previous 300 candles.

    Should only raise an exception for user error.

    Reference: https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getcandles
    """
    granularity, interval = self._granularity_interval(granularity=granularity, interval=interval)

    if end is None:
      end = int(time())
    if start is None:
      start = end - 299 * interval
    endpoint = f"products/{symbol_pair}/candles?start={start}&end={end}&granularity={granularity}"

    try:
      # The reply is a dictionary with key "candles". The value at "candles" is a
      # list of dictionaries, with each entry in the list being a single candle.
      reply = self._send_message("get", endpoint)
      candles = reply["candles"]
      if not isinstance(candles, list):
        raise TypeError(f"received non-list reply {candles}")
    except Exception as err:
      print(
        f"ERROR {datetime_iso()}, {func_name()}: failed to read candles due to "
        f"exception  {err}.\nReceived reply {locals().get('reply', '(no reply)')}."
      )
      candles = []

    if len(candles) == 0:
      candles = pd.DataFrame(columns=["start", "open", "high", "low", "close", "volume"])

    candles = pd.DataFrame(candles).rename(columns={"start": "unix"})
    candles["date"] = pd.to_datetime(candles["unix"], unit='s')
    candles = candles.reindex(columns=["unix", "date", "open", "high", "low", "close", "volume"])
    return candles
  #--------------------------------------------------------------------------}}}
  def _granularity_interval(self, granularity=None, interval=None):  # {{{
    """
    Convert between granularity and interval, sanitizing both. Returns the pair
    ($granularity, $interval).
    """
    # cache the lookup table
    if getattr(self, "_granularity_interval_lookup", False) == False:
      self._granularity_interval_lookup = {
        "ONE_MINUTE"     : (one_min := 60),
        "FIVE_MINUTE"    : 5 * one_min,
        "FIFTEEN_MINUTE" : 15 * one_min,
        "THIRTY_MINUTE"  : 30 * one_min,
        "ONE_HOUR"       : (one_hour := 60 * one_min),
        "TWO_HOUR"       : 2 * one_hour,
        "SIX_HOUR"       : 6 * one_hour,
        "ONE_DAY"        : 24 * one_hour,
      }
      # make it bidirectional
      self._granularity_interval_lookup.update({v:k for (k,v) in self._granularity_interval_lookup.items()})

    if granularity != None:
      granularity = granularity.upper()
      interval_lookup = self._granularity_interval_lookup[granularity]
      if interval not in (None, interval_lookup):
        raise ValueError(f"interval does not match granularity: {granularity=} {interval=} {interval_lookup=}")
      return granularity, interval_lookup
    if interval != None:
      interval = int(interval)
      granularity_lookup = self._granularity_interval_lookup[interval]
      if granularity != None and granularity.upper() != granularity_lookup:
        raise ValueError(f"granularity does not match interval: {granularity=} {interval=} {interval_lookup=}")
      return granularity_lookup, interval
    else:
      raise ValueError("one of granularity or interval must be set")
  #--------------------------------------------------------------------------}}}

  def order_place(self, order):  # {{{
    """
    Place an order. Returns the pair ($order_id, $reply), where $order_id
    identifies the order on the server (if order placement was successful) and
    $reply is the server's full reply. If placing the order fails due to a
    network or server error, then order_id is set to None. Should only raise an
    exception for user error.

    Market, limit, and stop-limit orders are documented onlie. A bracket order is a
    limit order to buy or sell, activated when the asset price is below a specified
    price or above a different specified price. If posted, the order will remain
    until canceled.

    The order parameter is a dictionary of form
    {
      "symbol_pair"    : The trading pair, e.g. "BTC-USD".
      "side"           : "buy" or "sell".
      "type"           : "market", "limit", "stop-limit", or "bracket".
      "price"          : Price to buy/sell at, required for limit and bracket orders.
      "quote_size"     : Amount of quote currency (e.g. USD) to spend, required for
                         market buys, optional for limit buys and bracket orders on
                         either side (if base_size is specified, it overrides this),
                         not used for stop-limit buys, which must use base_size.
      "base_size"      : Amount of base currency (e.g. BTC) to buy or sell, required
                         for market and limit sells, stop-limit orders on either
                         side, bracket orders on either side, and optional for limit
                         buys.
      "post_only"      : True or False. Whether to force the limit order to be a market
                         maker order and not a taker (there's a difference in fees).
      "stop_price"     : For stop-limit and bracket orders. Price at which the order
                         will convert to a limit order. If stop direction is up, then
                         trigger when the price goes above this, otherwise trigger
                         when the price goes below this. The exchange rejects orders
                         where the stop price is immediately triggered as well.
      "stop_dir"       : For stop-limit orders. Either "up" or "down".
      "expiration"     : Time at which to cancel the order. Format: either a numberic
                         type or a string of the format "2023-02-03T11:01:32". Only
                         applies to limit and stop-limit orders.
      "local"          : Whether the order is executing locally. If the order has
                         been given as an argument, this must be False.
    }.
    We translate this standard format to the one supported by CoinBase:
      {
        "client_order_id" : Client set unique UUID for this order.
        "product_id" : the trading pair, e.g. "BTC-USD".
        "side" : "BUY" or "SELL".
        "order_configuration" : dictionary of the form
          {
            "market_market_ioc" : market orders (immediate or cancel), a
              dictionary of the form
              {
                "quote_size" : (string) Amount of quote (e.g. USD) currency to
                  spend on order. Required for BUY orders.
                "base_size" : (string) Amount of base currency (e.g. BTC) to
                  spend on order. Required for SELL orders.
              },
            "limit_limit_gtc" : limit orders (good til cancelled), a dictionary
              of the form
              {
                "base_size" : See market order description.
                "limit_price" : (string) Ceiling price for which the order
                  should get filled.
                "post_only" : (bool) Whether to force the order to be a maker,
                  not a taker.
              },
            "limit_limit_gtd" : limit orders (good til date), a dictionary of
              the form
              {
                "base_size" : See market order description.
                "limit_price" : See limit order (GTC) description.
                "end_time" : (string) UTC time at which the order should be
                  cancelled if it's not filled. Format: 2023-02-02T12:25:11Z.
                  Note the 'Z' on the end!
                "post_only" : See limit order description.
              },

            "stop_limit_stop_limit_gtc" : stop limit orders (good til
              cancelled), a dictionary of the form
              {
                "base_size" : See market order description.
                "limit_price" : See limit order (GTC) description.
                "stop_price" : (string) Price at which the order should trigger
                   --- if stop direction is up, then the order will trigger when
                   the last trade price goes above this, otherwise order will
                   trigger when last trade price goes below this price.
                "stop_direction" : "STOP_DIRECTION_STOP_UP" or "STOP_DIRECTION_STOP_DOWN"
              },
            "stop_limit_stop_limit_gtd" : stop limit orders (good til date), a
              dictionary of the form
              {
                "base_size" : See market order description.
                "limit_price" : See limit order (GTC) description.
                "stop_price" : See stop limit order (GTC) description.
                "stop_direction" : See stop limit order (GTC) description.
                "end_time" : See stop limit order (GTD) description.
              }

            "trigger_bracket_gtc" : bracket orders (good til cancelled), a dictionary
              of the form
              {
                "base_size" : See market order description.
                "limit_price" : The specified price, or better, that the Order should
                  be executed at. A buy order will execute at or lower than the limit
                  price. A sell order will execute at or higher than the limit price.
                "stop_trigger_price" : The price where the position will be exited.
                  When triggered, a stop limit order is automatically placed with a
                  limit price 5% higher for BUYS and 5% lower for SELLS.
              }
            "trigger_bracket_gtd" : bracket orders (good til date), a dictionary of
              the form
              {
                "base_size" : See bracket order (GTC) description.
                "limit_price" : See bracket order (GTC) description.
                "stop_trigger_price" : See bracket order (GTC) description.
                "end_time" : See stop limit order (GTD) description.
              }
          }
        "leverage" : the amount of leverage for the order (default is 1.0).
        "margin_type" : "CROSS" or "ISOLATED". Defaults to "CROSS".
      }.
    A successful order placement gets a reply of the form
      {
        "success": True,   # Means success!
        "failure_reason": "UNKNOWN_FAILURE_REASON",
        "success_response":
          {
            "order_id": "adeb2ea8-2071-475d-9355-8dec46744548",
            "product_id": 'BTC-USD',
            "side": "SELL",
            "client_order_id": "5d1b2837-e481-4683-9936-a65416c11b83"
          },
        "order_configuration": {"market_market_ioc": {"base_size": 0.001}}
      },
    while an unsuccessful order placement gets a reply of the form
      {
        "success": False,  # Means failure!
        "failure_reason": "UNKNOWN_FAILURE_REASON",
        "error_response":
          {
            "error": "UNSUPPORTED_ORDER_CONFIGURATION",
            "message": "rpc error: code = InvalidArgument desc = Market sells must be parameterized in base currency",
            "error_details": '',
            "preview_failure_reason": "PREVIEW_INVALID_ORDER_CONFIG"
          },
        "order_configuration": {"market_market_ioc": {"quote_size": 23.56}}
      }

    Reference: https://docs.cdp.coinbase.com/advanced-trade/reference/retailbrokerageapi_postorder
    """
    # Validate the keys of the provided order.
    order_keys_valid = [ "symbol_pair", "side", "type", "price", "quote_size", "base_size", "post_only", "stop_price", "stop_dir", "expiration", "local" ]
    for k in order.keys():
      if k not in order_keys_valid:
        raise KeyError(f"invalid key in order:\n  {k=}\n  {order=}")
    if order.get("local", False):   # raise exception if the order is locally executing
      raise ValueError(f"cannot place remote order for order that is locally executing:\n  {order=}")

    order_CB = {
      "client_order_id" : str(uuid4()),
      "product_id"      : order["symbol_pair"].upper(),
      "side"            : order["side"].upper(),
    }

    # Build the order configuration (as {order_conf_key: order_conf}).
    order_type = order["type"].lower()
    order_side = order["side"].lower()
    if order_type == "market":
      order_conf_key = "market_market_ioc"
      if order_side == "buy":
        order_conf = {"quote_size": order["quote_size"]}
      elif order_side == "sell":
        order_conf = {"base_size": order["base_size"]}
    elif order_type == "limit":
      order_conf_key = "limit_limit_"
      order_conf = {"limit_price": order["price"]}
      if "post_only" in order.keys():
        order_conf["post_only"] = order["post_only"]
      if "expiration" not in order.keys():
        order_conf_key += "gtc"
      else:
        order_conf_key += "gtd"
        order_conf["end_time"] = order["expiration"]
      if order_side == "buy":
        # By default, use the specified base_size. If not available, calculate
        # the base size given the specified quote_size.
        order_conf["base_size"] = order.get("base_size", order.get("quote_size", np.nan) / order["price"])
      elif order_side == "sell":
        order_conf["base_size"] = order["base_size"]
    elif order_type == "stop-limit":
      order_conf_key = "stop_limit_stop_limit_"
      order_conf = {
        "base_size"      : order["base_size"],
        "limit_price"    : order["price"],
        "stop_price"     : order["stop_price"],
        "stop_direction" : f"STOP_DIRECTION_STOP_{order['stop_dir'].upper()}",
      }
      if "expiration" not in order.keys():
        order_conf_key += "gtc"
      else:
        order_conf_key += "gtd"
        order_conf["end_time"] = order["expiration"]
    elif order_type == "bracket":
      order_conf_key = "trigger_bracket_"
      order_conf = {
        "limit_price"        : order["price"],
        "stop_trigger_price" : order["stop_price"],
      }
      if "expiration" not in order.keys():
        order_conf_key += "gtc"
      else:
        order_conf_key += "gtd"
        order_conf["end_time"] = order["expiration"]
      if order_side == "buy":
        # By default, use the specified base_size. If not available, calculate
        # the base size given the specified quote_size.
        order_conf["base_size"] = order.get("base_size", order.get("quote_size", np.nan) / order["price"])
      elif order_side == "sell":
        order_conf["base_size"] = order["base_size"]
    else:
      print(
        f"WARNING {datetime_iso()}, {func_name()}: unrecognized order type "
        f"{order['type']=}."
      )

    # Format all values appropriately.
    format_base, format_quote = self.format_base_quote(order["symbol_pair"])
    for k in order_conf.keys():
      if k in ["quote_size", "limit_price", "stop_price", "stop_trigger_price"]:
        order_conf[k] = format_quote(order_conf[k])
      elif k in ["base_size"]:
        order_conf[k] = format_base(order_conf[k])
      elif k == "end_time":
        # If time is not a string, then convert it to YYYY-MM-DDTHH:MM:SSZ format.
        if not isinstance(order_conf[k], str):
          order_conf[k] = datetime_iso_gm(order_conf[k], sep='T')
        order_conf[k] = order_conf[k].upper().replace(' ', 'T')
        if not order_conf[k].endswith('Z'):
          order_conf[k] += 'Z'

    order_CB["order_configuration"] = {order_conf_key: order_conf}

    # Finally, place the order. Allow for a certain number of retries.
    for i in range(self.api_retry):
      try:
        reply = self._send_message("post", "orders", json=order_CB)
        if self._reply_is_error(reply) or not reply.get("success", False):
          raise ValueError(f"received reply error {reply=}")
        order_id = reply["success_response"]["order_id"]
        break
      except Exception as err:
        print(
          f"ERROR {datetime_iso()}, {func_name()}: failed to place order\n  "
          f"{order=}\n  {order_CB=}\ndue to exception\n  {err}. Will try "
          f"{self.api_retry - i - 1} more times."
        )
        reply = locals().get("reply", {})
        order_id = None

    return order_id, reply
  #--------------------------------------------------------------------------}}}
  def order_info(self, order_id):  # {{{
    """
    Get information on an order. Returns the pair ($order_std, $reply), where
    $order_std is a standardized order format and $reply is the server reply. If
    there is a network or server error getting the order, then catch it, issue a
    printed error warning, and return ({}, reply). Should only raise an exception for
    user error. The standardized format is specified below:
      {
        "order_id"      : order id on the exchange.
        "status"        : "open", "filled", "cancelled", "expired", or "failed"
        "settled"       : True or False.
        "perc_complete" : How much of the order has been filled.
        "symbol_pair"   : The trading pair (e.g. BTC-USD).
        "quote_delta"   : Change in quote currency (e.g. USD) (- for buy, + for
                          sell).
        "base_delta"    : Change in base currency (e.g. BTC) (+ for buy, - for
                          sell).
        "fees"          : Fees charged by the exchange.

        "order" : the standardized submitted order, like the argument to
          order_place(), of the form
          {
            "symbol_pair" : as in CB.order_place().
            "side"        : as above.
            "type"        : as above.
            "price"       : as above.
            "quote_size"  : as above.
            "base_size"   : as above.
            "stop_price"  : as above.
            "stop_dir"    : as above.
            "expiration"  : as above.
          }
      }.

    Example of filled market sell:
      {
        "order":
          {
            "order_id": "adeb2ea8-2071-475d-9355-8dec46744548",
            "product_id": "BTC-USD",
            "user_id": "0a38504c-1c25-545d-90a9-0984fe477e18",
            "order_configuration": {"market_market_ioc": {"base_size": 0.001}},
            "side": "SELL",
            "client_order_id": "5d1b2837-e481-4683-9936-a65416c11b83",
            "status": "FILLED",
            "time_in_force": "IMMEDIATE_OR_CANCEL",
            "created_time": "2023-02-03T15:46:38.458628Z",
            "completion_percentage": 100.0,
            "filled_size": 0.001,
            "average_filled_price": 23654.9165021,
            "fee": '',
            "number_of_fills": 2,
            "filled_value": 23.6549165021,
            "pending_cancel": False,
            "size_in_quote": False,
            "total_fees": 0.0946196660084,
            "size_inclusive_of_fees": False,
            "total_value_after_fees": 23.5602968360916,
            "trigger_status": "INVALID_ORDER_TYPE",
            "order_type": "MARKET",
            "reject_reason": "REJECT_REASON_UNSPECIFIED",
            "settled": True,
            "product_type": "SPOT",
            "reject_message": '',
            "cancel_message": '',
            "order_placement_source": "UNKNOWN_PLACEMENT_SOURCE"
          }
      }

    Example of filled market buy:
      {
        "order":
          {
            "order_id": "f0e6af8f-1673-4a6c-99ce-e4d21d17dc09",
            "product_id": "BTC-USD",
            "user_id": "0a38504c-1c25-545d-90a9-0984fe477e18",
            "order_configuration": {"market_market_ioc": {"quote_size": 23.56}},
            "side": "BUY",
            "client_order_id": "70bf46b7-2d63-4725-bff5-210c3d4be82e",
            "status": "FILLED",
            "time_in_force": "IMMEDIATE_OR_CANCEL",
            "created_time": "2023-02-03T16:07:51.942070Z",
            "completion_percentage": 100,
            "filled_size": 0.0009936175055444,  # how much we got, minus fees
            "average_filled_price": 23616.870000000963,
            "fee": '',
            "number_of_fills": 3,
            "filled_value": 23.46613545816733,
            "pending_cancel": False,
            "size_in_quote": True,
            "total_fees": 0.0938645418326693,
            "size_inclusive_of_fees": True,
            "total_value_after_fees": 23.56,
            "trigger_status": "INVALID_ORDER_TYPE",
            "order_type": "MARKET",
            "reject_reason": "REJECT_REASON_UNSPECIFIED",
            "settled": True,
            "product_type": "SPOT",
            "reject_message": '',
            "cancel_message": "Internal error",
            "order_placement_source": "UNKNOWN_PLACEMENT_SOURCE"
          }
      }

    Example of filled limit sell:
      {
        "order":
          {
            "order_id": "9caf94f0-2b3e-4603-93c9-a39959e38791",
            "product_id": "BTC-USD",
            "user_id": "0a38504c-1c25-545d-90a9-0984fe477e18",
            "order_configuration": {"limit_limit_gtc": {"base_size": 0.001, "limit_price": 23630, "post_only": True}},
            "side": "SELL",
            "client_order_id": "f4e3ed45-a7e7-4730-93a4-56f9b4c1a446",
            "status": "FILLED",
            "time_in_force": "GOOD_UNTIL_CANCELLED",
            "created_time": "2023-02-03T16:19:02.209172Z",
            "completion_percentage": 100.0,
            "filled_size": 0.001,
            "average_filled_price": 23630,
            "fee": '',
            "number_of_fills": 1,
            "filled_value": 23.63,
            "pending_cancel": False,
            "size_in_quote": False,
            "total_fees": 0.059075,
            "size_inclusive_of_fees": False,
            "total_value_after_fees": 23.570925,
            "trigger_status": "INVALID_ORDER_TYPE",
            "order_type": "LIMIT",
            "reject_reason": "REJECT_REASON_UNSPECIFIED",
            "settled": True,
            "product_type": "SPOT",
            "reject_message": '',
            "cancel_message": '',
            "order_placement_source": "UNKNOWN_PLACEMENT_SOURCE"
          }
      }

    Example of filled limit buy
      {
        "order":
          {
            "order_id": "25851da9-5cf2-4938-aba3-4816f03da4c5",
            "product_id": "BTC-USD",
            "user_id": "0a38504c-1c25-545d-90a9-0984fe477e18",
            "order_configuration": {"limit_limit_gtc": {"base_size": 0.000972, "limit_price": 23660, "post_only": False}},
            "side": "BUY",
            "client_order_id": "efeb779a-300c-4073-8628-a2a2c84c91de",
            "status": "FILLED",
            "time_in_force": "GOOD_UNTIL_CANCELLED",
            "created_time": "2023-02-03T16:27:58.756184Z",
            "completion_percentage": 100.0,
            "filled_size": 0.000972,
            "average_filled_price": 23656.63,
            "fee": '',
            "number_of_fills": 1,
            "filled_value": 22.99424436,
            "pending_cancel": False,
            "size_in_quote": False,
            "total_fees": 0.09197697744,
            "size_inclusive_of_fees": False,
            "total_value_after_fees": 23.08622133744,
            "trigger_status": "INVALID_ORDER_TYPE",
            "order_type": "LIMIT",
            "reject_reason": "REJECT_REASON_UNSPECIFIED",
            "settled": True,
            "product_type": "SPOT",
            "reject_message": '',
            "cancel_message": '',
            "order_placement_source": "UNKNOWN_PLACEMENT_SOURCE"
          }
      }

    Example of unfilled limit sell
      {
        "order":
          {
            "order_id": "4a4b8580-eabb-40ac-b72a-bef7bec92507",
            "product_id": "BTC-USD",
            "user_id": "0a38504c-1c25-545d-90a9-0984fe477e18",
            "order_configuration": {"limit_limit_gtc": {"base_size": 0.001, "limit_price": 24000, "post_only": 0}},
            "side": "SELL",
            "client_order_id": "efaeb219-5439-4ba4-98d8-2e8edc6deed8",
            "status": "OPEN",
            "time_in_force": "GOOD_UNTIL_CANCELLED",
            "created_time": "2023-02-03T16:35:19.552610Z",
            "completion_percentage": 0,
            "filled_size": 0,
            "average_filled_price": 0,
            "fee": '',
            "number_of_fills": 0,
            "filled_value": 0,
            "pending_cancel": False,
            "size_in_quote": False,
            "total_fees": 0,
            "size_inclusive_of_fees": False,
            "total_value_after_fees": 0,
            "trigger_status": "INVALID_ORDER_TYPE",
            "order_type": "LIMIT",
            "reject_reason": "REJECT_REASON_UNSPECIFIED",
            "settled": False,
            "product_type": "SPOT",
            "reject_message": '',
            "cancel_message": '',
            "order_placement_source": "UNKNOWN_PLACEMENT_SOURCE"
          }
      }

    Example of cancelled limit sell
      {
        "order":
          {
            "order_id": "4a4b8580-eabb-40ac-b72a-bef7bec92507",
            "product_id": "BTC-USD",
            "user_id": "0a38504c-1c25-545d-90a9-0984fe477e18",
            "order_configuration": {"limit_limit_gtc": {"base_size": 0.001, "limit_price": 24000, "post_only": False}},
            "side": "SELL",
            "client_order_id": "efaeb219-5439-4ba4-98d8-2e8edc6deed8",
            "status": "CANCELLED",
            "time_in_force": "GOOD_UNTIL_CANCELLED",
            "created_time": "2023-02-03T16:35:19.552610Z",
            "completion_percentage": 0,
            "filled_size": 0,
            "average_filled_price": 0,
            "fee": '',
            "number_of_fills": 0,
            "filled_value": 0,
            "pending_cancel": False,
            "size_in_quote": False,
            "total_fees": 0,
            "size_inclusive_of_fees": False,
            "total_value_after_fees": 0,
            "trigger_status": "INVALID_ORDER_TYPE",
            "order_type": "LIMIT",
            "reject_reason": "REJECT_REASON_UNSPECIFIED",
            "settled": False,
            "product_type": "SPOT",
            "reject_message": '',
            "cancel_message": "User requested cancel",
            "order_placement_source": "UNKNOWN_PLACEMENT_SOURCE"
          }
      }

    Example of expired limit sell
      {
        "order":
          {
            "order_id": "cc693231-a2dd-499b-bd45-501cbbe370a2",
            "product_id": "BTC-USD",
            "user_id": "0a38504c-1c25-545d-90a9-0984fe477e18",
            "order_configuration": {"limit_limit_gtd": {"base_size": 0.001, "limit_price": 23400, "end_time": "2023-02-03T20:12:15Z", "post_only": False}},
            "side": "SELL",
            "client_order_id": "8d127161-b381-4ebf-84fa-87f1b2581b6d",
            "status": "EXPIRED",
            "time_in_force": "GOOD_UNTIL_DATE_TIME",
            "created_time": "2023-02-03T20:07:15.834688Z",
            "completion_percentage": 0,
            "filled_size": 0,
            "average_filled_price": 0,
            "fee": '',
            "number_of_fills": 0,
            "filled_value": 0,
            "pending_cancel": False,
            "size_in_quote": False,
            "total_fees": 0,
            "size_inclusive_of_fees": False,
            "total_value_after_fees": 0,
            "trigger_status": "INVALID_ORDER_TYPE",
            "order_type": "LIMIT",
            "reject_reason": "REJECT_REASON_UNSPECIFIED",
            "settled": False,
            "product_type": "SPOT",
            "reject_message": '',
            "cancel_message": "Order expired",
            "order_placement_source": "UNKNOWN_PLACEMENT_SOURCE"
          }
      }
    """
    try:
      if order_id is None:
        raise ValueError("order_id is None")
      reply = self._send_message("get", f"orders/historical/{order_id}")
      if self._reply_is_error(reply) or "order" not in reply.keys():
        raise ValueError(f"received error reply\n  {reply=}")
      order_info = reply["order"]
    except Exception as err:
      print(
        f"ERROR {datetime_iso()}, {func_name()}: could not get order {order_id} due "
        f"to exception\n  {err}."
      )
      return {}, locals().get("reply", {})

    order_side = order_info["side"].lower()
    if order_side == "buy":
      mult_base, mult_quote = 1, -1
    elif order_side == "sell":
      mult_base, mult_quote = -1, 1

    order_std = {
      "order_id"      : order_info["order_id"],
      "status"        : order_info["status"].lower(),
      "settled"       : order_info["settled"],
      "perc_complete" : order_info["completion_percentage"],
      "base_delta"    : mult_base * order_info["filled_size"],
      "quote_delta"   : mult_quote * (order_info["filled_value"] - order_info["total_fees"]),
      "fees"          : order_info["total_fees"],
      "symbol_pair"   : order_info["product_id"].upper(),
    }

    # Process order_configuration dictionary into the order key for order_std.
    order = {"symbol_pair": order_std["symbol_pair"], "side": order_side}
    order_key, order_conf = next(iter(order_info["order_configuration"].items()))
    order_key = order_key.lower()
    if order_key == "market_market_ioc":  # order type
      order["type"] = "market"
    elif order_key.startswith("limit_limit_"):
      order["type"] = "limit"
    elif order_key.startswith("stop_limit_stop_limit_"):
      order["type"] = "stop-limit"
    elif order_key.startswith("trigger_bracket_"):
      order["type"] = "bracket"
    else:
      print(
        f"WARNING {datetime_iso()}, {func_name()}: unexpected order type in "
        f"order\n  {order}."
      )
      order["type"] = "unknown"

    # order base_size, quote_size, and stop_size
    order.update({k: order_conf[k] for k in ["base_size", "quote_size", "stop_price", "stop_trigger_price"] if k in order_conf.keys()})
    if "limit_price" in order_conf.keys():  # order price
      order["price"] = order_conf["limit_price"]
    if "stop_direction" in order_conf.keys():  # order stop_direction
      order["stop_dir"] = order_conf["stop_direction"].split('_')[-1].lower()
    if "end_time" in order_conf.keys():  # order expiration
      order["expiration"] = order_conf["end_time"][:-1]

    order_std["order"] = order

    return order_std, reply
  #--------------------------------------------------------------------------}}}
  def order_cancel(self, order_ids):  # {{{
    """
    Cancel an order. $order_ids can be either a single ID or a list. Returns a
    dictionary of the form
      { $order_id:
          {
            "active"     : True or False,
            "cancelled"  : True or False,
            "order_info" : $order_info,
            "replies"    : $replies,
          },
        ...
      },
    where $order_id is drawn from $order_ids, $order_info is from order_info(), and
    $replies is an array of replies recieved from the server during the various
    requests.

    An order is considered to have been successfully deactivated if one of the
    following holds:
    - the cancellation request is successful,
    - $order_info["status"] != "open",
    - the exchange cannot find an order with $order_id.
    It is possible to have a non-active order that has also failed to be cancelled
    (e.g. if it is not open).

    Should only raise an exception for user error.

    A successful cancellation request generates a reply of the form
      {
        "results":
          [
            {
              "success": True,
              "failure_reason": "UNKNOWN_CANCEL_FAILURE_REASON",
              "order_id": "4a4b8580-eabb-40ac-b72a-bef7bec92507"
            }
          ]
      }
    """
    if isinstance(order_ids, str):
      order_ids = [order_ids]
    if not order_ids:
      return {}

    try:
      reply = self._send_message("post", "orders/batch_cancel", json={"order_ids": order_ids})
      if self._reply_is_error(reply) or "results" not in reply:
        raise ValueError(f"received reply error {reply=}")
    except Exception as err:
      print(
        f"ERROR {datetime_iso()}, {func_name()}: could not cancel orders "
        f"{order_ids} due to exception\n  {err}."
      )
      reply = locals().get("reply", {})

    ret = {o_id: {"active": True, "cancelled": False, "order_info": {}, "replies": []} for o_id in order_ids}
    for res in reply.get("results", []):
      o_id = res["order_id"]
      if res["success"] == True:  # if the cancellation request was successful
        ret[o_id]["active"] = False
        ret[o_id]["cancelled"] = True
      ret[o_id]["replies"].append(res)

    for o_id in ret.keys():
      order_info, reply = self.order_info(o_id)
      ret[o_id]["order_info"] = order_info
      ret[o_id]["replies"].append(reply)
      if str_dict(reply).lower().find("not found") != -1:  # the exchange cannot find the order
        ret[o_id]["active"] = False
      elif "status" in order_info.keys() and order_info["status"] != "open":  # if the order is not open
        ret[o_id]["active"] = False

    return ret
  #--------------------------------------------------------------------------}}}
#----------------------------------------------------------------------------}}}1
