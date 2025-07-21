#!/usr/bin/python3 --
__version__ = "2024-12-16"

# imports {{{1
# general
from   itertools import product
from   sys       import exit
from   time      import time
import datetime as dt
import os

# ML
import numpy  as np
import pandas as pd

# custom modules
from cb_auth      import CB_API_KEY, CB_API_PRIVATE_KEY
from coinbase     import CoinBase
from mylibs.utils import date_iso
#----------------------------------------------------------------------------}}}1

def data_update(  # {{{
  *, CB, symbol_pair, interval, data=None, time_start=None, time_end, verbose
):
  """
  Parameters of note:
  - data: preexisting data, if any.
  - time_start: if preexisting data is specified, then time_start should *NOT* be
    specified.
  """
  if data is None:
    data = pd.DataFrame()
  else:
    if time_start != None:
      raise ValueError(
        "time_start cannot be specified when preexisting data is also specified."
        f"{time_start=} {data=}"
      )
    time_start = data.index.max() + interval // 2

  # Get the new data, index it by timestamp, try to get any data missing from the
  # update, and then merge it with the existing data and reindex everything.
  data_update = CB.candles(
    symbol_pair, interval=interval, start=time_start, end=time_end, verbose=verbose
  )
  data_update = data_update.set_index("unix", verify_integrity=True)
  data_update = try_get_missing(
    CB, data_update, symbol_pair=symbol_pair, interval=interval, verbose=verbose
  )
  data = pd.concat([data_update, data])
  data["symbol"] = symbol_pair
  index_full = np.arange(data.index.min(), data.index.max() + 1, step=interval)
  data = data.reindex(index_full)
  data["date"] = pd.to_datetime(data.index, unit='s')

  return data
#----------------------------------------------------------------------------}}}
def try_get_missing(CB, data, *, symbol_pair, interval, verbose=True):  # {{{
  """
  Detect missing candles in data and try to download them. Parameters:
  - CB: the CoinBase interface.
  - data: the data.
  - verbose: whether to print status information.
  """
  # Reindex the data to see what is missing.
  data = data.sort_index()
  index_full = np.arange(data.index.min(), data.index.max() + 1, step=interval)
  data = data.reindex(index_full)
  index_na = data[data.isna().any(axis=1)].index

  if len(index_na) == 0:
    return data
  elif verbose:
    print(
      f"Missing {len(index_na)} data entries ({len(index_na) / len(data):.2%}).",
      data.loc[index_na], sep='\n'
    )

  # Find consecutive blocks of missing indicies and try to get them again. The
  # index is ordered from large to small.
  end = index_na[0]
  for i in range(len(index_na)):
    if i == len(index_na) - 1 or index_na[i] - index_na[i+1] != interval:
      start = index_na[i]
      candles = CB.candles(symbol_pair, interval=interval, start=start, end=end)
      end = index_na[i+1] if i < len(index_na)-1 else None
      if len(candles) != 0:
        candles = candles.set_index("unix", verify_integrity=True)
        data.loc[candles.index] = candles
        if verbose:
          print(f"Recovered data:\n{candles}")
      elif verbose:
        print(
          f"Candles unavailable: {start} ({pd.to_datetime(start, unit='s')})"
          f" to {end} ({pd.to_datetime(end, unit='s')})."
        )
  return data
#--------------------------------------------------------------------------}}}
def data_update_save(  # {{{
  CB, filename_read, *,
  time_end=None, data_clip=None, save_dir="./", filename_write=None, verbose=True
):
  """
  Update previously downloaded historical data from CoinBase. Returns the pair
  (filename, data).

  Parameters:
  - CB: the CoinBase interface.
  - filename_read, filename_write: filenames to read and write. For
    filename_write, the default is to write to a file of the form
    "CoinBase_{symbol_pair}_{date}.csv".
  - save_dir: directory to save to. Default is ./.
  - time_end: end time (Unix epoch). By default, just the current time.
  - data_clip: date at which to clip the data. By default don't clip.
  - verbose: whether to print status information.
  """
  # Read the existing data and get the symbol pair and candle granularity from it
  if verbose:
    print(f"Reading data from {filename_read}.")
  data = pd.read_csv(filename_read, parse_dates=["date"]).set_index("unix", verify_integrity=True).sort_index()
  symbol_pair = data.iloc[0]["symbol"]
  interval = int(data.index.to_series().diff().min())

  if time_end == None:
    time_end = int(time())
  if filename_write == None:
    filename_write = f"CoinBase_{symbol_pair}_{date_iso(time_end)}.csv"
  filename_write = save_dir + filename_write

  data = data_update(
    CB=CB, symbol_pair=symbol_pair, interval=interval, data=data, time_end=time_end,
    verbose=verbose
  )

  # Save and return.
  if data_clip != None:
    data = data.reset_index().set_index("date", verify_integrity=True, drop=False)
    data = data.sort_index().loc[data_clip:].set_index("unix")
  data = data.sort_index(ascending=False)  # data on disk has old below new
  if verbose:
    print(f"Writing data to {filename_write}.")
  data.reset_index().to_csv(filename_write, index=False)
  return filename_write, data
#----------------------------------------------------------------------------}}}
def data_download_save(  # {{{
  CB, symbol_pair, interval, *,
  time_start=None, time_end=None, save_dir="./", filename=None, verbose=True
):
  """
  Download historical data from CoinBase. Returns the pair (filename, data).

  Parameters:
  - CB: the CoinBase interface.
  - symbol_pair: the trading pair (e.g. BTC-USD).
  - interval: candle interval in seconds, must be one of 60, 300, 900, 3600,
    21600, 86400
  - time_start, time_end: times (Unix epoch) to download between. Defaults to
    time_start=0 and time_end=now.
  - filename: filename to save to. Defaults to "CoinBase_{symbol_pair}_{date}.csv".
  - save_dir: directory to save to. Default is ./.
  - verbose: whether to print status information.
  """
  symbol_pair = symbol_pair.upper()
  if time_start == None:
    time_start = 0
  if time_end == None:
    time_end = int(time())
  if filename == None:
    filename = f"CoinBase_{symbol_pair}_{date_iso(time_end)}.csv"
  filename = save_dir + filename

  data = data_update(
    CB=CB, symbol_pair=symbol_pair, interval=interval, time_start=time_start,
    time_end=time_end, verbose=verbose
  )

  # Save and return.
  data = data.sort_index(ascending=False)  # data on disk has old below new
  if verbose:
    print(f"Writing data to {filename}.")
  data.reset_index().to_csv(filename, index=False)
  return filename, data
#----------------------------------------------------------------------------}}}

if __name__ == "__main__":  # {{{1
  # argument parsing {{{
  import argparse
  parser = argparse.ArgumentParser(
   description="Download or update a CoinBase historical candle data file. Stores "
               "downloaded data in file named "
               "CoinBase_<symbol_pair>_<YYYY-MM-DD>.csv."
  )
  parser.add_argument("file", nargs='?', help="The data file to update.", metavar="FILE")
  parser.add_argument(
    "-s", "--symbol-pair", metavar="PAIR",
    help="The symbol pair for the candles (e.g. BTC-USD). Required for downloads, "
         "should not be specified for updates."
  )
  parser.add_argument(
    "-i", "--interval", type=int, choices=[60, 300, 900, 3600, 21600, 86400],
    help="The resolution of the candles in seconds. Required for downloads, should "
         "not be specified for updates. Must be 60 (minute), 300 (5 minute), 900 "
         "(15 minute), 3600 (hour), 21600 (6 hour), or 86400 (day)."
  )
  parser.add_argument("--max-age", type=float, help="Maximum number of years to download.", metavar='YEARS')
  parser.add_argument("-v", "--verbose", action="store_true", help="Print status output.")
  parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

  options = parser.parse_args()

  filename    = options.file
  symbol_pair = options.symbol_pair
  interval    = options.interval
  max_age     = options.max_age
  verbose     = options.verbose

  # no arguments given
  if filename == symbol_pair == interval == max_age == None and verbose == False:
    parser.print_help()
    exit()

  if filename == None or not os.path.isfile(filename):  # download
    if None in (symbol_pair, interval):
      parser.error("symbol pair and interval are required for initial downloads.")
  elif (symbol_pair, interval) != (None, None):  # update
    parser.error("symbol pair/interval should not be specified for updates.")

  if max_age != None:
    time_start = time() - dt.timedelta(days=max_age*365).total_seconds()
    data_clip = dt.datetime.fromtimestamp(time_start)
  else:
    time_start = data_clip = None
  # }}}

  CB = CoinBase(CB_API_KEY, CB_API_PRIVATE_KEY)

  if filename == None or not os.path.isfile(filename):  # download
    data_download_save(
      CB, symbol_pair, interval, time_start=time_start, filename=filename,
      verbose=verbose
    )
  else:  # update
    data_update_save(CB, filename, data_clip=data_clip, verbose=options.verbose)
#----------------------------------------------------------------------------}}}1
