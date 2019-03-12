"""
Copyright (C) Enzo Busseti 2014-2019.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
from constants import *


def load_data(days_used, ignore_auctions=True, correct_zero_volumes=True, num_minutes_interval=1):
    dfs = []
    logger.info("Loading data for days:" + str(days_used))
    for symbol in ALL_SYMBOLS:
        for day in days_used:
            fname = PROCESSEDDATAFOLDER + symbol + "_" + day + "_profile.csv"
            try:
                # , header= None, names=["Time", "Volume", "Price"]  )
                df = pd.read_csv(fname)
            except IOError:
                logger.warning("%s couldn't be found" % fname)
                continue
            df.columns = ["Time", "Volume", "Price"]
            df.Time = pd.to_datetime(df.Time)

            # if we're keeping auction volume we collapse open and close
            if not ignore_auctions:
                df.Volume[1] += df.Volume[0]
                df.Volume[NUM_INTERVALS_IN_FILE -
                          2] += df.Volume[NUM_INTERVALS_IN_FILE - 1]
            # in any case we just ignore it
            df = df.reindex(df.index[1:-1])

            # correct for zero volumes
            if correct_zero_volumes and min(df.Volume) == 0:
                smallest_nonzero = min(df[df.Volume > 0].Volume)
                df.loc[(df.Volume == 0), 'Volume'] = smallest_nonzero
                df.Price = df.Price.fillna(method='ffill')
                df.Price = df.Price.fillna(method='bfill')

            # collapse into number of minutes
            if not num_minutes_interval in [1, 5, 10]:
                raise Exception('Unsupported minutes interval')
            if num_minutes_interval > 1:
                # we don't compress the price yet
                df = df.set_index('Time').resample('%dmin' %
                                                   num_minutes_interval, how='sum')
                df = df.reset_index()
                del df['Price']

            df["Symbol"] = symbol
            df["Day"] = day
            dfs.append(df)
    total_df = pd.concat(dfs).reset_index()
    del total_df['index']
    total_df['LogVolume'] = np.log(total_df.Volume)
    return total_df
