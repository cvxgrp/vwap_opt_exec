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

#! /usr/bin/env python
# -*- coding: utf-8 -*-

########################################
# Process WRDS TAQ files in this format:
#
# SYMBOL,DATE,TIME,PRICE,SIZE,G127,CORR,COND,EX
# DELL,20130201,4:00:01,13.9,500,0,0,T,P
# DELL,20130201,4:00:01,13.9,500,0,0,T,P
# DELL,20130201,4:00:01,13.9,500,0,0,T,P
#
# To output this format:
#
# 9:29:00,495697,9.15
# 9:30:00,495697,9.15
# 9:31:00,69707,9.16
# ...
# 15:59:00,5900,9.13333
# 16:00:00,1232322, 9.27
#
# where the first thing is the starting time: 9:29:00 means opening auction,
# 16:00:00 closing auction and all others are continuous trading periods.
# We can set how long the periods (but I tested only T = 390).
# The opening and closing auction prices are
# well defined. For the times of continuous trading p is defined as the period VWAP
#
# All the data complexity sits in managing the "condition codes" associated to the
# trade lines. I got these information from various sources, here is the summary.
#
# G127 - should be always 0. Raise exception if not.
# CORR - keep line if 0,1,2. Discard otherwise (wrong trade reporting)
# COND:
# To keep (normal):
# “”, "@" = nothing
# F, @F = intermarket sweep
# C / CF = cash trade
# N / NF = next day clearing
# R /RF= seller // maybe all out of time
# To keep (open/close):
# Q = official open
# M = Market Center Official Close
# O / @O = opening
# 6 /@6 = closing
# Uncertain:
# W = avg price trade
# P = Prior Reference Price
# X = Cross Trade
# ‘5’/@5 = Market Center Re-Opening Trade - I observed it only on a day the stock was halted (because of CEO change announcement).
# so if I see it maybe I should exclude the day
# Discard:
# Z = sold (out of sequence)
# 4 /@4 /C4 /N4 / R4 = derivativiley priced
# T = extended hours
# V = stock option
# U = Extended Hours (Sold Out of Sequence)

# ASSUMPTIONS
# I assume that in the file there is only one MarketTradingDay active every time,
# i.e. if I observe data for some new MTD the old one can be closed and saved
# I assume that data is in temporal sequence (for continuous trading stuff)
# and only opening/close volume can get reported late.

import numpy as np
import datetime
from constants import logger, PROCESSEDDATAFOLDER

kContinousCondition = set(
    ['', '@', 'F', '@F', 'C', 'CF', 'N', 'NF', 'R', 'RF', 'Q', 'M'])
kOpenCondition = set(['O', '@O'])
kCloseCondition = set(['6', '@6'])
kInValidCondition = set(
    ['Z', '4', '@4', 'C4', 'N4', 'R4', 'T', 'V', 'U', 'P', 'W', 'X'])
kSkipDayCondition = set(['5', '@5'])

# based on http://www1.nyse.com/press/1294398514465.html
kExcludeDays = set(['20121123'])

kSaveDir = PROCESSEDDATAFOLDER


class UnknownConditionCode(Exception):
    pass


class WrongOpeningTrades(Exception):
    pass


class WrongClosingTrades(Exception):
    pass


class InvalidDay(Exception):
    pass


class ContinuousTradingInterval(object):
    """Store the volumes and prices for an interval of continous trading."""

    def __init__(self, time):
        self.time = time
        #self.volume = 0
        self.prices = []
        self.volumes = []

    def add_trade(self, volume, price):
        #self.volume += volume
        self.prices.append(price)
        self.volumes.append(volume)

    def compute_avg_price(self):
        if len(self.prices) == 0:
            return np.NaN
        return sum(np.array(self.volumes) * np.array(self.prices)) / float(sum(self.volumes))
        # return (max(self.prices) + min(self.prices) + self.prices[-1])/3.

    def closeInterval(self):
        return sum(self.volumes), self.compute_avg_price()
        # return self.volume, self.compute_avg_price()


class MarketTradingDay(object):
    """Collect data for a trading day and save it."""

    def __init__(self, symbol, day, interval_lenght=1):
        """
        Class that holds data for one trading day. 
        If collapse_open_close then the volume of the opening auction is added
        to the first continuous trading interval, and the closing auction
        to the last.
        """
        if not interval_lenght in [1, 2, 5, 10, 15, 30]:
            logger.error("unsupported interval_lenght")
            raise Exception
        self.interval_lenght = interval_lenght
        self.symbol = symbol
        self.day = day
        # add p_0 and p_closing
        self.volumes = np.zeros(390 / interval_lenght + 2)
        self.prices = np.zeros(390 / interval_lenght + 2) * \
            np.nan  # set them to NaN
        self.times_list = self.compute_times()
        self.activeTradingIntervalIndex = 1
        self.activeTradingIntervalTime = self.times_list[
            self.activeTradingIntervalIndex]
        self.activeTradingInterval = ContinuousTradingInterval(
            self.activeTradingIntervalTime)
        self.InvalidDay = False

    def compute_times(self):
        """Compute list of interval start times."""
        numint = 390 / self.interval_lenght
        base = datetime.datetime(2000, 1, 1, 9, 30, 00)
        times_list = [(base + datetime.timedelta(minutes=self.interval_lenght * x)).time()
                      for x in range(0, numint)]
        times_list = [datetime.time(9, 29, 00)] + \
            times_list + [datetime.time(16, 00, 00)]
        return times_list

    def filterTime(self, time):
        """Filter out trades that are registered before market open."""
        return (time.hour < 9 or time.hour == 9 and time.minute < 30)

    def processContinuousTrade(self, time, volume, price):
        # this should be fine because I already filter out
        # the trades before market opening
        # work on the closing ones here
        if (time.hour >= 16):
            if (time.hour == 16 and time.minute == 0 and time.second == 0):
                if self.activeTradingIntervalIndex != len(self.prices) - 2:
                    # it should be like this
                    raise WrongClosingTrades
                self.activeTradingInterval.add_trade(volume, price)
            return

        while not (time.hour == self.activeTradingIntervalTime.hour and
                   time.minute - self.activeTradingIntervalTime.minute < self.interval_lenght):
            # close previous interval
            totalVol, avgPrice = self.activeTradingInterval.closeInterval()
            # print "closing", self.activeTradingIntervalIndex
            self.volumes[self.activeTradingIntervalIndex] = totalVol
            self.prices[self.activeTradingIntervalIndex] = avgPrice
            # increment time
            self.activeTradingIntervalIndex += 1
            self.activeTradingIntervalTime = self.times_list[
                self.activeTradingIntervalIndex]
            self.activeTradingInterval = ContinuousTradingInterval(
                self.activeTradingIntervalTime)
        # add current trade
        self.activeTradingInterval.add_trade(volume, price)

    def processOpeningTrade(self, volume, price):
        if np.isnan(self.prices[0]):
            self.prices[0] = price
        # let's try to halt everything if all opening prices don't agree
        if price != self.prices[0]:
            #raise WrongOpeningTrades
            # no, I just keep the last one
            self.prices[0] = price
        self.volumes[0] += volume

    def processClosingTrade(self, volume, price):
        if np.isnan(self.prices[-1]):
            self.prices[-1] = price
        # let's try to halt everything if all opening prices don't agree
        if price != self.prices[-1]:
            #raise WrongClosingTrades
            # no, I just keep the last one
            self.prices[-1] = price
        self.volumes[-1] += volume

    def processLine(self, time, price, volume, corr, cond):
        """I process the raw string parsed from the CSV,
        I only need these fields (G127 and EX are useless).
        I assume time has been converted in datetime.time,
        price is a float and size an int."""
        if self.filterTime(time):
            return
        if int(corr) > 2:
            return
        if cond in kContinousCondition:
            self.processContinuousTrade(time, volume, price)
            return
        if cond in kOpenCondition:
            self.processOpeningTrade(volume, price)
            return
        if cond in kCloseCondition:
            self.processClosingTrade(volume, price)
            return
        if cond in kInValidCondition:
            return
        if cond in kSkipDayCondition:
            logger.info("%s, %s - Invalid Day" % (self.symbol, self.day))
            self.InvalidDay = True
            return
        else:
            logger.info("Unknown condition, %s" % cond)
            raise UnknownConditionCode

    def save(self):
        # close last continuous trading interval
        if self.activeTradingIntervalIndex != len(self.prices) - 2:
            # it should be like this
            raise WrongClosingTrades
        totalVol, avgPrice = self.activeTradingInterval.closeInterval()
        self.volumes[self.activeTradingIntervalIndex] = totalVol
        self.prices[self.activeTradingIntervalIndex] = avgPrice
        # save file
        filename = kSaveDir + "%s_%s_profile" % (self.symbol, self.day)
        logger.info("saving %s" % filename)
        if self.InvalidDay:
            filename += '_INVALID'
        filename += '.csv'
        with open(filename, 'w') as f:
            f.write("TIME,SIZE,PRICE\n")
            for i, time in enumerate(self.times_list):
                f.write("%s,%d,%.3f\n" %
                        (time, self.volumes[i], self.prices[i]))


def readFile(fileObj, nrows=None):
    activeSymbol = None
    activeDay = None
    activeMarketTradingDay = None
    for index, line in enumerate(fileObj):
        # print line
        symbol, day, time, price, size, G127, corr, cond, EX = line.split(',')
        if day in kExcludeDays:
            continue
        if (symbol == activeSymbol and day == activeDay):
            # try:
            activeMarketTradingDay.processLine(datetime.time(*[int(i) for i in time.split(':')]),
                                               float(price), int(size), corr, cond)
            # except Exception as e:
            #	print "We couldn't process line %s" % line
            #	raise e
        else:
            if not activeMarketTradingDay is None:
                activeMarketTradingDay.save()
            activeSymbol = symbol
            activeDay = day
            activeMarketTradingDay = MarketTradingDay(symbol, day)
        if index == nrows:
            break
    activeMarketTradingDay.save()

if __name__ == '__main__':
    from constants import *
    import gzip
    import os

    raw_data_files = [fname for fname in os.listdir(RAWDATAFOLDER)
                      if fname[-2:] == 'gz']

    for raw_data_file in raw_data_files:
        logger.info('\n\n Processing %s' % raw_data_file)
        with gzip.open(RAWDATAFOLDER + raw_data_file) as f:
            f.readline()  # skip first line
            readFile(f)
