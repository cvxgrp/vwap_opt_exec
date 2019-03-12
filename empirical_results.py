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

from volume_estimation import VolumeEstimatorStatic, VolumeEstimatorLogNormal
from volume_prediction import VolumePredictorMultiLognormal
from solution_SHDP import SHDPSolution, StaticSolution


def rolling_simulator(dataset, lambdas, estimation_window_len, day_analyzer,
                      volume_model_meta_params,
                      num_days_to_test=None, debug=True):
    """Perform the rolling simulation on market data.

    Given a dataset with a "Day" column, 
    estimate volume model on estimation_window_len days, analyze
    the following day using the estimated parameters, and repeat.
    If num_days_to_test is set to None we repeat until the end 
    of the dataset, otherwise we stop after that number of estimations.
    """
    days = dataset.Day.unique()
    if len(days) <= estimation_window_len:
        raise Exception("The dataset provided has too few days.")

    start = estimation_window_len
    end = 0.
    if num_days_to_test == None:
        end = len(days)
    else:
        end = start + num_days_to_test

    for i in range(start, end):
        logger.info("Working on testing day" + days[i])
        training_days = days[i - estimation_window_len:i]
        training_set = dataset[dataset.Day.isin(training_days)]
        logger.info("Using as training set: " + str(training_days))

        logger.info("Fitting static volume model..")
        fitter = VolumeEstimatorStatic()
        static_model_params = fitter.fit(training_set)

        logger.info("Fitting log-normal volume model..")
        fitter = VolumeEstimatorLogNormal(volume_model_meta_params)
        lognormal_model_parameters = fitter.fit(training_set)
        logger.info("Building volume predictor..")
        predictor = VolumePredictorMultiLognormal(lognormal_model_parameters)

        # work on each symbol separately
        for symbol in dataset.Symbol.unique():
            logger.info("Processing %s - %s" % (symbol, days[i]))
            indexer = dataset[(dataset.Day == days[i]) &
                              (dataset.Symbol == symbol)].index
            analyze_day(lambdas, dataset, indexer,
                        symbol, static_model_params, predictor)


def analyze_day(lambdas, dataset, indexer, symbol,
                static_model_params, predictor):
    """Given an index object that selects
    a day, perform operations and update the dataset."""
    if len(indexer) == 0:
        logger.error("Data for this day missing."),
        return
    day_volumes = dataset.loc[indexer].Volume.values
    prices = dataset.loc[indexer].Price.values
    T = len(day_volumes)

    # choose C as 1% of the ADV for stock
    C = static_model_params['ADVs'][symbol] * 0.01

    # fix spread at 1pip
    s_t = np.ones(T) * 0.0001

    # get sigmas historically
    sigma_t = static_model_params['sigmas']

    # fix alpha such that 10% participation rate corresponds to half LO half MO
    alpha = 10.

    # static model solution
    dataset.loc[indexer, 'StaticSol'] =\
        StaticSolution(C, 0., static_model_params[
                       'M_t'], s_t, np.ones(T), sigma_t)

    dynamic_sols = dict([(el, np.zeros(T)) for el in lambdas])
    solver = SHDPSolution(s_t, C, alpha, sigma_t)  # , predictor)
    logger.debug("Running dynamic solution for %d values of lambda." %
                 len(lambdas))
    for t in range(T):
        logger.debug("Predicting volumes at time %d" % t)
        prediction_result = predictor.predict(day_volumes[:t], symbol)
        for lambda_var in lambdas:
            logger.debug("Solving SHDP for lambda = %f" % lambda_var)
            dynamic_sols[lambda_var][t] = solver.choose_action(dynamic_sols[lambda_var][:t],
                                                               day_volumes[:t], lambda_var, prediction_result)
    for lambda_var in lambdas:
        dataset.loc[indexer, "DynamicSol%.2e" %
                    lambda_var] = dynamic_sols[lambda_var]


def aggregate_results(dataset, alpha, spread):
    """Compute slippages and transaction costs from trading schedules.

    Read a dataset with columns for the solutions (trading schedules),
    computes Sigma~, cost(u_t) and returns a dataset."""

    solutions = ['StaticSol'] + \
        [el for el in dataset.columns if el[:7] == 'Dynamic']
    result_DF = pd.DataFrame(columns=['Symbol', 'Day'] +
                             ['Stilde_' + el for el in solutions] +
                             ['Cost_' + el for el in solutions])

    test_days = set(dataset.Day.unique()).intersection(ALL_DAYS)
    logger.info('test days' + str(test_days))

    for day in test_days:
        logger.info('Processing day %s' % day)
        for symbol in ALL_SYMBOLS:
            sample_day = dataset[(dataset.Day == day) &
                                 (dataset.Symbol == symbol)]
            if len(sample_day) == 0:
                continue
            # compute stuff
            C = sum(sample_day.StaticSol)
            V = sum(sample_day.Volume)
            market_VWAP = sum(sample_day.Volume * sample_day.Price) / V
            Stildes = np.zeros(len(solutions))
            Costs = np.zeros(len(solutions))
            for i, solution in enumerate(solutions):
                Costs[i] = -sum(spread * sample_day[solution]) / (2. * C) + \
                    sum(spread * (sample_day[solution]**2 /
                                  sample_day.Volume)) * alpha / (2. * C)
                solution_VWAP = sum(
                    sample_day[solution] * sample_day.Price) / C
                Stildes[i] = ((solution_VWAP - market_VWAP) /
                              market_VWAP) + Costs[i]
            result_DF.loc[len(result_DF)] = [symbol, day] + \
                list(Stildes) + list(Costs)

    return result_DF

if __name__ == '__main__':
    from functions import load_data
    import gzip
    import cPickle as pickle
    import sys

    # constants
    W = 20
    f = 1

    def usage():
        print "Usage:\n%s [cross_val|simulation]" % sys.argv[0]

    if len(sys.argv) < 2:
        usage()

    if sys.argv[1] == 'cross_val':

        CV_LEN = 10
        total_df = load_data(ALL_DAYS[:W + CV_LEN], ignore_auctions=True,
                             correct_zero_volumes=True, num_minutes_interval=1)

        lambdas = [np.inf]
        for b in range(1, 10):

            rolling_simulator(total_df, lambdas, W, analyze_day,
                              volume_model_meta_params={'num_factors': f, 'bandwidth': b})

            total_df['DynamicSolinf_b_%d' % b] = total_df.DynamicSolinf

        fname = SAVEFOLDER + 'result_cross_validation_b.pickle'
        logger.info("\ngzipping and pickling results -> %s" % fname)
        print fname
        test_days = total_df.Day.unique()[W:]
        tested_dataset = total_df[total_df.Day.isin(test_days)]
        with gzip.open(fname, 'w') as f:
            pickle.dump(tested_dataset, f)

    elif sys.argv[1] == 'simulation':

        total_df = load_data(ALL_DAYS[10:], ignore_auctions=True,
                             correct_zero_volumes=True, num_minutes_interval=1)

        b = 3  # chosen via CV
        lambdas = [0., 1., 10., 100., 1000., np.inf]

        rolling_simulator(total_df, lambdas, W, analyze_day,
                          volume_model_meta_params={'num_factors': f, 'bandwidth': b})

        # extract results
        fname = SAVEFOLDER + 'result_f_%d_b_%d_W_20_days.pickle' % (f, b)
        logger.info("\ngzipping and pickling results -> %s" % fname)
        print fname
        test_days = total_df.Day.unique()[W:]
        tested_dataset = total_df[total_df.Day.isin(test_days)]
        with gzip.open(fname, 'w') as f:
            pickle.dump(tested_dataset, f)
    else:
        usage()
