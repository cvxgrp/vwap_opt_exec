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

# Idea to structure the volume model
# we have a base class that provides the
# essential logic, we use it by subclassing
# so that we can plug different volume models

import abc
import numpy as np
from constants import logger


class VolumePredictor(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, T, model_fit_parameters):
        """Provide the parameters to the volume model.
        They are estimated in advance (on data from previous days)."""
        pass

    def predict(self, observed, symbol, debug=False):
        """Compute the relevant conditional expectations.
        Given a vector of the observed market 
        volumes m_1, ..., m_t (t can be zero),
        return the conditional expected values E[m_tau|t],
        E[1/m_tau|t] for t = t+1, ... T, and E[1/V|t].

        Given E[V|t] and Var[V|t] we estimate
        E[1/V|t] ~ 1/E[V|t] + Var[V|t]/(E[V|t])^3,
        and see whether the second term is superfluous."""

        result = {}
        result['pred_mt'], result['pred_1_over_mt'], result['var_V'] = \
            self._expected_values_prediction(observed, symbol)

        result['pred_V'] = np.sum(observed) + np.sum(result['pred_mt'])
        result['pred_1_over_V'] = 1. / result['pred_V'] + \
            result['var_V'] / (result['pred_V']**3)
        result['pred_1_over_V_square'] = 1. / (result['pred_V']**2) + \
            3 * result['var_V'] / (result['pred_V']**4)

        # logger.debug("Volume prediction, second term accounts for %f" %
        #            ((result['var_V']/(result['pred_V']**3)) / result['pred_1_over_V']))
        return result

    @abc.abstractmethod
    def _expected_values_prediction(self, observed, symbol):
        """Compute the conditional expectations (volume model specific).
        Given a vector of the observed market 
        volumes m_1, ..., m_t (t can be zero),
        return the conditional expected values E[m_tau|t],
        E[1/m_tau|t] for t = t+1, ... T, and Var[V|t].
        """
        pass


class VolumePredictorSimple(VolumePredictor):

    def __init__(self, expected, ADVs):
        """Expected volumes, normalized to sum to 1."""
        self.expected = expected
        self.ADVs = ADVs

    def _expected_values_prediction(self, observed, symbol):
        """Just project the observed volumes."""
        if len(observed) == 0:
            pred_mt = self.expected * self.ADVs[symbol]
        else:
            ADV = sum(observed) / sum(self.expected[:len(observed)])
            pred_mt = self.expected[len(observed):] * ADV
        return pred_mt, 1. / pred_mt, 0.


class VolumePredictorMultiLognormal(VolumePredictor):

    def __init__(self, model_fit_parameters):
        """We feed the estimated mu, b, Sigma"""
        self.mu = model_fit_parameters['mu']
        self.T = len(self.mu)
        self.b = model_fit_parameters['b']
        self.Sigma = model_fit_parameters['Sigma']
        self._precomputeCovariances()

    def _precomputeCovariances(self):
        """Precompute two lists of covariances-related
        quantities. First, the marginal covariances Var[m_tau|t] for 
        tau = t, ..., T. Then, objects that will be used to compute 
        the Var(V|t). Specifically, we cache:
        exp((S_ii + S_jj)/2)(exp(S_ij) - 1)
        where S is the covariance of the (T-t) remaining elements
        for every t. Refer to  
        http://en.wikipedia.org/wiki/Log-normal_distribution#Multivariate_log-normal

        Storing these matrices, with T = 390,
        costs about 150mb (with 8bytes floats).
        """
        self.conditionalMarginalCovariances = []
        self.conditionalLogNormalCovariancesIncomplete = []
        for t in range(self.T):
            cov = np.array(self._computeConditionalCovariance(t))
            cov_diagonal = np.diagonal(cov)
            expdiagonal = np.exp(cov_diagonal / 2.)
            logcov = np.exp(cov) - 1.
            logcov = (logcov.T * expdiagonal).T * expdiagonal
            self.conditionalMarginalCovariances.append(cov_diagonal)
            self.conditionalLogNormalCovariancesIncomplete.append(logcov)

    def _computeConditionalExpectation(self, observedDisturbances):
        """Given the full covariance matrix and a vector of
        the first n observed values (can be empty), predict
        the expected value of remaining observations. 
        We assume that the unconditional means are all 0."""
        n = len(observedDisturbances)
        if n == 0:
            return np.zeros(self.T)
        C = self.Sigma[:n, :n]
        B = self.Sigma[n:, :n]
        return np.dot(B, np.linalg.solve(C, observedDisturbances)).A1

    def _computeConditionalCovariance(self, n):
        """Given the full covariance matrix and the number n
        of disturbances already observed we return the
        reduced covariance matrix."""
        if n == 0:
            return self.Sigma
        A = self.Sigma[n:, n:]
        C = self.Sigma[:n, :n]
        B = self.Sigma[n:, :n]
        return A - (B * np.linalg.inv(C) * B.T)

    def _expected_values_prediction(self, observed, symbol):
        """Compute exactly E[m_tau|t], E[1/m_tau|t] for tau = t, ..., T,
        and Var(V|t). Use multivariate log-normal formula for the covariance
        matrix.
        """
        n = len(observed)
        priorMean = self.b[symbol] + self.mu
        pred_Disturbances = self._computeConditionalExpectation(np.log(observed)
                                                                - priorMean[:n])
        mu_t = (priorMean[n:] + pred_Disturbances).values
        pred_mt = np.exp(mu_t + self.conditionalMarginalCovariances[n] / 2.)
        pred_1_over_mt = np.exp(-mu_t +
                                self.conditionalMarginalCovariances[n] / 2.)
        covariance_matrix = self.conditionalLogNormalCovariancesIncomplete[n]
        var_V = np.sum((covariance_matrix.T * np.exp(mu_t)).T * np.exp(mu_t))
        return pred_mt, pred_1_over_mt, var_V
