import pandas as pd 
import numpy as np
import abc

class VolumeEstimator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, model_meta_parameters):
        pass

    @abc.abstractmethod
    def fit(self, dataset):
        """Returns dict of model parameters."""
        pass

class VolumeEstimatorStatic(VolumeEstimator):
    """Estimate volume model for the static 
    case."""
    def __init__(self, test = False):
        """No meta-parameters needed."""
        self.test = test
        return

    def fit(self, dataset):
        """Returns dict of model parameters."""
        result = {}
        day_volumes = dataset.groupby(['Symbol', 'Day']).Volume.sum()
        merged = pd.merge(dataset, day_volumes.reset_index(), 
            how='left', on=['Symbol', 'Day'], suffixes = ('', '_total'))
        merged['fractional_volume'] = merged.Volume / merged.Volume_total
        mean_frac_vol = merged.groupby(['Time']).fractional_volume.mean().values
        T = len(mean_frac_vol)
        result['M_t'] = np.concatenate([[0.], np.cumsum(mean_frac_vol)[:-1]])

        ## NO IT's NOT RIGHT 
        # if self.test:
        #     merged['inverse_volume'] = 1./merged.Volume
        #     daily_inverse_volume_sum = merged.groupby(['Symbol', 'Day']
        #         ).inverse_volume.sum()
        #     merged = pd.merge(merged, daily_inverse_volume_sum.reset_index(), 
        #         how='left', on=['Symbol', 'Day'], suffixes = ('', '_total'))
        #     merged['fractional_inverse_volume'] = merged.inverse_volume/ \
        #         merged.inverse_volume_total
        #     result['inverse_solution'] = merged.groupby(['Time']
        #         ).fractional_inverse_volume.mean().values

        ## OLD
        # merged['fractional_inverse_volume'] = 1./merged.fractional_volume
        # mean_inverse_vol = merged.groupby(['Time']).fractional_inverse_volume.median().values
        # mean_inverse_vol = T*mean_inverse_vol / sum(mean_inverse_vol) 
        # result['alpha_t'] = mean_inverse_vol

        # get ADV per symbol
        ADVs = dataset.groupby(['Symbol', 'Day']).Volume.sum().reset_index(
            ).groupby('Symbol').Volume.mean()
        result['ADVs'] = ADVs
        # compute sigmas
        normalized_price_diffs = dataset.Price.diff() / dataset.Price
        merged['price_diffs_squared'] = normalized_price_diffs**2
        sigmas = np.sqrt(merged.groupby('Time').price_diffs_squared.mean())
        sigmas[0] = np.nan
        result['sigmas'] = sigmas
        return result

class VolumeEstimatorLogNormal(VolumeEstimator):
    """Implement ad-hoc estimation of the parameters
    of the multivariate log-normal volume model."""
    def __init__(self, model_meta_parameters, test = False):
        """
        Take the model meta-parameters and the test switch.

        The meta parameters is a dict with two fields
        num_factors is the number of factors in the SVD
        bandwidth is the bandwidth of the extra part of
        the covariance.

        test is a switch to return parameters of 
        the SVD for testing/plotting.
        """
        self.num_factors = model_meta_parameters['num_factors']
        self.bandwidth = model_meta_parameters['bandwidth']
        self.test = test 

    def _computeB(self):
        b = self.training_set.groupby('Symbol').LogVolume.mean()
        self.training_set = self._removeBComponent(self.training_set, b)
        return b

    def _removeBComponent(self, dataset, b):
        dataset = pd.merge(dataset, b.reset_index(), how='left', 
            on=['Symbol'], suffixes = ('', '_mean'))
        dataset['LogVolumeDeMeaned'] = \
            dataset.LogVolume - dataset.LogVolume_mean
        del dataset['LogVolume_mean']
        return dataset

    def _computeI(self):
        I = self.training_set.groupby('Time').LogVolumeDeMeaned.mean()
        self.training_set = self._removeIComponent(self.training_set, I)
        return I

    def _removeIComponent(self, dataset, I):
        dataset = pd.merge(dataset, I.reset_index(), how='left', 
            on=['Time'], suffixes = ('', '_time'))
        dataset['Disturbances'] = \
            dataset.LogVolumeDeMeaned - dataset.LogVolumeDeMeaned_time
        del dataset['LogVolumeDeMeaned_time']
        return dataset

    def _computeSigma(self):
        train_matrix = self.training_set.set_index(
                ['Symbol', 'Day', 'Time']
                ).Disturbances.unstack().T.as_matrix()
        N = train_matrix.shape[1]
        empirical_covariance = np.dot(train_matrix, train_matrix.T)/(N-1)
        U,S,V = np.linalg.svd(train_matrix, full_matrices =0)
        if self.test:
            S_orig = np.array(S)
        S[self.num_factors:] = 0
        low_rank_covariance = (np.matrix(U) * np.matrix(np.diag(S**2)) * \
                            np.matrix(U).T)/(N-1)
        residuals_covariance = empirical_covariance - low_rank_covariance
        result = np.matrix(low_rank_covariance)
        for k in range(self.bandwidth):
            # select diagonal from empirical cov
            diagonal = np.diagonal(residuals_covariance, k)
            # add it to result
            result += np.diag(diagonal, k)
            if k > 0:
                result += np.diag(diagonal, -k)
        if self.test:
            return result, S_orig, U
        return result

    def fit(self, dataset):
        """training_set is a pandas df with a 
        Volume columns (and symbol, day, time).
        I need to get ownership of the training_set
        otherwise I'm not able to add columns."""
        self.training_set = dataset
        result = {}
        result['b'] = self._computeB()
        result['mu'] = self._computeI()
        if self.test:
            result['Sigma'], result['S_orig'], result['U'] = \
              self._computeSigma()
        else:
            result['Sigma'] = self._computeSigma()
        return result


if __name__ == "__main__":
    __TEST_STATIC = True

    from constants import *
    from functions import load_data
    import matplotlib.pyplot as plt

    total_df = load_data(ALL_DAYS[0:5], ignore_auctions = True, 
        correct_zero_volumes = True, num_minutes_interval=1)


    symbol = 'MMM'
    sample_day = total_df[(total_df.Day == '20120924') &
                      (total_df.Symbol == symbol)]

    fitter = VolumeEstimatorStatic(test=True)
    model_fit_parameters = fitter.fit(total_df)
    M_t = model_fit_parameters['M_t']

    #plot
    plt.plot([0.] + list(np.cumsum(sample_day.Volume.values)/ float(sample_day.Volume.sum())), 
        label='market')
    #plt.plot(model_fit_parameters['inverse_solution'], label='inverse_solution')

    plt.plot(list(M_t)+[1.], label='expected')
    plt.legend(loc='upper left')
    plt.show()

    plt.figure()
    plt.plot(model_fit_parameters['sigmas'])
    plt.title('Sigma')
    plt.show()
