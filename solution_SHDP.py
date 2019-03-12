import numpy as np 
import abc
# from volume_prediction import VolumePredictor

def StaticSolution(C, lambda_var, M_t, s_t, alpha_t, sigma_t):
    # CHANGE IT TO REFLECT TEXT, DON'T SHIP IT
    """Compute the static solution given the problem parameters."""
    T = len(s_t)
    if (np.all(s_t == s_t[0])): # constant spread
        return np.diff(np.concatenate([M_t,[1.]])) * C
    import cvxpy as cvx 
    u = cvx.Variable(T)
    U = cvx.Variable(T)
    objective = cvx.square(u).T*(s_t*alpha_t/2.) - np.sign(C)*u.T*s_t/2. + \
        lambda_var*(C**2)*(cvx.square(U).T*(sigma_t**2) - 
                            2 * U.T*(sigma_t**2*M_t)) 
    constraints = []
    for t in range(T):
        constraints += [U[t] == cvx.sum_entries(u[:t])/C]
    constraints += [cvx.sum_entries(u) == C]
    constraints += [C*u >= 0]
    problem = cvx.Problem(cvx.Minimize(objective), constraints)
    problem.solve()
    return u.value.A1 

class SHDPSolution(object):
    """Compute the solution using SHDP (Section 4)."""
    def __init__(self, s_t, C, alpha, sigma_t):#, volume_predictor):
        """We store the fixed values (throughout the execution)."""
        self.s_t = s_t
        self.T = len(s_t)
        self.alpha = alpha 
        self.C = C
        self.sigma_t = sigma_t
        self.rt = -np.sign(C) * self.s_t/2. / self.C #modified to use Stilde
        self.Rt = np.zeros(self.T)
        self.beta_t = np.zeros(self.T)
        self.gamma_t = np.zeros(self.T)
        self.delta_t = np.zeros(self.T)
        self.l_t = np.zeros(self.T)

    def _choose_action_lambda_infty(self, u_ts, m_ts, prediction_result):
        pred_1_over_V = prediction_result['pred_1_over_V']
        pred_m_t = prediction_result['pred_mt']
        u_t = self.C*pred_1_over_V*(pred_m_t[0] + sum(m_ts)) - sum(u_ts)
        u_t_bar = np.sign(self.C) * max(u_t *np.sign(self.C),0)
        return u_t_bar

    def choose_action(self, u_ts, m_ts, lambda_var, prediction_result):
        # TO BE REFACTORED - THIS IS THE CODE I WANT TO USE
        """Compute u_t given the observed values
        of u_tau and m_tau for tau = 1, ..., t-1.
        Apply projection on feasible set.
        """
        self.lambda_var = lambda_var
        t = len(u_ts)
        if t == self.T-1:
            return self.C - np.sum(u_ts)
        if lambda_var == np.inf:
            return self._choose_action_lambda_infty(u_ts, m_ts, prediction_result)
        x_t = np.array([np.sum(u_ts), np.sum(m_ts)])
        pred_1_over_V = prediction_result['pred_1_over_V']
        ### THIS IS A TEST
        self.Rt[t:] = self.alpha * self.s_t[t:]/2. * \
                prediction_result['pred_1_over_mt'] / self.C
        ###
        m_t = np.concatenate([m_ts, prediction_result['pred_mt']]) # for simplicity
        ##
        ## I added 1/C**2 every time lambda appears 
        ## to do the Stilde thing
        ##
        # final values
        self.beta_t[-1] = (self.lambda_var/(self.C**2))*self.sigma_t[-1]**2 + self.Rt[-1]
        self.gamma_t[-1] = -self.C * (self.lambda_var/(self.C**2)) * (self.sigma_t[-1]**2) * pred_1_over_V 
        self.delta_t[-1] = -self.rt[-1] -2*self.C*self.Rt[-1]
        self.l_t[-1] = self.C   
        # recursion
        for tau in np.arange(self.T-2,t-1,-1):
            self.l_t[tau] = -(self.rt[tau] + self.delta_t[tau+1] + 2*self.gamma_t[tau+1]*m_t[tau])/ \
                    (2*(self.Rt[tau] + self.beta_t[tau+1]))
            self.beta_t[tau] = (self.lambda_var/(self.C**2))*self.sigma_t[tau]**2 + \
                self.Rt[tau]*self.beta_t[tau+1]/(self.Rt[tau] + self.beta_t[tau+1])
            self.gamma_t[tau] =  -self.C * (self.lambda_var/(self.C**2)) * (self.sigma_t[tau]**2) * pred_1_over_V + \
                self.Rt[tau]*self.gamma_t[tau+1]/(self.Rt[tau] + self.beta_t[tau+1])
            self.delta_t[tau] = self.delta_t[tau+1] + 2*self.beta_t[tau+1]*self.l_t[tau] + 2*self.gamma_t[tau+1]*m_t[tau]
        K_t = - np.array([self.beta_t[t+1], self.gamma_t[t+1]])/(self.Rt[tau]+ self.beta_t[t+1])
        # compute action
        u_t = np.dot(K_t, x_t) + self.l_t[t]
        u_t_bar = np.sign(self.C) * max(u_t *np.sign(self.C),0)
        return u_t_bar

# class SHDPSolutionOLD(object):
#     """Compute the solution using SHDP (Section 4)."""
#     def __init__(self, lambda_var, C, s_t, alpha_t, sigma_t, volume_predictor):
#         """We store the fixed values (throughout the execution)."""
#         self.lambda_var = lambda_var
#         self.C = C
#         self.s_t = s_t
#         self.T = len(s_t)
#         self.alpha_t = alpha_t
#         self.sigma_t = sigma_t
#         if not isinstance(volume_predictor, VolumePredictor):
#             raise Exception('Must use a VolumePredictor class to predict volumes.')
#         self.volume_predictor = volume_predictor
#         self.Rt = self.alpha_t * self.s_t/2.
#         self.rt = -np.sign(C) * self.s_t/2.

#     def _build_D_and_K_and_d_and_l_given_t(self, prediction_result, t):
#         pred_1_over_V = prediction_result['pred_1_over_V']
#         pred_1_over_V_square = prediction_result['pred_1_over_V_square']
#         pred_m_t = prediction_result['pred_mt']
#         m_t = np.concatenate([[np.nan]*t, pred_m_t]) # for simplicity
#         Q_t_base = np.matrix([[1, -self.C*pred_1_over_V],
#                     [-self.C*pred_1_over_V, (self.C**2)*pred_1_over_V_square]])
#         D = np.zeros((T,2,2))
#         d = np.zeros((T,2))
#         K = np.zeros((T,2))
#         l = np.zeros(T)
#         D[-1] = Q_t_base * self.lambda_var*(self.sigma_t[-1]**2) + \
#                 self.Rt[-1] * np.matrix([[1,0],[0,0]])
#         d[-1] = [-self.rt[-1] - 2*self.C*self.Rt[-1], 0]
#         K[-1] = [-1,0] 
#         l[-1] = self.C       
#         for tau in range(T-2, t-1, -1):
#             K[tau] = - D[tau+1][:,0]/(self.Rt[tau] + D[tau+1][0,0])
#             l[tau] = - (self.rt[tau] + d[tau+1][0] + 2 * D[tau+1][0,1] * m_t[tau] ) /\
#                 (2 *self.Rt[tau] + 2 * D[tau+1][0,0])
#             D[tau] = Q_t_base * self.lambda_var*(self.sigma_t[tau]**2) + \
#                 D[tau+1] + np.dot(np.matrix(K[tau]).T, np.matrix(D[tau+1][0,:]))
#             d[tau] = d[tau+1] + 2*np.dot(np.matrix(D[tau+1]), [l[tau], m_t[tau]])
#         return D, d, K, l  

#     def choose_action(self, u_ts, m_ts, symbol):
#         """Compute u_t given the observed values
#         of u_tau and m_tau for tau = 1, ..., t-1.
#         Apply projection on feasible set.
#         """
#         t = len(u_ts)
#         #if t == self.T-1:
#         #    return self.C - np.sum(u_ts)
#         x_t = np.array([np.sum(u_ts), np.sum(m_ts)])
#         prediction_result = self.volume_predictor.predict(m_ts, symbol)
#         D, d, K, l = self._build_D_and_K_and_d_and_l_given_t(prediction_result, t)
#         u_t = np.dot(K[t], x_t) + l[t]
#         u_t_bar = np.sign(self.C) * max(u_t *np.sign(self.C),0)
#         return u_t_bar
#         #return u_t

#     def alternative_choose_action(self, u_ts, m_ts, symbol):
#         # TO BE REFACTORED - THIS IS THE CODE I WANT TO USE
#         """Compute u_t given the observed values
#         of u_tau and m_tau for tau = 1, ..., t-1.
#         Apply projection on feasible set.
#         """
#         t = len(u_ts)
#         if t == self.T-1:
#             return self.C - np.sum(u_ts)
#         x_t = np.array([np.sum(u_ts), np.sum(m_ts)])
#         # predict future volume-related quantities
#         prediction_result = self.volume_predictor.predict(m_ts, symbol)
#         pred_1_over_V = prediction_result['pred_1_over_V']
#         pred_m_t = prediction_result['pred_mt']
#         ###
#         ### THIS IS A TEST
#         ###
#         self.alpha_t[t:] = prediction_result['pred_1_over_mt']
#         self.Rt = self.alpha_t * self.s_t/2.
#         ###
#         ### 
#         ###
#         m_t = np.concatenate([m_ts, pred_m_t]) # for simplicity
#         # build variables
#         beta_t = np.zeros(self.T)
#         gamma_t = np.zeros(self.T)
#         delta_t = np.zeros(self.T)
#         l_t = np.zeros(self.T)
#         # final values
#         beta_t[-1] = self.lambda_var*self.sigma_t[-1]**2 + self.alpha_t[-1]*s_t[-1]/2.
#         gamma_t[-1] = -self.C * self.lambda_var * (self.sigma_t[-1]**2) * pred_1_over_V 
#         delta_t[-1] = -self.rt[-1] -2*self.C*self.Rt[-1]
#         l_t[-1] = self.C   
#         # recursion
#         for tau in np.arange(self.T-2,t-1,-1):
#             l_t[tau] = -(self.rt[tau] + delta_t[tau+1] + 2*gamma_t[tau+1]*m_t[tau])/ \
#                     (2*(self.Rt[tau] + beta_t[tau+1]))
#             beta_t[tau] = self.lambda_var*self.sigma_t[tau]**2 + \
#                 self.Rt[tau]*beta_t[tau+1]/(self.Rt[tau] + beta_t[tau+1])
#             gamma_t[tau] =  -self.C * self.lambda_var * (self.sigma_t[tau]**2) * pred_1_over_V + \
#                 self.Rt[tau]*gamma_t[tau+1]/(self.Rt[tau] + beta_t[tau+1])
#             delta_t[tau] = delta_t[tau+1] + 2*beta_t[tau+1]*l_t[tau] + 2*gamma_t[tau+1]*m_t[tau]
#         K_t = - np.array([beta_t[t+1], gamma_t[t+1]])/(self.Rt[tau]+ beta_t[t+1])
#         #print K_t, l_t[t]
#         # compute action
#         u_t = np.dot(K_t, x_t) + l_t[t]
#         u_t_bar = np.sign(self.C) * max(u_t *np.sign(self.C),0)
#         if np.isnan(u_t_bar):
#             raise Exception
#         #raise Exception
#         return u_t_bar
#         #return u_t

if __name__ == "__main__":
    __TEST_STATIC = False

    from constants import *
    from functions import load_data
    import matplotlib.pyplot as plt

    MODEL_META_PARAMS = {'num_factors':1, 'bandwidth':3}

    total_df = load_data(ALL_DAYS[20:50], ignore_auctions = True, 
        correct_zero_volumes = True, num_minutes_interval=1)
    test = load_data(ALL_DAYS[50:51], ignore_auctions = True, 
        correct_zero_volumes = True, num_minutes_interval=1)


    symbol = ALL_SYMBOLS[23]
    sample_day = test[(test.Symbol == symbol)]
    T = len(sample_day)

    # set constants
    #sigma_t = np.ones(T)/np.sqrt(T) # order of 1% variation per day, split over T
    #alpha_t = 1./np.zeros(T) # all nan
    #s_t = np.ones(T) / 10000. # 1pip of spread
    #C = 100. # aribtrarily fixed at 100. shares

    if not __TEST_STATIC:
        from volume_prediction import VolumePredictorMultiLognormal
        from volume_estimation import VolumeEstimatorLogNormal
        from volume_estimation import VolumeEstimatorStatic

        fitter = VolumeEstimatorStatic()
        static_model_params = fitter.fit(total_df)

        ## MARKET
        plt.plot(np.cumsum(sample_day.Volume.values)/ float(sample_day.Volume.sum()), 
            label='market')
        market_VWAP = sum(sample_day.Volume*sample_day.Price)/sum(sample_day.Volume)

        ## CONSTANTS
        # choose C as 1% of the ADV for stock
        C = static_model_params['ADVs'][symbol] * 0.01
        
        # fix spread at 1pip
        s_t = np.ones(T) * 0.0001
        
        # get sigmas historically
        sigma_t = static_model_params['sigmas']

        # fix alpha such that 10% participation rate corresponds to half LO half MO
        alpha = 10.

        def print_stats(solution):
            solution_VWAP = sum(solution*sample_day.Price)/sum(solution)
            print "VWAP square dev. %.2e"% (((solution_VWAP - market_VWAP)/market_VWAP)**2)
            print "Execution cost %.2e "% sum(alpha*(s_t/2.)*(solution)**2/(sample_day.Volume.values*C))

        ## STATIC SOLUTION
        M_t = static_model_params['M_t']
        print 'static'
        # look at static exec cost
        static_u = C*np.diff(np.concatenate([M_t, [1]]))
        print_stats(static_u)
        plt.plot(np.concatenate([M_t, [1]])[1:], label='Static')
        
        ## DYNAMIC SOLUTION
        fitter = VolumeEstimatorLogNormal(MODEL_META_PARAMS)
        model_fit_parameters = fitter.fit(total_df)
        predictor = VolumePredictorMultiLognormal(model_fit_parameters)

        lambdas = np.array([np.inf, 10000, 1., 0.])
        dynamic_sols = dict([(el, np.zeros(T)) for el in lambdas]) 
        solver = SHDPSolution(s_t, C, alpha, sigma_t)

        for t in range(T):
            prediction_result = predictor.predict(sample_day.Volume.values[:t], symbol)
            for lambda_var in lambdas:
                dynamic_sols[lambda_var][t] = solver.choose_action(dynamic_sols[lambda_var][:t], 
                    sample_day.Volume.values[:t], lambda_var, prediction_result)

        for lambda_var in lambdas:
            print 'lambda:', lambda_var
            print_stats(dynamic_sols[lambda_var])
            plt.plot(np.cumsum(dynamic_sols[lambda_var])/C, label='$\lambda$ = %e'%lambda_var)
        plt.legend(loc='upper left')
        plt.title('%s - %s'%(symbol, test.Day.unique()[0]))
        plt.show()

    if __TEST_STATIC: # PROBABLY WRONG

        from volume_estimation import VolumeEstimatorStatic

        fitter = VolumeEstimatorStatic()
        model_fit_parameters = fitter.fit(total_df)
        M_t = model_fit_parameters['M_t']
        alpha_t = model_fit_parameters['alpha_t']

        #plot
        plt.plot([0.] + list(np.cumsum(sample_day.Volume.values)/ float(sample_day.Volume.sum())), 
            label='market')

        plt.plot(list(M_t) + [1.], label='expected')

        lambdas = np.array([10., 0.01, .00001])
        #lambdas = np.array([ .001])
        u_ts = {}
        for lambda_var in lambdas:
            print lambda_var
            u_ts[lambda_var] = StaticSolution(C, lambda_var, M_t, s_t,
                                 alpha_t, sigma_t)
            plt.plot([0.] + list(np.cumsum(u_ts[lambda_var])/C), label='$\lambda$ = %f'%lambda_var)

        plt.legend()
        plt.show()







