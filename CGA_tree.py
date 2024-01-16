# %%
from scipy.optimize import minimize
import pandas as pd
import numpy as np
from scipy import optimize
from sklearn import linear_model
from sklearn.metrics import log_loss

# %%
class Ohit_logistic:
    def __init__(self,X,Y,K):
        # -----------------------#
        # X is a n*p dataframe/numpy
        # Y is a n*1 dataframe/numpy

        if type(X).__module__ == 'numpy' or type(X) == list:
            X = pd.DataFrame(X)
            X.columns = ['V'+str(i+1) for i in range(X.shape[1])]
        if type(Y).__module__ == 'numpy' or type(Y) == list:    
            Y = Y.reshape(-1)
            Y = pd.DataFrame({'Y':Y})
        ## filter problem col
        p1 = X.columns[np.where(X.apply(lambda x : len(x.unique())) == 1)].to_list()
        if len(p1)>0:
            X = X.drop(p1,axis = 1)
        ## init
        X = X.reset_index(drop = True)
        Y = Y.reset_index(drop = True)
        self.X = X 
        self.Y = Y
        self.K = K # number of iteration for each path
    def llik(self,y,G,gam):
        # y: numpy n*1; G: numpy n*d; gam: numpy d*1
        eta = G@gam
        return np.sum(eta * ((y==1) *1) - np.log(1+np.exp(eta)))

    def fd(self,y,G,gam):
        # y: numpy n*1; G: numpy n*d; gam: numpy d*1
        # return p series
        eta = G@gam
        if type(y).__module__ == 'pandas.core.frame':
            eta.columns = y.columns
        return -G.T @ ((y==1) *1) + G.T @ (1/(1+np.exp(-eta)))
    def gam_all(self,gam,Jhat_idx,p):
        gam = np.reshape(gam,-1)
        new_gam = [0 for i in range(p)]
        
        for i ,j in enumerate(Jhat_idx):
            new_gam[j] = gam[i]
        return  np.reshape(new_gam,(p,1))


    def MLE(self,y,G):
        init = [1]
        init.extend([0 for i in range(G.shape[1]-1)])
        gam0 = np.array(init)
        def loss(gamma):
            return(self.llik(y,G,gamma))
        res = minimize(loss,gam0, method = 'nelder-mead',options={'xatol': 1e-8, 'disp': True})
        return res.x
    def logistic_regression(self,X,Y):
        if X.__class__.__name__ == 'DataFrame':
            X = X.to_numpy()
        if Y.__class__.__name__ == 'DataFrame':
            Y = Y.to_numpy()

        Y = Y.reshape(-1)
        logr = linear_model.LogisticRegression()
        params = {'fit_intercept': False,
        'max_iter': 200}
        logr.set_params(**params)
        fit = logr.fit(X,Y)
        gam = fit.coef_
        logr_pred = fit.predict_proba(X)
        llk = log_loss(Y,logr_pred)
        return fit, gam,llk
    def CGA(self):
        X = self.X
        Y = self.Y
        K = self.K
        n,p = X.shape
        Jhat = ['beta0']+[None for i in range(K)]
        likelihood = [0 for i in range(K+1)]
        intercept = pd.DataFrame([[1] for i in range(n)])
        intercept.columns = ['beta0']
        
        _, gam, llk = self.logistic_regression(intercept,Y)
        likelihood[0] = llk
        intercept.index = X.index
        X_current = intercept
        X = pd.concat([intercept,X],axis = 1)
        
        # start iteration        
        for k in range(1,K+1):
            Jhat_idx = np.where(np.isin(X.columns,Jhat))[0]
            new_gam = self.gam_all(gam,Jhat_idx,X.shape[1])
            rq = np.abs(self.fd(Y,X,new_gam))
            rq = rq.to_numpy().reshape(-1)
            rq[Jhat_idx] = 0
            idx_selected = np.argsort(rq)[::-1][0]
            Jhat[k] =  X.columns[idx_selected]
            X_current = X[[i for i in Jhat if i!=None]]
            _, gam,llk =  self.logistic_regression(X_current,Y)
            likelihood[k] = llk
        
        self.Jhat = Jhat
        self.ll = likelihood
        
    def HDIC_Trim(self,c1 = 1):
        likelihood = self.ll
        Jhat = self.Jhat
        n,p = self.X.shape
        X = self.X
        ## hdic

        hdic = [likelihood[j]+c1*j*(np.log(p)/n) for j in range(len(likelihood))]
        self.hdic = hdic
        hdic_min = np.argmin(hdic)
        
        if hdic_min == 0: # choose one at least
            hdic_min = 1
        J_HDIC = Jhat[:hdic_min]

        ## trim
        if len(J_HDIC)<=2:
            J_Trim= J_HDIC
        else:
            intercept = pd.DataFrame([[1] for i in range(n)])
            intercept.columns = ['beta0']
            intercept.index = X.index
            X = pd.concat([intercept,X],axis = 1)
            X_subset = X[J_HDIC]
            _,_,basic =  self.logistic_regression(X_subset,self.Y)
            J_Trim = ['beta0']
            for j in range(len(J_HDIC)-1):
                trim_vars = J_HDIC.copy()
                trim_vars.remove(J_HDIC[j+1])
                X_trim = X_subset[trim_vars]
                _,_,llk = self.logistic_regression(X_trim,self.Y)
                if llk>basic+c1*(np.log(p)/n):
                    J_Trim.append(J_HDIC[j+1])
            if len(J_Trim) == 1:
                J_Trim.append(J_HDIC[1])
        self.J_HDIC = J_HDIC
        self.J_Trim = J_Trim
    
    def CGA_HDIC_Trim(self,c1 = 1):
        self.CGA()
        self.HDIC_Trim(c1)

    def Prediction(self,X_test,c1 = 0.5):
        
        try:
            J_Trim = self.J_Trim
        except:
            self.HDIC_Trim(c1 = c1)
            J_Trim = self.J_Trim
        X = self.X
        n,p = self.X.shape

        intercept = pd.DataFrame([[1] for i in range(n)])
        intercept.columns = ['beta0']
        intercept.index = X.index
        X = pd.concat([intercept,X],axis = 1)
        X_subset = X[J_Trim]
        fit,_,_ = self.logistic_regression(X_subset,self.Y)

        # test
        intercept = pd.DataFrame([[1] for i in range(X_test.shape[0])])
        intercept.columns = ['beta0']
        intercept.index = X_test.index
        X_test = pd.concat([intercept,X_test],axis = 1)
        X_subset_test = X_test[J_Trim]
        pred = fit.predict(X_subset_test.to_numpy())
        pred_proba = fit.predict_proba(X_subset_test.to_numpy())        
        return pd.DataFrame(pred),pd.DataFrame(pred_proba)

# %%
####################
####### MPCGA ######
####################
class MPCGA:
    def __init__(self,X,Y,K,max_set = 5,imp = 0.7,max_split =2,one_path = False):
        # -----------------------#
        # X is a n*p dataframe/numpy
        # Y is a n*1 dataframe/numpy

        if type(X).__module__ == 'numpy' or type(X) == list:
            X = pd.DataFrame(X)
            X.columns = ['V'+str(i+1) for i in range(X.shape[1])]
        if type(Y).__module__ == 'numpy' or type(Y) == list:    
            Y = Y.reshape(-1)
            Y = pd.DataFrame({'Y':Y})
        ## filter problem col
        p1 = X.columns[np.where(X.apply(lambda x : len(x.unique())) == 1)].to_list()
        if len(p1)>0:
            X = X.drop(p1,axis = 1)
        ## init
        X = X.reset_index(drop = True)
        Y = Y.reset_index(drop = True)
        self.X = X 
        self.Y = Y
        self.K = K # number of iteration for each path
        self.max_set = max_set # maximum number of candidate variable for each iteration
        self.imp = imp # if a gradient corresponding to x_t is greater than imp*max_gradient then x_t would be consider into candidate set
        self.max_split = max_split # the maximum iteration for split path
        self.one_path = one_path # if True, this ALG is CGA

    ####################
    #### likelihood ####
    ####################
    def logistic(self,X, beta):
        # """Given input data X and coeffecients beta, create the logistic form.
        # Parameters
        # ----------
        # X : nd-array, shape (n_samples, n_features)
        #     Training vector, where n_samples is the number of samples and n_features is the number of features.
        # beta : nd-array, shape (n_features)
        # Returns
        # -------
        # logistic form
        # Remarks
        # -------
        # 50 is used to avoid computation problems
        # """
        lterm = X@beta
        lterm[lterm < 50.0] = np.exp(lterm[lterm < 50.0]) / (1+np.exp(lterm[lterm < 50.0]))
        lterm[lterm >= 50.0] = 1.0
        return lterm
    def _logistic_loss(self,X,y):
        # """Given input data X and labels y, create the log-likelihood function.
        # Parameters
        # ----------
        # X : nd-array, shape (n_samples, n_features)
        #     Training vector, where n_samples is the number of samples and n_features is the number of features.
        # y : nd-array, shape (n_samples,)
        #     Target vector relative to X.
        # Returns
        # -------
        # loss : callable
        #     Logistic loss given X and y.
        # """

        n = X.shape[0]
        if type(y).__module__  == 'pandas.core.frame':
            y = y.to_numpy().reshape(-1)
        if type(X).__module__  == 'pandas.core.frame':
            X = X.to_numpy()
        

        def loss(beta):
            pterm = X @ beta
            pterm[pterm<50.0] = np.log(1+np.exp(pterm[pterm<50.0]))
            return np.sum(pterm/n) - (y @ X @ beta)/n
        return loss
    def _logistic_grad_hess(self,X, y):
        # """Given input data X and labels y, create the gradient and the Hessian of the log-likelihood function.
        # Parameters
        # ----------
        # X : nd-array, shape (n_samples, n_features)
        #     Training vector, where n_samples is the number of samples and n_features is the number of features.
        # y : nd-array, shape (n_samples,)
        #     Target vector relative to X.
        # Returns
        # -------
        # grad : callable
        #     the gradient of the logistic loss given X and y.
        # hess : callable
        #     the Hessian of the logistic loss given X and y.
        # """
        n = X.shape[0]
        if type(y).__module__  == 'pandas.core.frame':
            y = y.to_numpy().reshape(-1)
        if type(X).__module__  == 'pandas.core.frame':
            X = X.to_numpy()
        
        def grad(beta):
            pterm = self.logistic(X, beta)
            return (pterm - y) @ X / n

        def hess(beta):
            pterm = self.logistic(X, beta)
            return X.T * (pterm * (1-pterm)) @ X / n

        return grad, hess
    def _minimize(self,loss, x0, method, jac, hess, tol, options):
        res = optimize.minimize(loss, x0, method=method, jac=jac, hess=hess, tol=tol, options=options)
        return res.x, res.fun
    def fd(self,y,X,gam):
        # y: numpy n*1; G: numpy n*d; gam: numpy d*1
        # return p series

        n = X.shape[0]
        if type(y).__module__  == 'pandas.core.frame':
            y = y.to_numpy().reshape(-1)
        if type(X).__module__  == 'pandas.core.frame':
            X = X.to_numpy()
        return (self.logistic(X, gam) - y) @ X / n
    ##########################
    ####### basic fun ########
    ##########################
    def _hd_information_criterion(self,ic, loss, k, wn, n, p):
        """compute the information criterion with high dimensional penalty.
        Parameters
        ----------
        ic : str, {'HQIC', 'AIC', 'BIC'}
            The information criterion for model selection.
        loss : float
            the negative maximized value of the likelihood function of the model.
        k : int
            the number of free parameters to be estimated.
        wn : float
            the tuning parameter for the penalty term.
        n : int
            sample size.
        p : int
            the number of free parameters to be estimated.
        Returns
        -------
        val : float
            the value of the chosen information criterion given the loss and parameters.
        """
        val = 0.0
        if ic == 'HQIC':
            val = 2.0 * n * loss + 2.0 * k * wn * np.log(np.log(n)) * np.log(p)
        elif ic == 'AIC':
            val = 2.0 * n * loss + 2.0 * k * wn * np.log(p)
        elif ic == 'BIC':
            val = 2.0 * n * loss + k * wn * np.log(n) * np.log(p)
        return val

    def gam_all(self,gam,Jhat_idx,p):
        #######
        # gam : np-array (k,1)
        # Jhat_idx : list[k]
        # p : feature dimension
        # --------------
        # return 
        # new_gam : list[p] (the Jhat_idx in new_gam is gam)
        #######
        gam = np.reshape(gam,-1)
        new_gam = [0 for i in range(p)]
        
        for i ,j in enumerate(Jhat_idx):
            new_gam[j] = gam[i]
        # return  np.reshape(new_gam,(p,1))
        return  new_gam
    def replace_fun(self,Jhat,k,j):
        Jhat[k] = j
        return Jhat

    def logistic_regression(self,X,Y):
        if X.__class__.__name__ == 'DataFrame':
            X = X.to_numpy()
        if Y.__class__.__name__ == 'DataFrame':
            Y = Y.to_numpy()

        Y = Y.reshape(-1)
        logr = linear_model.LogisticRegression()
        params = {'fit_intercept': False,
        'max_iter': 200}
        logr.set_params(**params)
        fit = logr.fit(X,Y)
        gam = fit.coef_
        logr_pred = fit.predict_proba(X)
        llk = log_loss(Y,logr_pred)
        return fit, gam,llk

    def flatten_tuple(self,S,out = []):
        ##################
        # A recurisive algothm such that tuple to list
        ##################
        if S.__class__ == tuple:
            out.append(S)
        if isinstance(S,list):
            if len(S)>0:
                out = self.flatten_tuple(S[0],out)
            if len(S)>1:
                out = self.flatten_tuple(S[1:],out)
        return out

    ###############################
    ####### cut function ##########
    ###############################
    def best_cut_set(self,X_current,gam):
        ######################
        # X : (n,p) dataframe
        # X_current : (n,k) dataframe
        # Y : (n) dataframe
        # gam : k numpy
        # return the best set of cut vector
        #####################
        X = self.X
        colname = X.columns.to_list()
        n,p = X.shape

        X = X.to_numpy()

        # ----------------
        # sort x and get its corrosponding index
        # end point is the last index of each value in x
        # ---------------
        try:
            X_argsort = self.X_argsort
            end_point = self.end_point
            X_sort = self.X_sort
        except:
            X_argsort = np.argsort(X,axis = 0)
            X_sort = np.take_along_axis(X,X_argsort,axis = 0)
            X_sort = pd.DataFrame(X_sort)

            # first row is unique value / second row is unique index for the first element
            dt = X_sort.apply(np.unique,0,return_index = True)
            end_point = dt.loc[1,:].apply(lambda x:x[1:]-1)
            X_sort = X_sort.to_numpy()
            self.X_sort = X_sort
            self.end_point = end_point
            self.X_argsort = X_argsort

        

        # get max gradient
        Y = self.Y
        Y = Y.to_numpy()
        Y = np.reshape(Y,(n,1))
        add_part = 1/(1+np.exp(-X_current@gam))
        if add_part.__class__.__name__ == 'DataFrame' or add_part.__class__.__name__ == 'Series':
            add_part = add_part.to_numpy()
        
        
        add_part = np.reshape(add_part,(n,1))

        fd_cut = -Y+add_part
        fd_sum = np.sum(fd_cut)

        
        # expand gradient from (n,1) to (n,p) and each column is same
        fd_cut = np.reshape(fd_cut,(n,1))
        fd_cut = np.repeat(fd_cut,p,axis = 1)

        # sort gradient by index of sort for each x 
        fd_cut_sort = np.take_along_axis(fd_cut,X_argsort,axis = 0)

        # cumsum of sorted gradient
        fd_cut_sort_cumsum = np.cumsum(fd_cut_sort,axis = 0)
        fd_cut_sort_cumsum = np.maximum(abs(fd_cut_sort_cumsum),abs(fd_sum - fd_cut_sort_cumsum))

        # report cut_vector for each variable and each max cumsum of sorted gradient
        def process(i,end):
            max_idx = np.argmax(fd_cut_sort_cumsum[end,i]) 
            colname[i] = colname[i]+'_cut'+str(max_idx)
            max_value = fd_cut_sort_cumsum[max_idx,i]
            out = (X[:,i]>X_sort[max_idx,i]).astype('int')
            return list(out),max_value
        output = [process(i,end) for i,end in enumerate(end_point)]
        cut_vector = [i[0] for i in output]
        cut_vector = np.array(cut_vector).T
        max_value = [i[1] for i in output]
        
        # filter cut_vector by self-defined self.imp and max_set
        max_max_value = max(max_value)
        usd_num = sum(max_value>max_max_value*self.imp)
        if usd_num > self.max_set:
            set_usd = np.argsort(max_value)[::-1][:self.max_set].tolist()
            cut_vector_max = cut_vector[:,set_usd]
            cut_vector_max = pd.DataFrame(cut_vector_max)
            cut_vector_max.columns = [colname[i] for i in set_usd]
        else:
            cut_vector_max = cut_vector
            cut_vector_max = pd.DataFrame(cut_vector_max)
            cut_vector_max.columns = colname
        return cut_vector_max
    def cut_set(self,names,X_test):
        ##################
        # generate cut_vector depending on X_test by names
        ##################
        X = self.X
        for i, name in enumerate(names):
            try:
                name_split = name.split('_cut')
                variable = name_split[0]
                max_idx = int(name_split[1])
            except:
                continue
            
            try:
                v_test = X_test[variable]
            except:
                raise ValueError('X_test need to contain {}'.format(variable))
            v = X[variable]
            max_value = v.sort_values()[int(max_idx)]
            out = (v_test>max_value).astype('int')
            out = pd.DataFrame({name:out})
            try:
                output = pd.concat([output,out],axis = 1)
            except:
                output = out
        
        intercept = pd.DataFrame({'beta0':[1 for _ in range(X_test.shape[0])]})
        intercept.index = X_test.index
        try:
            output = pd.concat([intercept,output],axis = 1)
        except:
            output = intercept
        output.index = X_test.index
        output = pd.concat([output,X_test],axis = 1)
        return output

    ###########################
    ####### ALG　start ########
    ###########################
    def CGA_tree(self,X_current,gam,likelihood,hdic_cga,Jhat,k, wn = 1, method='dogleg', tol=1e-8, options=None,ic = 'AIC'):
        # X : dataframe n*p origin variables
        # X_curent : dataframe n*d used variables
        # Y : dataframe n*1
        # gam : numpy current para
        # likelihood : numpy current llik
        # Jhat : numpy selected set
        # K : max iteration
        # k : current interation
        # max_set : max number of path in each iteration
        # imp : rate of important
        # max_split : max iteration used to expand path
        #--------------------------------------------------
        
        K = self.K
        X = self.X
        n,p = X.shape
        Y = self.Y
        Jhat[(k+1):] = np.repeat(None,K-k)

        # get cut_vector
        Jhat_idx =np.where(np.isin(X_current.columns,Jhat ))[0]
        new_gam = self.gam_all(gam,Jhat_idx,X_current.shape[1])
        X_new = self.best_cut_set(X_current,new_gam)        
        X_new = (X_new-X_new.mean())/X_new.std()
        X_new = pd.concat([X_current,X_new],axis = 1)
        X_new = pd.concat([X_new,self.X],axis = 1)       
        X_new = X_new.T.drop_duplicates(keep = 'first').T
        
        # find potential path
        ## maximum gradient
        Jhat_idx = np.where(np.isin(X_new.columns,Jhat))[0]
        new_gam = self.gam_all(gam,Jhat_idx,X_new.shape[1])
        rq = np.abs(self.fd(Y,X_new,new_gam))
        rq = rq.reshape(-1)
        

        ## truncate max set       
        rq[Jhat_idx] = 0
        if k<=self.max_split:
            max_rq = rq.max()
            approx_max_rq_idx = np.argsort(rq)[::-1][:min(self.max_set,len(rq))]
            idx_usd = rq[approx_max_rq_idx]>self.imp*max_rq
            potential_cols = X_new.columns[approx_max_rq_idx[np.where(idx_usd)]]     
            potential_cols = potential_cols.tolist()
        else:
            potential_cols = [X_new.columns[np.argsort(rq)[::-1][0]]]
    
        # next iteration
        k = k+1
        output = []
        for potential_col in potential_cols:
            newJ = Jhat[:k]+[potential_col]
            newJ = sorted(newJ)
            for vname in newJ:
                try:
                    pathName = pathName + '+' + vname
                except:
                    pathName = vname
            if pathName in self.exist_path:
                del pathName
                continue
            else:
                self.exist_path.append(pathName)
                del pathName           

            X_current = X_new.loc[:,newJ]
            loss_cga = self._logistic_loss(X_current, Y)
            (loss_grad_cga, loss_hess_cga) = self._logistic_grad_hess(X_current, Y)
            
            # set the initial value
            Jhat_idx = np.where(np.isin(X_current.columns,Jhat))[0]
            new_gam = self.gam_all(gam,Jhat_idx,X_current.shape[1])
            
            # use scipy.optimize.minimize to minimize the loss function given its gradient and Hessian.
            (res_x, res_fun) = self._minimize(loss_cga, new_gam, method=method, jac=loss_grad_cga, hess=loss_hess_cga, tol=tol, options=options)
            new_gam = list(res_x)

            # calculate the value of information criterion
            hdic_new = self._hd_information_criterion(ic, res_fun, k, wn, n, p)
            llk_new = self.replace_fun(likelihood.copy(),k,res_fun)
            Jhat_new = self.replace_fun(Jhat.copy(),k,potential_col)
            hdic_cga_new = self.replace_fun(hdic_cga.copy(),k,hdic_new)


            # _, gam,llk =  self.logistic_regression(X_current,Y)
            # llk_new = self.replace_fun(likelihood.copy(),k,llk)
            # Jhat_new = self.replace_fun(Jhat.copy(),k,potential_col)
            if k< K:
                output.append(self.CGA_tree(X_current,new_gam,llk_new,hdic_cga_new,Jhat_new,k, wn = wn, method=method, tol=tol, options=None,ic = ic))
            if k == K:
                # return [(self.replace_fun(Jhat.copy(),k,potential_col),self.replace_fun(likelihood.copy(),k,res_fun),self.replace_fun(hdic_cga.copy(),k,hdic_new))]
                return [(Jhat_new,llk_new,hdic_cga_new)]
        return output

    def MPCGA(self, wn = 1, method='dogleg', tol=1e-8, options=None,ic = 'AIC'):
        X = self.X
        Y = self.Y
        K = self.K
        if self.one_path:
            self.max_set = 1
            self.max_split =1

        n,p = X.shape
        Jhat = ['beta0']+[None for _ in range(K)]
        likelihood = [0 for _ in range(K+1)]
        hdic_cga = [0 for _ in range(K+1)]
        X_current = pd.DataFrame([[1] for i in range(n)])
        X_current.columns = ['beta0']
        loss_cga = self._logistic_loss(X_current.to_numpy(), Y)
        (loss_grad_cga, loss_hess_cga) = self._logistic_grad_hess(X_current.to_numpy(), Y)
        
        # set the initial value
        gam = [0]

        # use scipy.optimize.minimize to minimize the loss function given its gradient and Hessian.
        (res_x, res_fun) = self._minimize(loss_cga, gam, method=method, jac=loss_grad_cga, hess=loss_hess_cga, tol=tol, options=options)
        gam = list(res_x)
        likelihood[0] = res_fun
        
        # calculate the value of information criterion
        hdic_cga[0] = self._hd_information_criterion(ic, res_fun, 0, wn, n, p)
        X_current.index = X.index
        self.exist_path  = ['beta0']
        
        # start iteration        
        self.all_CGA = self.CGA_tree(X_current,gam,likelihood,hdic_cga,Jhat,k = 0, wn = wn, method=method, tol=tol, options=None,ic = ic)
        self.all_CGA = self.flatten_tuple(self.all_CGA,[])
        
        # flatten all paths
        path = []
        ll = []
        hdic_ = []
        
        for i in range(len(self.all_CGA)):
            path.append(self.all_CGA[i][0])
            ll.append(self.all_CGA[i][1])
            hdic_.append(self.all_CGA[i][2])
        
        self.path = path
        self.ll = ll
        self.hdic_ = hdic_
    
    def HDIC_Trim(self, wn = 1, method='dogleg', tol=1e-8, options=None,ic = 'AIC'):
        Y = self.Y
        X = self.X
        n,p = X.shape
        path = self.path
        ll = self.ll
        hdic_ = self.hdic_
        n,p = self.X.shape
        
        #hdic
        path_hdic = []
        for i in range(len(path)):
            path1 = path[i]
            likelihood = ll[i]
            hdic = hdic_[i]
            # hdic = [likelihood[j]+2*c1*j*(np.log(p)/n) for j in range(len(likelihood))]
            hdic_min = np.argmin(hdic)
            if hdic_min == 0: # choose one at least
                path_hdic.append(path1[:2])
            else:
                path_hdic.append(path1[:(hdic_min+1)])
        

        #trim
        path_trim = []
        for i in range(len(path)):
            path1 = path_hdic[i]
            if len(path1)<=2:
                path_trim.append(path1)
            else:
                k = len(path1)
                X_subset = self.cut_set(path1,self.X)
                X_current = X_subset[path1]
                (loss_grad_cga, loss_hess_cga) = self._logistic_grad_hess(X_current.to_numpy(), Y)
                loss_cga = self._logistic_loss(X_current,Y)
                (res_x, res_fun) = self._minimize(loss_cga, [0 for _ in range(k)], method=method, jac=loss_grad_cga, hess=loss_hess_cga, tol=tol, options=options)
                basic = self._hd_information_criterion(ic, res_fun, k, wn, n, p)
                trim_set = ['beta0']
                for j in range(len(path1)-1):
                    trim_vars = path1.copy()
                    trim_vars.remove(path1[j+1])
                    X_trim = X_subset[trim_vars]
                    (loss_grad_cga, loss_hess_cga) = self._logistic_grad_hess(X_trim.to_numpy(), Y)
                    loss_cga = self._logistic_loss(X_trim,Y)
                    (res_x, res_fun) = self._minimize(loss_cga, [0 for _ in range(k-1)], method=method, jac=loss_grad_cga, hess=loss_hess_cga, tol=tol, options=options)
                    llk = self._hd_information_criterion(ic, res_fun, k-1, wn, n, p)
                    if llk>basic:
                        trim_set.append(path1[j+1])
                if len(trim_set) == 1:
                    trim_set.append(path1[1])
                path_trim.append(sorted(trim_set))
                
        path_hdic = [sorted(i) for i in path_hdic]
        path_trim = sorted( path_trim)

        self.path_hdic = pd.DataFrame(path_hdic).drop_duplicates(keep = 'first')
        self.path_trim = pd.DataFrame(path_trim).drop_duplicates(keep = 'first')

    # main ALG
    def MPCGA_HDIC_Trim(self,c1 = 0.5,ic = 'AIC'):        
        self.MPCGA(wn = c1,ic = ic)
        self.HDIC_Trim(wn = c1,ic = ic)

    # predict function and return pred and prob
    def MPCGA_pred(self,X_test,c1 = 1,Trim = False,Mtrim = False,ic = 'AIC'):
        if Trim :
            try:
                path_trim = self.path_trim
            except:
                self.MPCGA_HDIC_Trim(c1 = c1,ic = ic)
                path_trim = self.path_trim
        else:
            try:
                path_trim = self.path_hdic
            except:
                self.MPCGA_HDIC_Trim(c1 = c1,ic = ic)
                path_trim = self.path_hdic
        X = self.X
        n,p = self.X.shape
        output_pred = []
        output_pred_proba = []

        # model trim
        if Mtrim :
            use_idx = []
            for i in range(path_trim.shape[0]):
                vars = path_trim.iloc[i].tolist()
                while None in vars:
                    vars.remove(None)
                X_subset = self.cut_set(vars,X)
                X_subset = X_subset[vars]
                fit,_,ll = self.logistic_regression(X_subset,self.Y)
                try:
                    if ll_min>ll:
                        ll_min = ll
                        num_min = len(vars)
                except:
                    ll_min = ll
                    num_min = len(vars)
            

        for i in range(path_trim.shape[0]):
            vars = path_trim.iloc[i].tolist()
            while None in vars:
                vars.remove(None)
            X_subset = self.cut_set(vars,X)
            X_subset = X_subset[vars]
            fit,_,ll = self.logistic_regression(X_subset,self.Y)
            if Mtrim:
                if ll>ll_min + 4* c1 * max(1,num_min-len(vars)) *(np.log(p)/n):
                    continue
                else:
                    use_idx.append(i)
            X_test_cut = self.cut_set(vars,X_test)
            pred = fit.predict(X_test_cut[vars].to_numpy())
            pred_proba = fit.predict_proba(X_test_cut[vars].to_numpy())
            output_pred.append(pred)
            output_pred_proba.append(pred_proba[:,1])
        if Mtrim:
            self.path_Mtrim = path_trim.iloc[use_idx,:]

        # choose maximum number of 0 or 1 as predict
        pred =  pd.DataFrame(output_pred).T.mode(axis = 1)
        pred_proba =  pd.DataFrame(output_pred_proba).T.mean(axis = 1)
        
        # if number of 0 equal number of 1, check max prob
        if pred.shape[1] == 2:  
            for j in range(pred.shape[0]):
                if pred.iloc[j,1] == 1:
                    pred.iloc[j,0] = (pred_proba.iloc[j]>0.5).astype('int')
        return pd.DataFrame(pred.iloc[:,0]),pd.DataFrame(pred_proba)
