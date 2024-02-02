import numpy as np
from scipy import optimize
from sklearn.metrics import mean_absolute_error 

'''
Program script for master thesis in computer science, Ralf Bartho 02. Feb 2024
Please note that there are very small deviations from the code documented in the thesis. 
The parameter "reg" has been renamed to "reg_type" and "lambd" has been renamed to "reg_strength" for easier reading.
All functions have been integrated into the main class "OrdReg", so that repeating arguments like
like reg_strength or reg_type do not have to be passed through each function. 
'''


class OrdReg():
    """
  Ordinal Regression model with regularization.

  Parameters:
  - reg_strength: float, default=1.
      Strength of the regularization.
  - reg_type: {'L1', 'L2', 'ElNet'}, default='L2'
      Type of regularization.
  - max_iterations: int, default=100
      Maximum number of iterations during optimization.
  """
     
    def __init__(self, reg_strength=1., reg_type='L2', max_iterations=100):
        self.reg_strength = reg_strength
        self.reg_type = reg_type
        self.max_iterations = max_iterations
        self.classes = None
        self.num_classes = None
        self.num_features = None
        self.coef = None
        self.thresholds = None
    
    ####################################
    ######### helper functions #########
    ####################################
    
    def logistic_loss(self, x):
        '''
       Implementation of the logistic loss. Note, that:
       log(1. + np.exp(-x)) gets overflow for large negative x and
       log(1+np.exp(x)) -x gets overflow for large postive x
       --> seperate computation for postive and negative values
       
       Parameters:
        - x: array_like
            Input values.

        Returns:
        - array_like
            Computed logistic loss values.
        '''
        return np.where(x > 0, np.log(1. + np.exp(-x)), np.log(1 + np.exp(x)) - x)  

    def logistic_function(self, x):
        '''
        Implementation of the logistic funcion. Note, that:
        1. / (1 + np.exp(-x)) gets overflow for large negative x and
        np.exp(x) / (1 + np.exp(x)) gets overflow for large postive x
        --> seperate computation for postive and negative values
        
        Parameters:
        - x: array_like
            Input values.

        Returns:
        - array_like
            Computed logistic function values.
        """
        '''
        exp = np.exp(x)
        return np.where(x > 0, 1. / (1 + exp), exp / (1 + exp))


    def GLL_function(self, params, X, y, arr_help):
        """
        Computes the generalized logistic loss.

        Parameters:
        - params: array_like
            Parameters to optimize.
        - X: array_like
            Input features.
        - y: array_like
            Target values.
        - arr_help: array_like
            Matrix for thresholds computation.

        Returns:
        - float
            Computed generalized logistic loss.
        """
        beta = params[:self.num_features]
        thresholds_diff = params[self.num_features:]
        thresholds = arr_help.dot(thresholds_diff)

        linear_term = thresholds[:, np.newaxis] - X.dot(beta)
        signs = np.sign(np.arange(self.num_classes - 1)[:, np.newaxis] - y + 0.5)

        loss = self.logistic_loss(signs * linear_term)
        loss = np.sum(loss)

        if self.reg_type == 'L1':
            loss += self.reg_strength * np.sum(np.abs(beta))
        elif self.reg_type == 'L2':
            loss += self.reg_strength * np.dot(beta, beta)
        elif self.reg_type == 'ElNet':
            loss += 0.5 * self.reg_strength * (np.sum(np.abs(beta)) + np.dot(beta, beta))
        return loss

    
    def grad_GLL_function(self, params, X, y, arr_help):
        """
        Computes the gradient of the generalized logistic loss.

        Parameters:
        - params: array_like
            Parameters to optimize.
        - X: array_like
            Input features.
        - y: array_like
            Target values.
        - thresholds_matrix: array_like
            Matrix for thresholds computation.

        Returns:
        - array_like
            Computed gradient of the generalized logistic loss.
        """
        beta = params[:self.num_features]
        thresholds_diff = params[self.num_features:]
        thresholds = arr_help.dot(thresholds_diff)

        linear_term = thresholds[:, np.newaxis] - X.dot(beta)
        signs = np.sign(np.arange(self.num_classes - 1)[:, np.newaxis] - y + 0.5)

        temp = signs * self.logistic_function(-signs * linear_term)
        temp_sum0 = np.sum(temp, axis=0)
        temp_sum1 = -np.sum(temp, axis=1)

        grad_thresholds = arr_help.T.dot(temp_sum1)
        grad_beta = X.T.dot(temp_sum0)

        if self.reg_type == 'L1':
            grad_beta += self.reg_strength * (beta >= 0).astype(float)
        elif self.reg_type == 'L2':
            grad_beta += 2 * self.reg_strength * beta
        elif self.reg_type == 'ElNet':
            grad_beta += 2 * self.reg_strength * beta + self.reg_strength * (beta >= 0).astype(float)
        return np.concatenate((grad_beta, grad_thresholds), axis=0)

    def predict_method(self, X, beta, theta):
        tmp = theta[:, np.newaxis] - X.dot(beta)
        pred = np.sum(tmp < 0, axis=0).astype(int)
        return pred
    
    ####################################
    ##### sklearn like API functions ###
    ####################################
    
    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        
        Parameters:
        - X: array_like
            Training data.
        - y: array_like
            Target values.
        
        Returns:
        - self: object
            Returns self.
        """
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.num_features = X.shape[1]
        y_adjusted = y - y.min()

        arr_help = np.tril(np.ones((self.num_classes - 1, self.num_classes - 1)))

      
        params = np.zeros(self.num_features + self.num_classes - 1)
        params[self.num_features:] = np.arange(self.num_classes - 1)
      
        bounds = [(None, None)] * (self.num_features + 1) + \
                 [(0, None)] * (self.num_classes - 2)
      
        ## choise of optimization 'L-BFGS-B' method see docs: https://scipy-lectures.org/advanced/mathematical_optimization/#choosing-a-method
        result = optimize.minimize(self.GLL_function, params, method='L-BFGS-B',
                                   jac=self.grad_GLL_function, bounds=bounds,
                                   args=(X, y_adjusted, arr_help), options={'maxiter': self.max_iterations})
        
        self.coef = result.x[:self.num_features]
        self.thresholds = arr_help.dot(result.x[self.num_features:])
        return self
      

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters:
        - X: array_like
            Input features.

        Returns:
        - array_like
            Predicted values.
        """
        predictions = self.predict_method(X, self.coef, self.thresholds)
        return predictions + np.min(self.classes)
      
    def score(self, X, y_true):
        """
        Returns the mean absolute error.

        Parameters:
        - X: array_like
            Input features.
        - y_true: array_like
            True target values.

        Returns:
        - float
            Mean absolute error.
        """
        y_pred = self.predict(X)
        return mean_absolute_error(y_true, y_pred)



##########################################################################
######################## minimal working example #########################
##########################################################################

if __name__ == "__main__":
    
    X = np.array([[1,2,3,4],
                  [5,6,7,8],
                  [10,11,12,13]])
    
    
    ### y values are assumed to be integers and all classes need to have samples in the y, so y can not be y= [1,2,4] missing the 3
    y = np.array([1,2,3])
    
    
    ### Example with insufficient optimization iterations = high mean absolute error
    print('### Example with insufficient optimization iterations = high mean absolute error')
    clf = OrdReg(reg_strength=1., reg_type='L2', max_iterations=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print('Mean absolute error:' , clf.score(X,y))
    print('Predictions:', y_pred)
    print('True values:', y)
    print('')
    
    ### Example with sufficient iterations
    print('### Example with sufficient iterations')
    clf = OrdReg(reg_strength=1., reg_type='L2', max_iterations=100)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print('Mean absolute error:' , clf.score(X,y))
    print('Predictions:', y_pred)
    print('True values:', y)