# Black-Litterman code for US Equity python (BL_Equity.py)
# Copyright (c) Xuefei Liu, 2016.
#This package has the following contributes: 
#1. Get dataframe from US Equity and tranform the price series into covariance matrix and risk aversion coeff needed in the BlackLitterman inputs
#2. Quantify the subjective views in a csv file and output the view in a dataframe which makes the input P, Q, Omega needed in the BlackLitterman inputs obvious
#3. All the inputs can be calculated individually by its corresponding function 

#This package uses the code from
# Black-Litterman example code for python (hl.py)
# Copyright (c) Jay Walters, blacklitterman.org, 2012.

# python imports
import itertools

# 3rd party imports
import numpy as np
import pandas as pd
import Quandl
from scipy import linalg
from hl import blacklitterman

class Blacklitterman_Equity(object):
    """Black-Litterman model for US Equity"""
    def __init__(self, Tickers, viewfile, delta = None, tau = None, start = '2011-10-01', end = '2014-01-01'):
        '''
        @Summary:
            This function use all the functions above to calculate BlackLitterman Portfoilo's weight and return in US equity markets. 
            This is the main innovation from this package.
        @Inputs:
            Tickers  - TIicker name from Quandl Yahoo
            viewfile - .csv file of views
            start    - Start date of time series, i.e. '2010-01-01'
            end      - End date of time series, i.e. '2014-01-01'
            delta    - The risk aversion coeffficient of the market portfolio
            tau      - A measure of the uncertainty of the prior estimate of the mean returns
        @Outputs
            Er       - Posterior estimate of the mean returns
            w        - Unconstrained weights computed given the Posterior estimates
                       of the mean and covariance of returns.
            lambda   - A measure of the impact of each view on the posterior estimates.
        '''
        
        #Save parameters
        super(Blacklitterman_Equity, self).__init__()
        self.Tickers    = Tickers
        self.viewfile   = viewfile
        self.delta      = delta 
        self.tau        = tau
        self.start      = start 
        self.end        = end 

        # get price data, covariance and weight
        self.data   = self.read_data(Tickers, start, end, 'daily', 'rdiff')
        self.cov    = self.get_covariance(Tickers,start,end)
        self.weight = self.get_weight(Tickers)

        #If the user does not assign other values of tau and delta, use the one from functions above.
        if self.delta is None:
            self.delta = self.get_riskaversion()
        if self.tau is None:
            self.tau = self.get_tau(Tickers)

        #Get P and Q matrix from view dataframe
        try:
            self.view = self.get_views(viewfile,Tickers)
        except ValueError as err:
            if 'Empty view' in err.message:
                print 'Empty view'
                return
        else:
            q = self.view['value'].values.tolist()
            Q = []
            for i in q:
                Q.append([i])
            Q = np.asarray(Q)
            P = self.view[Tickers].values

            #Get Omega matrix
            tauV  = self.tau*self.cov
            Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])

            # blacklitterman result [er, w, lmbda]
            self.result = blacklitterman(self.delta,self.weight,self.cov,self.tau,P,Q,Omega,False)

    def read_data(self, Tickers, start = '2011-10-01', end = '2014-01-01', freq = 'daily', transformation='None'):
        '''
        @Summary:
            This function reads stock price time series from Quandl Yahoo database
        @Inputs:
            Tickers         - List of symbols, i.e. ['AAPL', 'MSFT'] 
            start           - Start date of time series, i.e. '2010-01-01'
            end             - End date of time series, i.e. '2014-01-01'
            freq            - Frequency of time series, i.e. 'daily','weekly','annualy'
            transformation  - 'None' means price series,'rdiff' means return series(r'[t]]=(r[t]-r[t-1])/r[t-1])
        @Outputs:
            Dataframe of price or return series according to transformation
        '''

        QuandlTickers = ['Yahoo/' + ticker + '.6' for ticker in Tickers]
        return Quandl.get(QuandlTickers, authtoken = 'yzX9L_uwzXfinLyqpUhH', \
                          trim_start = start, trim_end = end, \
                          collapse = freq, transformation = transformation, returns = 'pandas')#ADJUSTED CLOSED PRICE

    def get_covariance(self, Tickers, start, end):
        '''
        @Summary:           
            Get covariance matrix of portfolio
        @Inputs::    
            Tickers - List of symbols, i.e. ['AAPL', 'MSFT'] 
            start   - Start date of time series, i.e. '2010-01-01'
            end     - End date of time series, i.e. '2014-01-01'
        @Outputs:
            Numpy ndarray covariance matrix(N*N)
        '''
        return np.cov(self.data.transpose().fillna(0).values)*252

    def get_weight(self, Tickers):
        '''
        @Summary:
            Get the market equilibrium weights of the portfolio, using Quandl Core US Fundamentals Data(SF1) Databse
        @Inputs::    
            Tickers - List of symbols, i.e. ['AAPL', 'MSFT'] 
        @Outputs
            Numpy ndarray weight(N*0)
        '''
        #Initialization
        QuandlTickers = ['SF1/'+ ticker +'_MARKETCAP' for ticker in Tickers]
        caps          = []

        for i, ticker in enumerate(QuandlTickers):
            caps_len = len(caps)
            caps.extend(Quandl.get(ticker,authtoken = 'yzX9L_uwzXfinLyqpUhH')[-1:].values.tolist())
            while len(caps) == caps_len:
                #Due to Internet connetcion problem, sometimes we can not get all the data from www.quandl.com.
                #So we have to try multiple times if we can not get one of the market cap.(At most twice)
                caps.extend(Quandl.get(ticker,authtoken = 'yzX9L_uwzXfinLyqpUhH')[-1:].values.tolist())

        # standardize
        caps   = np.array(list(itertools.chain(*caps)))
        weight = caps / np.sum(caps)
        return weight

    def get_riskaversion(self, market = ['INDEX_GSPC'], start='1995-01-01',end='2015-01-01',risk_free = 0.):
        '''
        @Summary:
            We get the risk aversion coef from the market or benchmark portfolio. I use S&P500 by default
        @Inputs:
            Tickers   - Market Index name from Quandl Yahoo
            start     - Start date of time series, i.e. '1995-01-01'
            end       - End date of time series, i.e. '2015-01-01'
            risk_free - Risk free rate, 0 by default.
        @Outputs:
            risk aversion coeff by formula (E(R)-r)/sigma^2.
        '''
        annual_returns = self.read_data(market,start,end,'annual','rdiff').values
        #Sometimes quandl data base fail to get the data
        if np.std(annual_returns)==0:
            annual_returns = self.read_data(market,start,end,'annual','rdiff').values
        return (annual_returns.mean() - risk_free) / (np.std(annual_returns)**2)

    def get_tau(self, Tickers):
        '''
        @Summary:
            This function returns the parameter tau used in Black-Litterman model.
            I use tau=1/n according to http://www.blacklitterman.org/cookbook.html
        @Inputs:
            Tickers - Market Index name from Quandl Yahoo
        @Outputs:
            tau

        '''
        return 1.0/len(Tickers)

    def get_views(self, viewfile, Tickers):
        '''
        @Summary:
            This function reads formated view from a .csv file and Ticker name in the portfolio, then transform the "views" 
            in .csv file to dataframe of views. Actually this view is a combined P and Q matrix in Black-Litterman inputs. A sample is given.

            According to Black-Litterman formula, all the views can be classified as relative or absolute. The portfoilo can be classied
            as 'Underperformed' and 'Outperformed'. 

            In the .csv file, each row represents a piece of view; "GroupA" means Outperformed and "GroupB" means Underperformed.
            For example,
            If GroupB is blank, it means GroupA is better than any other stocks in the portfolio, which is an absolute view.
            If both Group have ticker names, then it means a relative view.
            GroupA should not be blank.
            
            Methods for specifying the values of Matrix P vary. Litterman (2003, p. 82) assigns a percentage value to the asset(s) in question. 
            Satchell and Scowcroft (2000) use an equal weighting scheme.This weighting scheme ignores the market capitalization of the assets involved in the view.
            As suggested by the implementation from http://www.blacklitterman.org/, I use the market capitalization weighting scheme.
        @Inputs:
            Tickers - Market Index name from Quandl Yahoo
            viewfile - .csv file of views
        @Output:
            Dataframe of views. Its value is a combined matrix of P and Q in BlackLitterman formula.
        '''
        #Construct an empty dataframe dq of views(k*n).k views and n stocks.
        df       = pd.read_csv(viewfile)
        Q        = np.asarray(df.Value.values.tolist())
        row_num  = df.shape[0]
        dq       = pd.DataFrame(index = range(row_num),columns = Tickers)
        for i in range(row_num):

            #By default, GroupA should be empty.
            if pd.isnull(df.ix[i,'GroupA']):
                print "Lack of input, GroupA should not be Null"
                raise ValueError('Empty view')
                return
            else:
                #Get the ticker from GroupA
                tickerA = df.ix[i,'GroupA'].split(",")
                weightsA = self.get_weight(tickerA)
                dq.ix[i,tickerA] = weightsA
            if pd.notnull(df.ix[i,'GroupB']):
                tickerB = df.ix[i,'GroupB'].split(",")
                weightsB = self.get_weight(tickerB)
                dq.ix[i,tickerB] = - weightsB
       
        dq.fillna(0, inplace=True)
        
        #Append Q value to the dataframe
        dq['value'] = Q
        return dq

    def display(self):
        '''
        @Summary:
            This fucntion displayes the two most important output: "Posterior weight" and "Posterior return" 
            of Black-Litterman portfolio in a DataFrame
        @Input:
            Tickers - TIicker name of the portfolio
            res     - Output from blacklitterman fucntion or blacklitterman_equity function
        @Output:
            Dataframe of "Posterior weight" and "Posterior return" of the tickers
        '''

        df = pd.DataFrame(index = self.Tickers,columns = ['Posterior return','Posterior weight'])
        df['Posterior return'] = self.result[0]
        df['Posterior weight'] = self.result[1]
        return df


def main():
    '''
    Sample Ticker : ['C','BAC','MSFT','TSLA','AAPL','CMG','EBAY','SBUX','TGT','WFC']
    Sample viewfile : view.csv
    Test Function: read_data,get_views,blacklitterman_equity,display
    Output: price and view dataframe, posterior return and weight dataframe.

    '''

    # create a blacklitterman Model 
    portfolio    = ['C','BAC','MSFT','TSLA','AAPL','CMG','EBAY','SBUX','TGT','WFC']
    viewFile     = 'view.csv'
    bl_equity    = Blacklitterman_Equity(portfolio, viewFile)

    # print price series
    print "\nPrice Series"
    print bl_equity.data.head()

    # print dataframe of view
    print "\nView dataframe:"
    print bl_equity.view

    # print blacklitterman result
    print "\nBlackLitterman dataframe:"
    print bl_equity.display()
    

if __name__ == "__main__":
    main()