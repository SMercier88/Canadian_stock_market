# -*- coding: utf-8 -*-
"""
Script to extract information on stocks and sectors on the canadian stock market.

We create a Stock class, with the following methods:
    get_key_stats: extract fundamental indicators on the stock
    
    analysts: extract analysts predictions on the stock
    
    rank_of_stock: calculate the rank of the stock, with respect to other stocks
                   in the same sector, for a given list of fundamental indicators
                   
    get_charts: display the stock's historical price on a candlestick chart
    
We also create a Sector class, with the following methods:
    get_tickers: identify all stocks (tickers) in a given industry
    
    get_average_best: calculate the average, median and the best value (and stock) between
                    all stocks in the industry for a given list of fundamental
                    indicators
                    
    export_csv_data: export to a csv file the fundamental indicators of all stocks
                    in a given industry
                    
    rank_all_stocks: rank all stocks, with respect to the other stocks in the same
                    industry, for a given list of fundamental indicators
                    
    identify_best_stock: identify the stock with the best average ranking with 
                        respect to all the stocks in a given industry
                        
Note that the Stock class also inherits all the methods from the Sector class
    
Note: H/T to Sentdex for the tutorials Matplotlib, which have been
of great help on how to properly plot historical stock prices. More specifically,
see: https://www.youtube.com/watch?v=0e-lsstqCdY

@author: SMercier88
"""

# Here are our imports
import bs4 as bs
import csv
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib import style
import numpy as np
from operator import itemgetter
import pandas_datareader.data as web
import requests
import scipy.stats.mstats as mstats
from urllib.request import urlopen


style.use('ggplot')

# Dictionary identifying if we prefer that a fundamental indicator be high (H) or low (L)
best ={}
best['EPS'] = 'H'
best['Op_margin'] = 'H'
best['PE_f'] = 'L'
best['PE_t'] = 'L'
best['PEG'] = 'L'
best['PR'] = 'L'
best['P/B'] = 'L'
best['QEG'] = 'H'
best['QRG'] = 'H'
best['ROE'] = 'H'
best['SR'] = 'L'
best['D/E'] = 'L'
best['Dividend_t'] = 'H'
best['PE_t'] = 'L'
best['Buy_rating'] = 'L'
best['PT_diff'] = 'H'
best['CR'] = 'H'

# When we parse/extract data from Yahoo! finance, we get these weird name for the fundamental indicators;
# these "sk" (Simplified Keys) dictionaries translate them to simpler names  
sk_fundamental_ind ={}
sk_fundamental_ind['Diluted EPS (ttm)'] = 'EPS'
sk_fundamental_ind['Forward P/E 1'] = 'PE_f'
sk_fundamental_ind['PEG Ratio (5 yr expected) 1'] = 'PEG'
sk_fundamental_ind['Payout Ratio 4'] = 'PR'
sk_fundamental_ind['Price/Book (mrq)'] = 'P/B'
sk_fundamental_ind['Quarterly Earnings Growth (yoy)'] = 'QEG'
sk_fundamental_ind['Quarterly Revenue Growth (yoy)'] = 'QRG'
sk_fundamental_ind['Return on Equity (ttm)'] = 'ROE'
sk_fundamental_ind['Short Ratio 3'] = 'SR'
sk_fundamental_ind['Total Debt/Equity (mrq)'] = 'D/E'
sk_fundamental_ind['Trailing Annual Dividend Yield 3'] = 'Dividend_t'
sk_fundamental_ind['Trailing P/E '] = 'PE_t' 
sk_fundamental_ind['Current Ratio (mrq)'] = 'Current_ratio' 
sk_fundamental_ind['Operating Margin (ttm)'] = 'Op_margin'

sk_price ={}
sk_price['200-Day Moving Average 3'] = '200ma'
sk_price['50-Day Moving Average 3'] = '50ma'
sk_price['52 Week High 3'] = '52_high'
sk_price['52 Week Low 3'] = '52_low'
sk_price['52-Week Change 3'] = '52_change'

# Class to create an object for each sector; the information the user needs to input
# is the name of the sector, formatted following Wikipedia TSX Composite Index page
class Sector:
    def __init__(self, sector):
        self.sector = sector
        
    # Method to identify all stocks within the sector, and store the stocks in a
    # tickers attribute
    def get_tickers(self):
        # The ticker-sector pair information can be found on Wikipedia
        resp = requests.get('https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index')
        soup = bs.BeautifulSoup(resp.text, "lxml")
        # The info is in a table, the ticker being in the first column, and the 
        # sector of the given ticker in the third column
        table = soup.find('table', {'class':'wikitable sortable'})
        self.tickers = []
        for row in table.findAll('tr')[1:]:
            if row.findAll('td')[2].text == self.sector: # If the stock is in the given sector...
                self.tickers.append(row.findAll('td')[0].text) # ...we store its ticker in the tickers attribute
                
        self.number = len(self.tickers)
                
    # Method to calculate average, median value and best value (with the associated ticker) for a list of fundamental indicators
    def get_average_best(self, metrics):      
        # Extract fundamental indicators for each stock in the sector by calling the Stock class
        self.metrics = metrics
        self.average = {}
        self.median = {}
        self.best = {}
        self.get_tickers()
        self.stocks_list = [Stock(ticker) for ticker in self.tickers] # We create a stock object for each ticker in the sector
        
        for stock in self.stocks_list:
            stock.get_key_stats() # We extract the fundamental indicators for each stock using the get_key_stats method
            stock.analysts() # We extract the analysts' coverage using the analysts method
            
        for metric in metrics:
            # For each fundamental indicator, we extract the value for each stock and store it in the all_data list 
            all_data = [] 
            for stock in self.stocks_list:
                if metric in stock.Key_stats:
                    try:
                        all_data.append([stock.ticker, float(stock.Key_stats[metric])])
                    except:
                          pass 
                      
            # If we have atleast two stocks with a value for the given fundamental indicator, we
            # calculate the average, median and mean values, and store them in their respective
            # dictionary
            if len(all_data) >= 2:
                all_data = sorted(all_data, key=itemgetter(1))
                self.median[metric] = all_data[int(len(all_data)/2)]
                self.average[metric] = np.nanmean([c[1] for c in all_data])
                if best[metric] == 'H': # The "best" value depends on if we want the indicator to be low or high
                    self.best[metric] = all_data[len(all_data) - 1]
                elif best[metric] == 'L':
                    self.best[metric] = all_data[0]                                       
            
    # Method to extract to a csv file the given fundamental indicators for all stocks in the sector
    def export_csv_data(self, metrics):
        # We extract the fundamental indicators from Yahoo! Finance using the get_average_best method
        self.get_average_best(metrics)
        
        metric_array = ['ticker'] + metrics # Name of the columns in our csv file
        for stock in self.stocks_list:
            metric_vector = [stock.ticker]
            for metric in metrics:
                if metric in stock.Key_stats:
                    metric_vector.append(stock.Key_stats[metric]) 
                else:
                    metric_vector.append('nan')
            
            metric_array = np.vstack((metric_array, metric_vector))
        
        # Extract to a csv file, named "Statistics_?.csv" where "?" is the name of the sector        
        with open('Statistics_{}.csv'.format(self.sector), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(metric_array)
                   
    # Method to calculate the rank of all stocks in the sector for a given list of metrics
    def rank_all_stocks(self, metrics, update=True):        
        # We load the fundamental indicators from a csv file (if update=False), 
        # or from the export_csv_data method (if update=True)
        if update:
            self.export_csv_data(metrics)
        
        try:
            with open('Statistics_{}.csv'.format(self.sector), 'r') as f:
                Stats = []
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    Stats.append(row)
        except:
            print('Cannot find file...probably does not exists in the directory')
        
        #        
        ranks = [c[0] for c in Stats]        
        for i1 in range(len(Stats[0])): # For each fundamental indicator in the csv file...            
            if Stats[0][i1] in metrics: # ...if that fundamental indicator is in our list of metrics...
                array = np.array([float(c[i1]) for c in Stats[1:]]) # ...we store its value for each stock in an array variable
                rank = mstats.rankdata(np.ma.masked_invalid(array)) # We rank each stock for this fundamental indicator
                rank[rank == 0] = np.nan
                
                if Stats[0][i1] in best:
                    if best[Stats[0][i1]] == 'H': # The rank depends on if we want the value of the fundamental indicator to be high or low
                        rank = np.nanmax(rank) + 1 - rank
                
                rank = list(rank)
#                for i in range(len(rank)):
#                    if not np.isnan(rank[i]):
#                        rank[i] = int(rank[i])
                        
                rank = [Stats[0][i1]] + rank # We stack the ranks for all fundamental indicators in a matrix ranks
                ranks = np.vstack((ranks, rank))
        
        # We extract the rank of each stock for each fundamental indicator in a csv file name 
        # "Rank_?.csv", where "?" is the name of the sector
        with open('Rank_{}.csv'.format(self.sector), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(np.transpose(ranks))        
    
    # Method to identify the stocks with the best ranks given a list of fundamental indicators
    def identify_best_stock(self, metrics, update=True, min_nb_of_ranks=2):   
        # We load the fundamental indicators from a csv file (if update=False), 
        # or from the export_csv_data method (if update=True)
        if update:
            self.rank_all_stocks(metrics)
        
        try:
            with open('Rank_{}.csv'.format(self.sector), 'r') as f:
                Ranks = []
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    Ranks.append(row)
        except:
            print('Cannot find file...probably does not exists in the directory')
        
        # Each stock has a rank for each fundamental indicator...we calculate here 
        # its average rank over the given list of fundamental indicators. We store the average rank
        # for each stock in a stocks_average_rank dictionary
        self.stocks_average_rank = {}
        for i1 in range(1, len(Ranks)):
            self.stocks_average_rank[Ranks[i1][0]] = [np.nanmean(np.array([float(c) for c in Ranks[i1][1:]])) if np.count_nonzero(~np.isnan([float(c) for c in Ranks[i1][1:]])) >= min_nb_of_ranks else 100] 
        
        # The "best" stock is the one with the lowest average rank
        self.best_stock = min(self.stocks_average_rank.items(), key=lambda c: c[1])   

# Class to create an object for each stock; the information the user needs to input
# is the stock ticker
class Stock(Sector):
    def __init__(self, ticker):
        self.ticker = ticker
        self.all_stats = {}
        self.Key_stats = {}
        self.price = {}
        # Identify the sector of our stock
        resp = requests.get('https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index')
        soup = bs.BeautifulSoup(resp.text, "lxml")
        table = soup.find('table', {'class':'wikitable sortable'})
        for row in table.findAll('tr')[1:]: # We loop through the rows to find our stock
            if row.findAll('td')[0].text == self.ticker: # If we have found our ticker...
                sector = row.findAll('td')[2].text # ...the sector is in the third column of the table
        
        super().__init__(sector)
    
    # Method to extract fundamental indicators of the stock from Yahoo! finance
    def get_key_stats(self):
               
        try:
            # The Website url has the same basic pattern for each stock
            resp = requests.get('https://finance.yahoo.com/quote/{}.TO/key-statistics?p={}.TO'.format(self.ticker.replace('.','-'), 
                                self.ticker.replace('.','-')))
            # We parse the text, and loop through the tables, in which are stored 
            # the fundamental indicators
            soup = bs.BeautifulSoup(resp.text, "lxml")
            tables = soup.findAll('table')
            for table in tables:
                for row in table.findAll('tr')[1:]:
                    # The name of the fundamental indicator is in the first column of the table
                    keyv = row.findAll('td')[0].text

                    try:
                        attri = row.findAll('td')[1].text
                        # We remove the last value in the string if it is not a number (it is often
                        # a "%" sign)
                        if not attri[len(attri)-1].isdigit():
                            attri = attri[:len(attri)-1]
                        
                        # We store the fundamental indicator in the proper dictionary
                        self.all_stats[keyv] = float(attri)
                        if keyv in sk_fundamental_ind:
                            # We translate the name of the fundamental indicator extracted from 
                            # Yahoo! finance to the more simple name in our sk dictionary 
                            keyv = sk_fundamental_ind[keyv]
                            self.Key_stats[keyv] = float(attri)
                            
                        elif keyv in sk_price:
                            keyv = sk_price[keyv]
                            self.price[keyv] = float(attri)
                        
                    except Exception as e:
                        pass
                    
                    # If we haven't been able to extract an indicator, replace with an NaN
                    for _, key in sk_fundamental_ind.items():
                        if key not in self.Key_stats:
                            self.Key_stats[key] = np.nan
                           
                    for _, key in sk_price.items():
                        if key not in self.price:
                            self.price[key] = np.nan 
                                            
                    # Remove negative price per earnings
                    if keyv in ['PE_f', 'PE_t', 'PEG']:
                        if self.Key_stats[keyv] < 0:
                            self.Key_stats[keyv] = np.nan
            
        except Exception as e:
            pass
    
    # Method to get analysts' coverage, and store the value in our "Key_stats" dictionary
    def analysts(self):
        try:
            # We call the Yahoo! Finance page of the stock presenting the analysts's coverage information, and store
            # the page source content in the "contents" variable
            page = urlopen('https://finance.yahoo.com/quote/{}.TO/analysts?p={}.TO'.format(self.ticker.replace('.','-'), 
                           self.ticker.replace('.','-')))
            contents = page.read()
            page.close()
            
            # The different indicators can always be found between specific sequence of characters,
            # given in these split1 and split2 variables below
            # Number of analysts covering the stock:
            split1 = b'"numberOfAnalystOpinions":{"raw":'
            split2 = b',"fmt"'
            self.Key_stats['Nb_analysts'] = float(contents.split(split1)[1].split(split2)[0])
            
            # Buy_recommendation:
            split1 = b'"recommendationMean":{"raw":'
            split2 = b',"fmt"'
            self.Key_stats['Buy_rating'] = float(contents.split(split1)[1].split(split2)[0])
            
            # Target_price
            split1 = b'"targetMeanPrice":{"raw":'
            split2 = b',"fmt"'
            self.Key_stats['PT'] = float(contents.split(split1)[1].split(split2)[0])
            
            # Current_price
            split1 = b'"regularMarketPrice":{"raw":'
            split2 = b',"fmt"'
            self.price['CP'] = float(contents.split(split1)[1].split(split2)[0])
            self.Key_stats['PT_diff'] = (float(self.Key_stats['PT']) - float(self.price['CP']))/float(self.price['CP'])*100
               
        except Exception as e:
            pass
        
    # Method to calculate the rank of a stock, with respect to the other stocks in the same industry,
    # for a given list of fundamental indicators (provided in the argument metrics)
    def rank_of_stock(self, metrics, update=True):
           
        # To rank our stock, we need to extract the fundamental indicators of all the other
        # stocks in the same sector. If the argument update==True, we do so by calling the 
        # rank_all_stocks method of the Sector class; otherwise, we use the fundamental indicators
        # stored in a csv file, if it is available        
        if update:        
            self.rank_all_stocks(metrics)
            
        try:
            with open('Rank_{}.csv'.format(self.sector), 'r') as f:
                Ranks = []
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    Ranks.append(row) # Rank of all the stocks in the same sector for all the fundamental indicators
        except:
            print('Cannot find csv file...probably does not exists in the directory')
        
        
        # For each fundamental indicator, we calculate the rank of our stock with respect
        # to the other stocks in the same sector, and store it in a rank attribute        
        self.rank = {}
        rank_list = [rank for rank in Ranks if rank[0] == self.ticker]
        for i1 in range(len(Ranks[0])):
            if Ranks[0][i1] in metrics:
                self.rank[Ranks[0][i1]] = rank_list[0][i1]
    
            
    # Method to plot the historical price of our stock on a candlestick chart, in addition
    # to the moving average and volume information
    def get_charts(self, start=(2010, 1, 1), candles_size=10, moving_average=100):
        candles = '{}D'.format(candles_size)
        start = dt.datetime(*start) # We want to plot from the date in start to...
        end = dt.date.today()       # ...the date it is today
        
        # Extract the price of our stock from Yahoo! Finance
        df = web.DataReader(self.ticker.replace('.','-') + '.TO', 'yahoo', start, end)
        
        # We calculate the open, high, low and close prices within a given period (argument candles)
        # using the ohlc method from pandas
        df_ohlc = df['Adj Close'].resample(candles).ohlc()
        df_volume = df['Volume'].resample(candles).sum() # We sum all the volumes over that period
        df['{}ma'.format(moving_average)] = df['Adj Close'].rolling(window=moving_average, min_periods=0).mean()
        
        df_ohlc.reset_index(inplace=True)
        df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
        
        # We plot the price information on the top plot, and the volume info on the bottom one
        ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
        ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex = ax1)
        ax1.xaxis_date()
        ax1.set_ylabel('Price ($)')
        
        candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
        ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
        ax2.set_ylabel('Volume')
        
        # We add the price moving average on top of the candlestick chart
        ax1.plot(df.index, df['{}ma'.format(moving_average)], color = 'b', linewidth=1.0)    