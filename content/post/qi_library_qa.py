# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 13:47:35 2017

@author: njumq
"""

import pandas as pd
import numpy as np
import talib
import QUANTAXIS as QA
from QUANTAXIS.QAFetch.QATdx import (
    QA_fetch_get_option_day,
    QA_fetch_get_option_min,
    QA_fetch_get_index_day,
    QA_fetch_get_index_min,
    QA_fetch_get_stock_day,
    QA_fetch_get_stock_info,
    QA_fetch_get_stock_list)
from QUANTAXIS.QAUtil import (DATABASE,QA_util_log_info)
from QUANTAXIS.QAData.data_resample import QA_data_day_resample
from QUANTAXIS.QAFetch import QAQuery
import datetime
import pymongo
import concurrent
import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import pandas as pd
import pymongo
from QUANTAXIS.QAFetch.QAQuery_Advance import *
from QUANTAXIS.QAFetch.QATdx import *

from QUANTAXIS.QAFetch import QA_fetch_get_stock_block
from QUANTAXIS.QAFetch.QATdx import (
    QA_fetch_get_option_day,
    QA_fetch_get_option_min,
    QA_fetch_get_index_day,
    QA_fetch_get_index_min,
    QA_fetch_get_stock_day,
    QA_fetch_get_stock_info,
    QA_fetch_get_stock_list,
    QA_fetch_get_future_list,
    QA_fetch_get_index_list,
    QA_fetch_get_future_day,
    QA_fetch_get_future_min,
    QA_fetch_get_stock_min,
    QA_fetch_get_stock_xdxr,
    select_best_ip
)

from QUANTAXIS.QAUtil import (
    DATABASE,
    QA_util_get_next_day,
    QA_util_get_real_date,
    QA_util_log_info,
    QA_util_to_json_from_pandas,
    trade_date_sse
)
from QUANTAXIS.QAUtil import Parallelism
from QUANTAXIS.QAFetch.QATdx import ping, get_ip_list_by_multi_process_ping, stock_ip_list
from multiprocessing import cpu_count
from QUANTAXIS.QAFetch.QAQuery_Advance import *

def watch_list():
    codelist = ['000786', '600066','600817','000830', '300059','601318','600309','600031',
                '600009', '603883','000818']
    return codelist

def sz50():
    res = QA_fetch_stock_block_adv()
    code50 = res.get_block('上证50').code
    return code50
    
def QA_get_stockinfo(codelist : list):
    res = QAQuery.QA_fetch_stock_info(codelist, collections=DATABASE.stock_info)
    return res.values
    
def QA_get_code_name():
    codelist = QA_fetch_stock_list()
    codelist = codelist[['code','name']].reset_index(drop=True)
    return codelist

    
def QA_get_stock_name_day(codelist, start_date = '2019-07-01'):
  import datetime
  today = datetime.datetime.today().strftime('%Y-%m-%d')
  res = QA_fetch_stock_block_adv()
  #code50 = res.get_block('上证50').code
  #start_date = '2019-07-01'
  end_date = today
  df = QA_fetch_stock_day_adv(codelist, start_date, end_date).to_qfq().data
  df = df.reset_index()
  codelist = QA_fetch_stock_list()
  codelist = codelist[['code','name']].reset_index(drop=True)
  df = df.merge(codelist, on='code', how='left')[['date','code','open','high','low','close','volume','amount', 'name']]
  return df

def QA_fetch_data_day_adv(code, start_date, end_date):
    """
    fetch the day data, support for stock, index and etf
    :param code:
    :param start:
    :param end:
    :return:
    """


    coll_stock_day = DATABASE.stock_day
    stock_list = DATABASE.stock_list
    code_list = coll_stock_day.distinct('code')
    index_day = DATABASE.index_day
    index_list = index_day.distinct('code')

    if code in code_list:
        df = QA.QA_fetch_stock_day_adv([code], start_date, end_date)
    else:
        df = QA.QA_fetch_index_day_adv([code], start_date, end_date)

    data = df.data[['open','high','low','close','volume','amount']]
    data['date'] = df.datetime

    return data

def QA_fetch_latest(code):
    price=QA.QAFetch.QATdx.QA_fetch_get_stock_latest(code, ip ="61.152.249.56", port=7709)
    return price.close.values[0]
    #return price[['open','close','high','low','vol']]
    



def QA_filter_day(threshold : float, cond_day : int, cond_week : int, cond_month :int):
    """
    Stock selection system
    
    Args:
        threshold (float): To select stock have perchange change over it.
        cond_day (int): 0 for current price is less than day 5MA, otherwise great than day 5MA.
        cond_weeek (int): 0 for current price is less than week 5MA, otherwise great than week 5MA.
        cond_month (int): 0 for current price is less than month 5MA, otherwise great than month 5MA.
        
    Return:
        (list, list): codelist, namelist of the selected stock.

    """
    coll_stock_day = DATABASE.stock_day
    stock_list = DATABASE.stock_list
    code_list = coll_stock_day.distinct('code')
    name_list = stock_list.distinct('name')
    para1 = threshold # threshold for selecting the stock 
    today = datetime.datetime.now()
    start_date = today - datetime.timedelta(days = 100)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    df = QA.QA_fetch_stock_day_adv(code_list, start_date, end_date)
    codelist1 = df.pct_change[df.pct_change > 0.09].reset_index().code.unique()
    #####
    start_date = today - datetime.timedelta(days = 200)
    start_date = start_date.strftime("%Y-%m-%d")
    code_result = []
    name_result = []
    for code in codelist1:
        day_data =  QA.QA_fetch_stock_day_adv(code,start_date,end_date).to_qfq()
        week_data = QA_data_day_resample(day_data,'W')
        month_data = QA_data_day_resample(day_data, 'M')
        ind_day = QA.SMA(day_data['close'], 5)
        ind_week = QA.SMA(week_data['close'], 5)
        ind_month = QA.SMA(month_data['close'], 5)
        if cond_day == 0:
            cond1 = day_data['close'][-1] < ind_day[-1]
        else:
            cond1 = day_data['close'][-1] > ind_day[-1]
        if cond_week == 0:
            cond2 = day_data['close'][-1] < ind_week[-1]
        else:
            cond2 = day_data['close'][-1] > ind_week[-1]
        if cond_month == 0 :    
            cond3 = day_data['close'][-1] < ind_month[-1]
        else:
            cond3 = day_data['close'][-1] > ind_month[-1]

        if cond1 and cond2 and cond3 :
            print(code)
            code_result.append(code)
    for code in code_result:
        name_result.append(stock_list.find_one({'code':code})['name'])
    return (code_result, name_result)

def QA_resample_day_R(code, start_date, end_date, type):
    """
    Resample the QA data structor to R data format

    :param data: QA.datastruct format
    :param type: W (week), M (Month)
    :return:
    """
    data = QA_fetch_data_day_adv(code, start_date, end_date)
    data = data.drop(['date'], axis=1)
    data = QA.QAData.data_resample.QA_data_day_resample(data,type)
    data = data.reset_index()
    return data



def QA_util_firstDayTrading(codelist: list):
    """
    取得交易品种的第一个上市日期，或第一个交易日。支持混合股票,index,etf
    """
    coll_stock_day = DATABASE.stock_day
    coll_index_day = DATABASE.index_day
    coll_stock_day.create_index(
    [("code",
      pymongo.ASCENDING),
     ("date_stamp",
      pymongo.ASCENDING)]
    )
    coll_index_day.create_index(
    [("code",
      pymongo.ASCENDING),
     ("date_stamp",
      pymongo.ASCENDING)]
    )
    
    dates = []
    for code in codelist:
        ref = coll_stock_day.find({"code": code})
        ref2 = coll_index_day.find({'code': code})
        #print('{} is ref is {}, ref2 is {}'.format(code, ref.count(), ref2.count()))
        if ref.count() > 0:
            start_date = ref[0]['date']
            dates.append(start_date)
        elif ref2.count() > 0:
            start_date = ref2[0]['date']
            dates.append(start_date)
        else:
            raise ValueError('{} 没有数据'.format(code))
            
    return pd.DataFrame({'code':codelist, 'date': dates} )
    
def QA_SU_save_stock_day(stock_list, client=DATABASE, ui_log=None, ui_progress=None):
    '''
     save stock_day
    保存日线数据
    :param client:
    :param ui_log:  给GUI qt 界面使用
    :param ui_progress: 给GUI qt 界面使用
    :param ui_progress_int_value: 给GUI qt 界面使用
    '''
    #stock_list = QA_fetch_get_stock_list().code.unique().tolist()
    
    coll_stock_day = client.stock_day
    coll_stock_day.create_index(
        [("code",
          pymongo.ASCENDING),
         ("date_stamp",
          pymongo.ASCENDING)]
    )
    err = []

    def __saving_work(code, coll_stock_day):
        try:
            QA_util_log_info(
                '##JOB01 Now Saving STOCK_DAY==== {}'.format(str(code)),
                ui_log
            )

            # 首选查找数据库 是否 有 这个代码的数据
            ref = coll_stock_day.find({'code': str(code)[0:6]})
            end_date = str(now_time())[0:10]

            # 当前数据库已经包含了这个代码的数据， 继续增量更新
            # 加入这个判断的原因是因为如果股票是刚上市的 数据库会没有数据 所以会有负索引问题出现
            if ref.count() > 0:

                # 接着上次获取的日期继续更新
                start_date = ref[ref.count() - 1]['date']

                QA_util_log_info(
                    'UPDATE_STOCK_DAY \n Trying updating {} from {} to {}'
                    .format(code,
                            start_date,
                            end_date),
                    ui_log
                )
                if start_date != end_date:
                    coll_stock_day.insert_many(
                        QA_util_to_json_from_pandas(
                            QA_fetch_get_stock_day(
                                str(code),
                                QA_util_get_next_day(start_date),
                                end_date,
                                '00'
                            )
                        )
                    )

            # 当前数据库中没有这个代码的股票数据， 从1990-01-01 开始下载所有的数据
            else:
                start_date = '1990-01-01'
                QA_util_log_info(
                    'UPDATE_STOCK_DAY \n Trying updating {} from {} to {}'
                    .format(code,
                            start_date,
                            end_date),
                    ui_log
                )
                if start_date != end_date:
                    coll_stock_day.insert_many(
                        QA_util_to_json_from_pandas(
                            QA_fetch_get_stock_day(
                                str(code),
                                start_date,
                                end_date,
                                '00'
                            )
                        )
                    )
        except Exception as error0:
            print(error0)
            err.append(str(code))

    for item in range(len(stock_list)):
        QA_util_log_info('The {} of Total {}'.format(item, len(stock_list)))

        strProgressToLog = 'DOWNLOAD PROGRESS {} {}'.format(
            str(float(item / len(stock_list) * 100))[0:4] + '%',
            ui_log
        )
        intProgressToLog = int(float(item / len(stock_list) * 100))
        QA_util_log_info(
            strProgressToLog,
            ui_log=ui_log,
            ui_progress=ui_progress,
            ui_progress_int_value=intProgressToLog
        )

        __saving_work(stock_list[item], coll_stock_day)

    if len(err) < 1:
        QA_util_log_info('SUCCESS save stock day ^_^', ui_log)
    else:
        QA_util_log_info('ERROR CODE \n ', ui_log)
        QA_util_log_info(err, ui_log)
    
    

    
def gaptest_tdx(yu, start_date, end_date): # not require gappercent
    #gappercent = 0.2
    yu1 = yu[:]
    yu2 = yu1[yu1['date'] <= end_date].sort_values('date')
    yu2 = yu2[yu2['date'] >= start_date]
    deltatime = []
    notfilldate = []
    gapdays=[]
    gapdates=[]
    for i in np.arange(yu2.shape[0]):
        date = yu2.iloc[i,:]["date"]
        gapdates.append(date)
        if yu2.iloc[i,:]['gap'] >= 0:
            yu3 = yu[yu['low'] <= yu[yu['date'] == date]['close2'].values[0]]
        else:
            yu3 = yu[yu['high'] >= yu[yu['date'] == date]['close2'].values[0]]
        yu4 = yu3[yu3['date'] >= date].sort_values('date').set_index('date')
        #yu4 = yu3[yu3['date'] >= date].sort_values('Date')
        if yu4.empty:
            print (date, round(yu2.iloc[i,:]['gap'],2),round(yu2.iloc[i,:]['close2'],2))
            notfilldate.append(date)
            gapdays.append(555)
        else:
            #deltatime.append((datetime.strptime(yu4.index[0] , "%Y-%m-%d") - datetime.strptime(date, "%Y-%m-%d")).days)
            deltatime.append((yu4.index[0] - date).days)
            gapdays.append((yu4.index[0] - date).days)
    gaps = pd.DataFrame({'date': gapdates, 'length':gapdays})
    #gaps.set_index('Date', inplace = True)
    yu2 = yu2.drop(columns=['volume'])
    gap_data = pd.merge(gaps, yu2, on= 'date',how = 'right' )
    return gap_data,gaps

def get_ma_ouptput(codes):
    """
    get moving average day, week, month for output
    :param codes:
    :return:

    codes = [("code", "name")]
    """
    #codes = [('000636', '风华高科') ,  ('600066', '宇通客车'),  ('000786','北新建材'), ('600817','ST宏盛'),
    #     ('000830', '鲁西化工'), ('510050', '50ETF')]

    #codes2 = ['QQQ','SPY','IWM','DIA', 'GLD']


    today = datetime.datetime.now()
    start_date = today - datetime.timedelta(days = 200)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    output_m =[]
    output_w =[]
    output_d =[]

    coll_stock_day = DATABASE.stock_day
    stock_list = DATABASE.stock_list
    code_list = coll_stock_day.distinct('code')
    index_day = DATABASE.index_day
    index_list = index_day.distinct('code')


    for (code, symbol) in codes:
        ''''
        if code  == '510050':
            df = QA.QA_fetch_index_day_adv([code], start_date, end_date)
        else:
            df = QA.QA_fetch_stock_day_adv([code], start_date, end_date)
        #df = ts.get_hist_data(code, ktype='D', start=start_date, end=end_date)
        if code != '510050':
            df = df.to_qfq()()
        else:
            df = df()
        '''
        if code in code_list:
            df = QA.QA_fetch_stock_day_adv([code], start_date, end_date)
            df = df.to_qfq()()
        else:
            df = QA.QA_fetch_index_day_adv([code], start_date, end_date)()

        df_w = QA.QAData.data_resample.QA_data_day_resample(df, 'W')

        df_m = QA.QAData.data_resample.QA_data_day_resample(df, 'M')

        df['ma5'] = talib.SMA(df['close'].values, timeperiod=5)
        df_w['ma5'] = talib.SMA(df_w['close'].values, timeperiod=5)
        df_m['ma5'] = talib.SMA(df_m['close'].values, timeperiod=5)
        df = df.reset_index().tail(1)
        df_w = df_w.reset_index().tail(1)
        df_m = df_m.reset_index().tail(1)
        df['indicator'] = 'No'
        df_w['indicator'] = 'No'
        df_m['indicator'] = 'No'
        df.loc[df.close > df.ma5,'indicator'] = "Yes"
        df_w.loc[df_w.close > df_w.ma5,'indicator'] = "Yes"
        df_m.loc[df_m.close > df_m.ma5,'indicator'] = "Yes"

        df_m['symbol'] = symbol
        #df_m['code'] = code
        df_m['type'] = '月5MA'
        #df['code'] = code
        df['type'] = '日5MA'
        df['symbol'] = symbol
        #df_w['code'] = code
        df_w['type'] = '周5MA'
        df_w['symbol'] = symbol
        output_m.append(df_m)
        output_d.append(df)
        output_w.append(df_w)
    result_m = pd.concat(output_m)
    result_d = pd.concat(output_d)
    result_w = pd.concat(output_w)

    cols = ['symbol','code','date','type','close','ma5','indicator']
    result = [result_m[cols], result_d[cols], result_w[cols]]
    combined = pd.concat(result)
    combined = combined.sort_values(['code', 'type'])
    combined['diff'] = round(combined.close - combined.ma5,2)
    combined['close'] = round(combined.close, 2)
    combined['ma5'] = round(combined.ma5, 2)
    combined['date'] = combined.date.map(lambda x: x.strftime("%Y-%m-%d"))
    combined['type'] = pd.Categorical(combined['type'], ['日5MA', '周5MA', '月5MA'])
    combined = combined.sort_values(['symbol', 'type'])

    return combined

    
def gap_list(code : str):
    """
    Get gap list
    """
    today = datetime.datetime.now()
    start_date = today - datetime.timedelta(days = 200)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    coll_stock_day = DATABASE.stock_day
    stock_list = DATABASE.stock_list
    code_list = coll_stock_day.distinct('code')
    index_day = DATABASE.index_day
    index_list = index_day.distinct('code')
    if code in code_list:
        df = QA.QA_fetch_stock_day_adv([code], start_date, end_date)
        df = df.to_qfq()()
    else:
        df = QA.QA_fetch_index_day_adv([code], start_date, end_date)
       # df = df.to_qfq()()
    df = df.reset_index()
    df = getgapdata_ib(df)
    gap_data, gaps = gaptest_tdx(df, '2018-05-13', end_date)
    gaps = gap_data.sort_values('date', ascending= True) # used for calcualing gap day not closed count
    gaps_nc = gaps[gaps.length==555]
    gaps_nc = getGaps_nc(gaps_nc) # get gap day not closed count
    gaps_nc = gaps_nc.sort_values('date', ascending= False)
    gap_list = gaps_nc[['code','date','close','close2','gap','up_count','down_count']]
    current_close  = df.head(1).close.values[0]
    gap_list['diff'] = current_close - gap_list.close2
    gap_list['current_close'] = current_close
    return gap_list 

def read_data_5M(filename):
    '''
    Read csv 5m data file fro IBridgePY
    set time as the index and convert the date properly
    '''
    data = pd.read_csv(filename )
    data.time = data['time'].astype(str).str[:-6]
    data.set_index('time', inplace = True)
    data.index = pd.DatetimeIndex(data.index)
    return data
    
def read_data_1D(filename):
    '''
    Read csv 1 Day data file fro IBridgePY
    set time as the index and convert the date properly
    '''
    data = pd.read_csv(filename, parse_dates = [0] )
    #data.time = data['time'].astype(str).str[:-6]
    data.set_index('time', inplace = True)
    data.index = pd.DatetimeIndex(data.index)
    return data

def getgapdata(yutong2): # for ts data
    '''
    Get gap dataframe, input data will be sorted in the accending 
    date order, the gap for the first date is zero
    '''
    #yutong = yutong[yutong['Volume'].astype(int) != 0] # remove zero volume
    yutong = yutong2[:]
    yutong.sort_index(ascending= False, inplace = True)
    close = yutong['close'].values
    open = yutong['open'].values
    close2 = np.append(close[1:], open[-1]) # previous day close
    yutong['close2'] = close2
    yutong['gap'] = yutong['open'].astype('float') - yutong['close2'].astype('float')
    yutong['gap percent'] = yutong['gap'] / yutong['close2'].astype('float')* 100
    yutong['date']= yutong.index
    return yutong

def getgapdata_ib(yutong2): # for ts data
    '''
    Get gap dataframe, input data will be sorted in the accending 
    date order, the gap for the first date is zero
    '''
    #yutong = yutong[yutong['Volume'].astype(int) != 0] # remove zero volume
    yutong = yutong2[:]
    yutong.date= yutong.date.astype('datetime64[ns]')
    yutong = yutong.sort_values('date', ascending = False)
    #yutong.sort_index(ascending= False, inplace = True)
    close = yutong['close'].values
    open = yutong['open'].values
    close2 = np.append(close[1:], open[-1]) # previous day close
    yutong['close2'] = close2
    yutong['gap'] = yutong['open'].astype('float') - yutong['close2'].astype('float')
    yutong['gap percent'] = yutong['gap'] / yutong['close2'].astype('float')* 100
    #yutong['date']= yutong.index
    return yutong


def gaptest(yu, gappercent, start_date, end_date):
    #gappercent = 0.2
    yu1 = yu[abs(yu['gap percent']) > gappercent ]
    yu2 = yu1[yu1['date'] <= end_date].sort_values('date')
    yu2 = yu2[yu2['date'] >= start_date]
    deltatime = []
    notfilldate = []
    gapdays=[]
    gapdates=[]
    for i in np.arange(yu2.shape[0]):
        date = yu2.iloc[i,:]["date"]
        gapdates.append(date)
        if yu2.iloc[i,:]['gap'] >= 0:
            yu3 = yu[yu['low'] <= yu[yu['date'] == date]['close2'].values[0]]
        else:
            yu3 = yu[yu['high'] >= yu[yu['date'] == date]['close2'].values[0]]
        yu4 = yu3[yu3['date'] >= date].sort_values('date').set_index('date')
        #yu4 = yu3[yu3['date'] >= date].sort_values('Date')
        if yu4.empty:
            print (date)
            notfilldate.append(date)
            gapdays.append(555)
        else:
            #deltatime.append((datetime.strptime(yu4.index[0] , "%Y-%m-%d") - datetime.strptime(date, "%Y-%m-%d")).days)
            deltatime.append((yu4.index[0] - date).days)
            gapdays.append((yu4.index[0] - date).days)
    return (yu1,yu2,notfilldate, deltatime, gapdates, gapdays)


def gaptest2(yu, start_date, end_date): # not require gappercent
    #gappercent = 0.2
    yu1 = yu[:]
    yu2 = yu1[yu1['date'] <= end_date].sort_values('date')
    yu2 = yu2[yu2['date'] >= start_date]
    deltatime = []
    notfilldate = []
    gapdays=[]
    gapdates=[]
    for i in np.arange(yu2.shape[0]):
        date = yu2.iloc[i,:]["date"]
        gapdates.append(date)
        if yu2.iloc[i,:]['gap'] >= 0:
            yu3 = yu[yu['low'] <= yu[yu['date'] == date]['close2'].values[0]]
        else:
            yu3 = yu[yu['high'] >= yu[yu['date'] == date]['close2'].values[0]]
        yu4 = yu3[yu3['date'] >= date].sort_values('date').set_index('date')
        #yu4 = yu3[yu3['date'] >= date].sort_values('Date')
        if yu4.empty:
            print (date, round(yu2.iloc[i,:]['gap'],2),round(yu2.iloc[i,:]['close2'],2))
            notfilldate.append(date)
            gapdays.append(555)
        else:
            #deltatime.append((datetime.strptime(yu4.index[0] , "%Y-%m-%d") - datetime.strptime(date, "%Y-%m-%d")).days)
            deltatime.append((yu4.index[0] - date).days)
            gapdays.append((yu4.index[0] - date).days)
    gaps = pd.DataFrame({'date': gapdates, 'length':gapdays})
    #gaps.set_index('Date', inplace = True)
    yu2 = yu2.drop(columns=['volume','barCount','average'])
    gap_data = pd.merge(gaps, yu2, on= 'date',how = 'right' )
    return gap_data,gaps


def getGaps_nc(gaps_nc):
    gaps_nc['up']=0
    gaps_nc.loc[gaps_nc['gap']>0,'up'] = 1
    y = gaps_nc['up']
    up_count = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)
    gaps_nc.loc[:,'up_count'] = up_count 

    gaps_nc['down']=0
    gaps_nc.loc[gaps_nc['gap']<=0,'down'] = 1
    y = gaps_nc['down']
    down_count = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)
    gaps_nc.loc[:,'down_count'] = down_count 

    return gaps_nc

def getBarContCount(df):
    MA = talib.SMA(df.close.values, timeperiod=5)
    df['up']=0
    df.loc[df.close-MA >= 0,'up'] = 1

    df['down']=0
    df.loc[df.close-MA < 0,'down'] = 1

    y = df['down']
    down_count = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)
    df.loc[:,'down_count'] = down_count 

    y = df['up']
    up_count = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)
    df.loc[:,'up_count'] = up_count 

    return df




def get_innerday(df): 
    data_30m = df[:]
    #data_30m.index = pd.MultiIndex.from_arrays([data_30m.index.date, data_30m.index.time], names=['Date','Time'])
    data_30m['date'] = data_30m.index.date
    data_30m['time'] = data_30m.index.time

    data_30m['change_per'] = (data_30m.close - data_30m.open)/data_30m.open*100
    data_30m['change'] = (data_30m.close - data_30m.open)
    data_30m['abs_change_per'] = np.abs((data_30m.close - data_30m.open)/data_30m.open*100)
    data_30m['abs_change'] = np.abs((data_30m.close - data_30m.open))

    #times= ['09:30:00','10:00:00','10:30:00','11:00:00','11:30:00','12:00:00',
    #        '12:30:00','13:00:00','13:30:00','14:00:00','14:30:00','15:00:00',
    #       '15:30:00']

    times = np.unique(data_30m['time'].values)

    innerday= pd.DataFrame()
    change_abs_mean = []
    change_abs_var = []
    for time in times:
        change_abs_mean.append(data_30m.iloc[data_30m.index.indexer_at_time(time)]['abs_change_per'].mean())
        change_abs_var.append(data_30m.iloc[data_30m.index.indexer_at_time(time)]['abs_change_per'].var())
    innerday['time'] = times
    innerday['abs_mean']= change_abs_mean
    innerday['abs_var']= change_abs_var
    return data_30m, innerday
    

def get_innerday2(df): 
    data_30m = df[:]
    times = np.unique(data_30m['time'].values)

    innerday= pd.DataFrame()
    change_abs_mean = []
    change_abs_var = []
    for time in times:
        change_abs_mean.append(data_30m.loc[data_30m['time'] == time,'abs_change_per'].mean())
        change_abs_var.append(data_30m.loc[data_30m['time'] == time,'abs_change_per'].var())
    innerday['time'] = times
    innerday['abs_mean']= change_abs_mean
    innerday['abs_var']= change_abs_var
    return innerday

def gap_fill_rate_positive(df, gap_size, large): # input gap_data
    assert gap_size > 0, 'gap size needs be positive'
    gap_data = df[:]
    if large == True:
        a = gap_data[gap_data['gap percent'] >= gap_size]['Length'].values
    else:
        a = gap_data[ gap_data['gap percent'] <= gap_size][gap_data['gap percent'] >= 0 ]['Length'].values
    (gap_day, gap_day_num) = np.unique(a, return_counts=True) # always missng 4 & 7 & 9, 
    same_day_gap_fill = gap_day_num[0]/len(a)*100
    print('Same day gap fill chance is ', same_day_gap_fill)
    return same_day_gap_fill, gap_day, gap_day_num


def gap_fill_rate_negative(df, gap_size, large): # input gap_data
    assert gap_size < 0, 'gap size needs be negative'
    gap_data = df[:]
    if large == True:
        a = gap_data[gap_data['gap percent'] <= gap_size]['Length'].values
    else:
        a = gap_data[ gap_data['gap percent'] >= gap_size][gap_data['gap percent'] <= 0 ]['Length'].values
    (gap_day, gap_day_num) = np.unique(a, return_counts=True) # always missng 4 & 7 & 9, 
    same_day_gap_fill = gap_day_num[0]/len(a)*100
    print('Same day gap fill chance is ', same_day_gap_fill)
    return same_day_gap_fill, gap_day, gap_day_num


def gap_fill_rate_abs(df, gap_size, large): # input gap_data
    '''
    Calculate gap fill rate combining both negative and positive gap percent instances
    '''
    gap_data = df[:]
    if large == True:
        a = gap_data[np.abs(gap_data['gap percent']) >= np.abs(gap_size)]['Length'].values
    else:
        a = gap_data[np.abs(gap_data['gap percent']) <= np.abs(gap_size)]['Length'].values
    (gap_day, gap_day_num) = np.unique(a, return_counts=True) # always missng 4 & 7 & 9, 
    same_day_gap_fill = gap_day_num[0]/len(a)*100
    print('Same day gap fill chance is ', same_day_gap_fill)
    return same_day_gap_fill, gap_day, gap_day_num
































