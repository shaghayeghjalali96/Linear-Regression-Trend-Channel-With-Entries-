import pandas as pd 
import numpy as np
import math 
import config
from finta import TA
import collections
from time import time
from multiprocessing import Manager
import multiprocessing as mp
import datetime
import logging 
import os
from logging.handlers import QueueHandler, QueueListener

logger = logging.getLogger('api.sub')

class ResultModel:
      def __init__(self, j,linearRegression,intercept,slope,deviation,startingPointY,startingPointYup,startingPointYlow,upperband,lowerband):
          self.index=j
          self.linearRegression=linearRegression
          self.intercept=intercept
          self.slope=slope
          self.deviation=deviation
          self.startingPointY=startingPointY
          self.startingPointYup=startingPointYup
          self.startingPointYlow=startingPointYlow
          self.upperband=upperband
          self.lowerband=lowerband


def worker_init(q):
    # all records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    logger = logging.getLogger('api.sub')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)

def _linear(df ,totalCount, period = 15, deviation2 = 2):

    periodMinusOne = period - 1 
    Ex = 0
    Ey = 0.0
    Ex2 = 0.0
    Exy = 0.0
    df['close'] = df['close'].fillna(0) 
    n = totalCount
    num_candel = len(df)
    n2 = n

    for j in range (1):
        Ex = 0
        Ey = 0.0
        Ex2 = 0.0
        Exy = 0.0
        for i in range(period): 
            closeI = df['close'][num_candel - i - j- 1] 
            Ex = Ex + i 
            Ey = Ey + closeI 
            Ex2 = Ex2 + (i * i) 
            Exy = Exy + (closeI * i) 

        ExEx = Ex * Ex
        if Ex2 == ExEx:
            slope = 0.0
        else:
            slope = (period * Exy - Ex * Ey) / (period * Ex2 - ExEx)
        linearRegression = (Ey - slope * Ex) / period
        intercept = linearRegression + n2 * slope
        deviation = 0.0 

        for i in range(period):
            deviation = deviation + pow(df['close'][num_candel-i-j-1] - (intercept - slope * (n-i-j-1)), 2.0)
        deviation = deviation2 * math.sqrt(deviation / periodMinusOne)
        startingPointY = linearRegression + slope * periodMinusOne
        n2 = n2 - 1
        df_last = df[len(df)-1:]
        df_last['deviation']=deviation
        df_last['linearRegression']=linearRegression
        df_last['slope']=slope
        return df_last

def CalcResult (res:ResultModel,linearRegressionL,interceptL,slopeL,deviationL,startingPointYL,startingPointYupL,startingPointYlowL,upperbandL,lowerbandL):    
    linearRegressionL[res.index]=res.linearRegression
    interceptL[res.index]=res.intercept
    slopeL[res.index]=res.slope
    deviationL[res.index]=res.deviation
    startingPointYL[res.index]=res.startingPointY
    startingPointYupL[res.index]=res.startingPointYup
    startingPointYlowL[res.index]=res.startingPointYlow
    upperbandL[res.index]=res.upperband
    lowerbandL[res.index]=res.lowerband

def Process(j,df,n,period,num_candel,periodMinusOne,deviation2,linearRegressionLDic,interceptLDic,slopeLDic,deviationLDic,startingPointYLDic,startingPointYupLDic,startingPointYlowLDic,upperbandLDic,lowerbandLDic):
    os.environ["OPENBLAS_MAIN_FREE"] = "1"
    startTimer=datetime.datetime.now()
    logger.info("startPLinear j:{} n:{} p:{} nc:{} pmo:{} d:{}".format(j,n,period,num_candel,periodMinusOne,deviation2))
    try:        
        Ex = 0
        Ey = 0.0
        Ex2 = 0.0 
        Exy = 0.0
        for i in range(period): 
            closeI = df['close'][num_candel - i - j- 1] 
            Ex = Ex + i 
            Ey = Ey + closeI
            Ex2 = Ex2 + (i * i) 
            Exy = Exy + (closeI * i) 

        ExEx = Ex * Ex
            
        if Ex2 == ExEx:
            slope = 0.0
        else:
            slope = (period * Exy - Ex * Ey) / (period * Ex2 - ExEx)
        linearRegression = (Ey - slope * Ex) / period
        intercept = linearRegression + (n-j) * slope
        deviation = 0.0        

        for i in range(period):
            deviation += pow(df['close'][num_candel-i-j-1] - (intercept - slope * (n-i-j-1)), 2.0)
        deviation = deviation2 * math.sqrt(deviation / periodMinusOne)
        startingPointY = linearRegression + slope * periodMinusOne
        startingPointYup = (startingPointY + deviation)
        startingPointYlow = startingPointY - deviation
        upperband = linearRegression + deviation
        lowerband = linearRegression - deviation  

        linearRegressionLDic[j]=linearRegression
        interceptLDic[j]=intercept
        slopeLDic[j]=slope
        deviationLDic[j]=deviation
        startingPointYLDic[j]=startingPointY
        startingPointYupLDic[j]=startingPointYup
        startingPointYlowLDic[j]=startingPointYlow
        upperbandLDic[j]=upperband
        lowerbandLDic[j]=lowerband
    except Exception as ex:
        logger.error("ErrorPLinear j:{} n:{} p:{} nc:{} pmo:{} d:{}".format(j,n,period,num_candel,periodMinusOne,deviation2),exc_info=True)

    stopTimer=datetime.datetime.now()
    logger.info("startPLinear time:{} j:{} n:{} p:{} nc:{} pmo:{} d:{}".format((stopTimer-startTimer),j,n,period,num_candel,periodMinusOne,deviation2))    

def backTestlinear(df ,totalCount, period = 15, deviation2 = 2):
    startTimer=datetime.datetime.now()
    
    periodMinusOne = period - 1  
    num_candel = len(df)
    df['close'] = df['close'].fillna(0) 

    logger.info("startBLinear tc:{} nc:{} d:{}".format(totalCount,num_candel,deviation2))
    try:        
        pool = mp.Pool(int(mp.cpu_count()/2)+1,worker_init,[logger])
        manager = Manager()
        linearRegressionLDic = manager.dict()
        interceptLDic = manager.dict()
        slopeLDic = manager.dict()
        deviationLDic = manager.dict()
        startingPointYLDic = manager.dict()
        startingPointYupLDic = manager.dict()
        startingPointYlowLDic = manager.dict()
        upperbandLDic = manager.dict()
        lowerbandLDic = manager.dict()

        startTimerMultiProc=datetime.datetime.now()

        for j in range(num_candel-period):
            if(j%1000==0):
                 logger.info("startBLinearMultiProcc index:{}".format(j))    
            pool.apply_async(Process, args=(j, df,totalCount,period,num_candel,periodMinusOne,deviation2,linearRegressionLDic,interceptLDic,slopeLDic,deviationLDic,startingPointYLDic,startingPointYupLDic,startingPointYlowLDic,upperbandLDic,lowerbandLDic))
        
        endTimerMultiProc=datetime.datetime.now()
        logger.info("startBLinearMultiProcc time:{} tc:{} nc:{} d:{}".format((endTimerMultiProc-startTimerMultiProc),totalCount,num_candel,deviation2))
        pool.close()
        pool.join()
        
        df = df[period:]
        df = df[::-1]    
        df['linearRegression'] =list(collections.OrderedDict(sorted(linearRegressionLDic.items())).values())
        df['slope'] = list(collections.OrderedDict(sorted(slopeLDic.items())).values())
        df['deviation'] = list(collections.OrderedDict(sorted(deviationLDic.items())).values())
        df['startingPointY'] = list(collections.OrderedDict(sorted(startingPointYLDic.items())).values())
        df['startingPointYup'] = list(collections.OrderedDict(sorted(startingPointYupLDic.items())).values())
        df['startingPointYlow'] = list(collections.OrderedDict(sorted(startingPointYlowLDic.items())).values())
        df['upperband'] = list(collections.OrderedDict(sorted(upperbandLDic.items())).values())
        df['lowerband'] = list(collections.OrderedDict(sorted(lowerbandLDic.items())).values()) 
        df = df[::-1]
        df = df.reset_index()
        df['slopeup'] = 0.0
        df['slopedown'] = 0.0
        # df['slopeup'] = (df['upperband'] - df['startingPointYup']) / periodMinusOne
        # df['slopedown'] = (df['lowerband'] - df['startingPointYlow']) / periodMinusOne
        df['slopeup'] = (df['upperband'] - df['upperband'].shift(1)) 
        df['slopedown'] = (df['lowerband'] - df['lowerband'].shift(1))                 
    except Exception as ex:
        logger.error("ErrorBLinear tc:{} nc:{} d:{}".format(totalCount,num_candel,deviation2),exc_info=True)
    stopTimer=datetime.datetime.now()
    logger.info("EndAllPBLinear time:{} tc:{} nc:{} d:{}".format((stopTimer-startTimer),totalCount,num_candel,deviation2))

    return df

def backTestlinearSingelProc(df ,totalCount, period = 15, deviation2 = 2):
    logger.info("startBLinearSingel tc:{} p:{} d:{}".format(totalCount,period,deviation2))
    startTimer=datetime.datetime.now()
    periodMinusOne = period - 1  
    Ex = 0
    Ey = 0.0
    Ex2 = 0.0
    Exy = 0.0
    df['close'] = df['close'].fillna(0) 
    n = totalCount
    num_candel = len(df)
    n2 = n
    linearRegressionL = []
    interceptL = []
    slopeL = []
    deviationL = []
    startingPointYL = []
    startingPointYupL = []
    startingPointYlowL = []
    upperbandL = []
    lowerbandL = []
    try:
        for j in range (num_candel-period):
            Ex = 0
            Ey = 0.0
            Ex2 = 0.0 
            Exy = 0.0
            for i in range(period): 
                closeI = df['close'][num_candel - i - j- 1] 
                Ex = Ex + i 
                Ey = Ey + closeI
                Ex2 = Ex2 + (i * i) 
                Exy = Exy + (closeI * i) 

            ExEx = Ex * Ex
            if Ex2 == ExEx:
                slope = 0.0
            else:
                slope = (period * Exy - Ex * Ey) / (period * Ex2 - ExEx)
            linearRegression = (Ey - slope * Ex) / period
            intercept = linearRegression + n2 * slope
            deviation = 0.0 

            for i in range(period):
                deviation = deviation + pow(df['close'][num_candel-i-j-1] - (intercept - slope * (n-i-j-1)), 2.0)
            deviation = deviation2 * math.sqrt(deviation / periodMinusOne)
            startingPointY = linearRegression + slope * periodMinusOne
            startingPointYup = (startingPointY + deviation)
            startingPointYlow = startingPointY - deviation
            upperband = linearRegression + deviation
            lowerband = linearRegression - deviation
            n2 = n2 - 1
            linearRegressionL.append(linearRegression)
            interceptL.append(intercept)
            slopeL.append(slope) 
            deviationL.append(deviation)
            startingPointYL.append(startingPointY)
            startingPointYupL.append(startingPointYup)
            startingPointYlowL.append(startingPointYlow)
            upperbandL.append(upperband)
            lowerbandL.append(lowerband)
        df = df[period:]
        df = df[::-1]
        df['linearRegression'] = linearRegressionL
        df['slope'] = slopeL
        df['deviation'] = deviationL
        df['startingPointY'] = startingPointYL
        df['startingPointYup'] = startingPointYupL
        df['startingPointYlow'] = startingPointYlowL
        df['upperband'] = upperbandL
        df['lowerband'] = lowerbandL 
        df = df[::-1]
        df = df.reset_index()
        df['slopeup'] = 0.0
        df['slopedown'] = 0.0

        df['slopeup'] = (df['upperband'] - df['upperband'].shift(1)) 
        df['slopedown'] = (df['lowerband'] - df['lowerband'].shift(1)) 
    except Exception as ex:
       logger.error("ErrorBLinearSingel tc:{} nc:{} d:{}".format(totalCount,num_candel,deviation2),exc_info=True)
    stopTimer=datetime.datetime.now()
    logger.info("endBLinearSingel time:{} tc:{} p:{} d:{}".format((stopTimer-startTimer),totalCount,period,deviation2))
    return df
