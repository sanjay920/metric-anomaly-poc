#from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels as sm
import pandas as pd
import os
import logging

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

RETRAINING_INTERVAL_MINUTE = 5
confidence_interval_level=3 ## should be 2(95%) or 3(99.7%) or 2.57(99%)

class MetricModel:

    def __init__(self):
        ## paramters for SARIMAX
        self.order = (1,1,1)
        self.seasonal_order= (0,0,0,0)
        self.alpha = 0.003 if confidence_interval_level == 3 else 0.05 if confidence_interval_level == 2 else 0.01

    def train_predict(self, data_series: pd.Series):

        myfit = SARIMAX(data_series, order=self.order, seasonal_order=self.seasonal_order).fit(disp=False)
        fc_series = myfit.forecast(RETRAINING_INTERVAL_MINUTE)
        intervals = myfit.get_forecast(RETRAINING_INTERVAL_MINUTE).conf_int(alpha=self.alpha)
        lower_series, upper_series = intervals["lower y"], intervals["upper y"]
        return fc_series, lower_series, upper_series