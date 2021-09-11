import os
import logging 
from datetime import datetime
import pandas as pd
from datetime import datetime
from queue import Queue
import time
import json
import os
import numpy as np
# from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from metric_model import MetricModel
import ruptures as rpt

RETRAINING_INTERVAL_MINUTE = 5
ROLLING_TRAINING_SIZE = 1440
MIN_DATA_SIZE = 10

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

    ## TODELETE 
# registry = CollectorRegistry()
# g = Gauge('opni_metrics_y', 'des',  registry=registry)

class MetricAnomalyDetector:

    def __init__(self, metric_name):
        self.metric_name =   metric_name
        self.metric_xs = []
        self.metric_y = []
        self.metric_realy = []
        self.pred_history = []
        self.alert_counter = 0
        self.metric_model = MetricModel()
        self.anomaly_history = []
        self.start_point = 0

    def verify_new_data(self, data):
        xs_raw, y_raw = data[0]["value"]
        xs_new = datetime.fromtimestamp(float(xs_raw)).strftime("%Y-%m-%d %H:%M:%S")
        y_new = float(y_raw)
        ts = datetime.fromtimestamp(xs_raw) # example format : '2019-03-13T12:02:49'
        # g.set(y_new)
        ## check the prediction for y_new to verify if it's anomaly?
        is_anomaly = 0
        is_alert = 0
        json_payload = {"timestamp" : ts,"is_anomaly": is_anomaly, "metric_name":self.metric_name,
                             "alert_score" : 0, "is_alert":is_alert, "y": y_new,
                        "yhat": y_new, "yhat_lower": y_new, "yhat_upper": y_new, "confidence_score": 0
                              } 
        if len(self.metric_xs) >= MIN_DATA_SIZE: ## which means there's a prediction for this datapoint
            y_pred, y_pred_low, y_pred_high = self.pred_history[len(self.metric_xs)]
            if self.metric_name != "disk_usage":
                is_anomaly = 1 if ( y_new < y_pred_low or y_new > y_pred_high) else 0
            else:
                is_anomaly = 1 if y_new >= 80 else 0

            ## rule for alerting. 
            '''
            Two kinds of alert decision rules: ref: https://github.com/nfrumkin/forecast-prometheus/blob/master/notebooks/Anomaly%20Detection%20Decision%20Rules.ipynb
                1. The Accumulator. 
                    The accumulator detection rules is based on the assumption that anomalous data is persistent. 
                    Rather than detecting anomalies point-wise, we have a running counter which increments when a point is flagged as anomalous and
                     decremenets by 2 when a normal point is detected. If the counter reaches a certain threshhold value, then we raise a flag.
                2. The Tail Probability
                    This anomaly detection rule uses the idea that the recent past's noise is comparable to the current noise. 
                    Using the gaussian noise assumption, we calculate the tail probability that that the mean of the current values is comparable
                     to the values in the recent past.
                    The implementation can be found in this paper: https://arxiv.org/pdf/1607.02480.pdf
            To emsemble them, we can either take the intersect or the union.
            '''
            if is_anomaly == 1:
                self.alert_counter += 1
            else:
                if self.alert_counter > 0:
                    self.alert_counter -= 2
                    # if self.alert_counter <= 0: ## if it gets back to normal, auto-correct the outlier value. TODO: test this.
                    #     for i1 in range(-5, 0):
                    #         if self.anomaly_history[i1] == 1:
                    #             #self.metric_y[i1] = y_new ## replace outlier with y_new.
                    #             if self.metric_y[i1] > self.pred_history[i1][2]: ## replace with yhat_upper or yhat_lower
                    #                 self.metric_y[i1] = self.pred_history[i1][2]
                    #             elif self.metric_y[i1] < self.pred_history[i1][1]:
                    #                 self.metric_y[i1] = self.pred_history[i1][1]

            self.alert_counter = max(self.alert_counter, 0)
            if self.alert_counter >= 5:
                is_alert = 1
                logger.fatal(f"Alert for {self.metric_name} at time : {xs_new}")
 
            json_payload["yhat"] = y_pred
            json_payload["yhat_lower"] =  max(0, y_pred_low) #y_pred_low
            json_payload["yhat_upper"] =  y_pred_high
            json_payload["confidence_score"] = 1 ##TODO
            json_payload["is_anomaly"] = is_anomaly
            json_payload["is_alert"] = is_alert
            json_payload["alert_score"] = self.alert_counter

        else: ## cornercase for cold start period (first 10 mins)
            self.pred_history.append((y_new, y_new, y_new))
        # push_to_gateway('http://localhost:9796', job='batchA', registry=registry)

        self.anomaly_history.append(is_anomaly) 
        if self.metric_name == "cpu_usage" and is_alert == 0 and is_anomaly == 1:
            logger.debug("fixed anomaly value.")
            if y_new < y_pred_low:
                y_train = y_pred_low
            else: # y_new > y_pred_high
                y_train = y_pred_high
        else:
            y_train = y_new
        self.metric_xs.append(xs_new)
        self.metric_y.append(y_train)
        self.metric_realy.append(y_new)

        ## TODO: add CPD(change points detection) or compute_vol() to better make desicion of handling anomaly data points for training.
        ## include or exclude or quarantine this new data?
        #y_train =y_new if is_anomaly==0 else np.NaN 
        trace_back_time = 30
        if len(self.anomaly_history) >= trace_back_time and self.anomaly_history[-trace_back_time] == 1 and self.metric_name == "cpu_usage":
            training_y = self.metric_y[self.start_point:]
            if len(training_y) >= 2 * trace_back_time:
                cpd = rpt.Pelt(model="rbf").fit(np.array(training_y))
                change_locations = (cpd.predict(pen=10))[:-1] # remove last one because it's always the end of array so it's meaningless
                for l in reversed(change_locations):
                    if abs(l - len(training_y) + trace_back_time) <= 5:
                        logger.debug(f"reset start_point from {self.metric_xs[self.start_point]} to {self.metric_xs[self.start_point + l]}")
                        self.start_point = self.start_point + l
                        break

        #logger.debug(json_payload)
        return json_payload

    def fit_model(self):
        ## modeling
        if len(self.metric_xs) >= MIN_DATA_SIZE and (len(self.metric_xs) - MIN_DATA_SIZE) % RETRAINING_INTERVAL_MINUTE == 0: ## TODO: better rule
            training_dataseries = pd.Series(self.metric_y, self.metric_xs)
            ## max training size should be [ROLLING_TRAINING_SIZE]
            if len(self.metric_y) - self.start_point > ROLLING_TRAINING_SIZE:
                self.start_point = len(self.metric_y) - ROLLING_TRAINING_SIZE
            training_dataseries = training_dataseries[self.start_point:] 

            preds, lower_bounds, upper_bounds = self.metric_model.train_predict(training_dataseries)
            for p in zip(preds, lower_bounds, upper_bounds):
                self.pred_history.append(p)
            ## the preds should be persistent to a DB such as influxDB and then visualized in Grafana.
