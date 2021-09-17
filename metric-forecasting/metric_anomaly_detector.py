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
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionTimeout
from elasticsearch.helpers import BulkIndexError, async_streaming_bulk
from metric_model import ArimaModel
import ruptures as rpt

RETRAINING_INTERVAL_MINUTE = 1
ROLLING_TRAINING_SIZE = 1440
MIN_DATA_SIZE = 10
ALERT_THRESHOLD = 5
CPD_TRACE_BACK_TIME = 30 ## minutes

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

ES_ENDPOINT = os.getenv("ES_ENDPOINT", "https://localhost:9200")
ES_USERNAME = os.getenv("ES_USERNAME", "admin")
ES_PASSWORD = os.getenv("ES_PASSWORD", "admin")

ES_RESERVED_KEYWORDS = {
    "_id",
    "_index",
    "_if_seq_no",
    "_if_primary_term",
    "_parent",
    "_percolate",
    "_retry_on_conflict",
    "_routing",
    "_timestamp",
    "_type",
    "_version",
    "_version_type",
}

class MetricAnomalyDetector:

    def __init__(self, metric_name):
        self.metric_name =  metric_name
        self.metric_xs = []
        self.metric_y = []
        self.metric_rawy = []
        self.pred_history = []
        self.alert_counter = 0
        self.metric_model = ArimaModel()
        self.anomaly_history = []
        self.start_index = 0 # start index of training data. update by CPD.

    def load(self, history_data):
        for h in history_data:
            ## TODO: fillin the time gap with null datapoint
            d = h["_source"]
            self.metric_xs.append(d["timestamp"])
            self.metric_y.append(d["y"])
            self.metric_rawy.append(d["y"])
            self.pred_history.append((d["yhat"], d["yhat_lower"], d["yhat_upper"]))
            self.anomaly_history.append(d["is_anomaly"])
            ## should also save/load alerts scores and is_alert?
        logger.debug(f"history data loaded : {len(self.metric_xs)}")
        ## TODO: also need to fill missing data ?


    def run(self, data):
        self.train_and_predict()
        xs_raw, y_raw = data[0]["value"]
        xs_raw = int(xs_raw) // 60 * 60 ## so it always at :00 second
        xs_new = datetime.fromtimestamp(float(xs_raw)).isoformat() # example format : '2019-03-13T12:02:49'
        y_new = float(y_raw)
        ## check the prediction for y_new to verify if it's anomaly?
        is_anomaly = 0
        is_alert = 0
        json_payload = {"timestamp" : xs_new,"is_anomaly": is_anomaly, "metric_name":self.metric_name,
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

            self.alert_counter = max(self.alert_counter, 0)
            if self.alert_counter >= ALERT_THRESHOLD:
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

        self.anomaly_history.append(is_anomaly) 
        if self.metric_name == "cpu_usage" and is_alert == 0 and is_anomaly == 1:
            logger.debug("fix anomaly value.")
            if y_new < y_pred_low:
                y_train = y_pred_low
            else: # y_new > y_pred_high
                y_train = y_pred_high
        else:
            y_train = y_new
        self.metric_xs.append(xs_new)
        self.metric_y.append(y_train)
        self.metric_rawy.append(y_new)


        if len(self.anomaly_history) >= CPD_TRACE_BACK_TIME and self.anomaly_history[-CPD_TRACE_BACK_TIME] == 1 and self.metric_name == "cpu_usage":
            self.change_point_detection()

        #logger.debug(json_payload)
        return json_payload

    def change_point_detection(self):
        training_y = self.metric_y[self.start_index:]
        if len(training_y) >= 2 * CPD_TRACE_BACK_TIME:
            cpd = rpt.Pelt(model="rbf").fit(np.array(training_y)) ## change point detection. maybe volatility is another option?
            change_locations = (cpd.predict(pen=10))[:-1] # remove last one because it's always the end of array so it's meaningless
            for l in reversed(change_locations):
                if abs(l - len(training_y) + CPD_TRACE_BACK_TIME) <= 5: # the change point should near to an anomaly?
                    logger.debug(f"reset start_index from {self.metric_xs[self.start_index]} to {self.metric_xs[self.start_index + l]}")
                    self.start_index = self.start_index + l
                    break

    def train_and_predict(self):
        ## modeling
        if len(self.metric_xs) >= MIN_DATA_SIZE: #and (len(self.metric_xs) - MIN_DATA_SIZE) % RETRAINING_INTERVAL_MINUTE == 0: ## TODO: better rule
            training_dataseries = pd.Series(self.metric_y, self.metric_xs).asfreq(freq='T')
            ## max training size should be [ROLLING_TRAINING_SIZE]
            if len(self.metric_y) - self.start_index > ROLLING_TRAINING_SIZE:
                self.start_index = len(self.metric_y) - ROLLING_TRAINING_SIZE
            training_dataseries = training_dataseries[self.start_index:] 

            self.metric_model.train(training_dataseries)

            preds, lower_bounds, upper_bounds = self.metric_model.predict()
            for p in zip(preds, lower_bounds, upper_bounds):
                self.pred_history.append(p)
            ## the preds should be persistent to a DB such as influxDB and then visualized in Grafana.
