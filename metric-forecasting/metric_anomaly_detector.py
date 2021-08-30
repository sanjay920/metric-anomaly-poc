import os
import logging 
from datetime import datetime
import pandas as pd
from datetime import datetime
from queue import Queue
import time
import json
import os
# from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from metric_model import MetricModel

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
        self.pred_history = []
        self.alert_counter = 0
        self.metric_model = MetricModel()

    def verify_new_data(self, data):
        xs_new, y_new = data[0]["value"]
        xs_new = datetime.fromtimestamp(float(xs_new)).strftime("%Y-%m-%d %H:%M:%S")
        y_new = float(y_new)
        # g.set(y_new)
        ## check the prediction for y_new to verify if it's anomaly?
        json_payload = {"time" : xs_new,"is_anomaly": False, "metric_name":self.metric_name,
                             "alert_score" : 0, "y": y_new } 
        if len(self.metric_xs) >= MIN_DATA_SIZE: ## which means there's a prediction for this datapoint
            y_pred, y_pred_low, y_pred_high = self.pred_history[len(self.metric_xs) - MIN_DATA_SIZE]
            is_anomaly = True if ( y_new < y_pred_low or y_new > y_pred_high) else False

            ## rule for alerting. 
            '''
            Two kinds of alert decision rules:
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
            if is_anomaly == True:
                self.alert_counter += 1
            else:
                self.alert_counter -= 2
            self.alert_counter = max(self.alert_counter, 0)
            if self.alert_counter >= 1:
                logger.fatal(f"Alert at time : {xs_new}")
            json_payload["yhat"] = y_pred
            json_payload["yhat_lower"] =  y_pred_low
            json_payload["yhat_upper"] =  y_pred_high
            json_payload["confidence_score"] = 1 ##TODO
            json_payload["is_anomaly"] = is_anomaly
            json_payload["alert_score"] = self.alert_counter
        
        # push_to_gateway('http://localhost:9796', job='batchA', registry=registry)

        ## TODO: add compute_vol() to better make desicion of handling anomaly data points for training.
        ## include or exclude or quarantine this new data?
        #y_train =y_new if is_anomaly==False else np.NaN 
        y_train = y_new
        self.metric_xs.append(xs_new)
        self.metric_y.append(y_train)
        logger.debug(json_payload)
        return json_payload

    def fit_model(self):
        ## modeling
        if len(self.metric_xs) >= MIN_DATA_SIZE and (len(self.metric_xs) - MIN_DATA_SIZE) % RETRAINING_INTERVAL_MINUTE == 0: ## TODO: better rule
            training_dataseries = pd.Series(self.metric_y, self.metric_xs)
            preds, lower_bounds, upper_bounds = self.metric_model.train_predict(training_dataseries)
            for p in zip(preds, lower_bounds, upper_bounds):
                self.pred_history.append(p)
            ## the preds should be persistent to a DB such as influxDB and then visualized in Grafana.
