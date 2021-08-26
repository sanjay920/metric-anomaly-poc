import asyncio
import logging
import pandas as pd
from datetime import datetime
from queue import Queue
import time
import json
import os
from prometheus_api_client import PrometheusConnect
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels as sm
from opni_nats import NatsWrapper

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

PROMETHEUS_ENDPOINT = os.getenv( "PROMETHEUS_ENDPOINT", "http://localhost:9090")

RETRAINING_INTERVAL_MINUTE = 5
ROLLING_TRAINING_SIZE = 1440
MIN_DATA_SIZE = 10
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

prom = PrometheusConnect(url=PROMETHEUS_ENDPOINT, disable_ssl=True)
LOOP_TIME_SECOND = float(60.0) # unit: second, type: float

confidence_interval_level=3 ## should be 2(95%) or 3(99.7%) or 2.57(99%)


# nw = NatsWrapper()

async def update_metrics(inference_queue):
    ## orderedDict?
    metric_xs = []
    metric_y = []
    pred_history = []
    alert_counter = 0

    ## paramters for SARIMAX
    order = (1,1,1)
    seasonal_order= (0,0,0,0)

    ## TODELETE 
    registry = CollectorRegistry()
    g = Gauge('opni_metrics_y', 'des',  registry=registry)

    while True:
        new_data = await inference_queue.get()
        if len(new_data["metric_payload"]) == 0:
            continue
        logger.debug(f"Got new payload {new_data}")
        xs_new, y_new = new_data["metric_payload"][0]["value"]
        xs_new = datetime.fromtimestamp(float(xs_new)).strftime("%Y-%m-%d %H:%M:%S")
        y_new = float(y_new)
        g.set(y_new)
        ## check the prediction for y_new to verify if it's anomaly?
        if len(metric_xs) >= MIN_DATA_SIZE:
            y_pred, y_ci_low, y_ci_high = pred_history[len(metric_xs) - MIN_DATA_SIZE]
            is_anomaly = True if ( y_new < y_ci_low or y_new > y_ci_high) else False

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
                alert_counter += 1
            else:
                alert_counter -= 2
            alert_counter = max(alert_counter, 0)
            if alert_counter >= 1:
                logger.fatal(f"Alert at time : {xs_new}")
            logger.info(f"is_anomaly: {is_anomaly}, y:{y_new}, y_pred:{y_pred}, interval:{y_ci_low}-{y_ci_high}")
            json_payload = {"is_anomaly": is_anomaly, "yhat_lower": y_ci_low, "yhat_upper": y_ci_high, "metric_name": "cluster:node_cpu:sum_rate5m",
                             "alert_score" : alert_counter, "y": y_new, "y_hat":y_pred, "confidence_score": 0 }     
            #await nw.publish(nats_subject="forecasted_metric_bounds", payload_df=json.dumps(json_payload).encode())
        push_to_gateway('http://localhost:9796', job='batchA', registry=registry)

        ## TODO: add compute_vol() to better make desicion of handling anomaly data points for training.
        #y_train =y_new if is_anomaly==False else np.NaN 
        y_train = y_new
        metric_xs.append(xs_new)
        metric_y.append(y_train)
        ## modeling
        if len(metric_xs) >= MIN_DATA_SIZE and (len(metric_xs) - MIN_DATA_SIZE) % RETRAINING_INTERVAL_MINUTE == 0:
            df1 = pd.Series(metric_y, metric_xs)
            myfit = SARIMAX(df1, order=order, seasonal_order=seasonal_order).fit(disp=False)
            alpha = 0.003 if confidence_interval_level == 3 else 0.05 if confidence_interval_level == 2 else 0.01
            fc_series = myfit.forecast(RETRAINING_INTERVAL_MINUTE)
            intervals = myfit.get_forecast(RETRAINING_INTERVAL_MINUTE).conf_int(alpha=alpha)
            lower_series, upper_series = intervals["lower y"], intervals["upper y"]
            for p in zip(fc_series, lower_series, upper_series):
                pred_history.append(p)
            ## the preds should be sent to a DB such as influxDB and then visualized in Grafana.
        


async def scrape_prometheus_metrics(inference_queue):
    starttime = time.time()
    while True:
        ## TODO: should have a preprocessing service to scrape metrics from prometheus every minute, and fillin missing values if necessarys
        current_cpu_usage_data = prom.custom_query(query='1 - (avg(irate({__name__=~"node_cpu_seconds_total|windows_cpu_time_total",mode="idle"}[2m])))')
        inference_queue_payload = {"metric_payload": current_cpu_usage_data}
        logger.debug(datetime.fromtimestamp(float(current_cpu_usage_data[0]["value"][0])).strftime("%Y-%m-%d %H:%M:%S"))
        await inference_queue.put(inference_queue_payload)
        await asyncio.sleep(LOOP_TIME_SECOND - ((time.time() - starttime) % LOOP_TIME_SECOND))

# async def init_nats():
#     logging.info("Attempting to connect to NATS")
#     await nw.connect()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    inference_queue = asyncio.Queue(loop=loop)
    prometheus_scraper_coroutine = scrape_prometheus_metrics(inference_queue)
    update_metrics_coroutine = update_metrics(inference_queue)

    # task = loop.create_task(init_nats())
    # loop.run_until_complete(task)

    loop.run_until_complete(
        asyncio.gather(
            prometheus_scraper_coroutine,
            update_metrics_coroutine
        )
    )
    try:
        loop.run_forever()
    finally:
        loop.close()