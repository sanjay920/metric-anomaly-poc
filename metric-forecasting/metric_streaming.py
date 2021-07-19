import asyncio
import logging
from prophet import Prophet
import pandas as pd
from datetime import datetime
from queue import Queue
import json
from prometheus_api_client import PrometheusConnect
from opni_nats import NatsWrapper

nw = NatsWrapper()
TIME_INTERVAL = 3
TRAINING_METRIC_DATASET_LIMIT = 1440
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
PROMETHEUS_ENDPOINT = "http://rancher-monitoring-prometheus.cattle-monitoring-system.svc.cluster.local:9090"
prom = PrometheusConnect(url=PROMETHEUS_ENDPOINT, disable_ssl=True)

async def fit_predict_model(dataframe,periods=15,daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False):
    prophet_model = Prophet(daily_seasonality=daily_seasonality, weekly_seasonality=weekly_seasonality, yearly_seasonality=yearly_seasonality)
    prophet_model.fit(dataframe)
    future = prophet_model.make_future_dataframe(periods=periods,freq="1MIN",include_history=False,)
    future_forecast = prophet_model.predict(future)
    future_forecast["timestamp"] = future_forecast["ds"]
    future_forecast = future_forecast.set_index("timestamp")
    return future_forecast

async def scrape_prometheus_metrics(inference_queue):
    while True:
        current_cpu_usage_data = prom.custom_query(query='(1 - avg(irate(node_cpu_seconds_total{mode="idle"}[5m]))) * 100')
        inference_queue_payload = {"metric_payload": current_cpu_usage_data}
        await inference_queue.put(inference_queue_payload)
        await asyncio.sleep(60)

async def is_anomaly(nearest_index_value, current_metric_value):
    yhat_lower = max(0,nearest_index_value['yhat_lower'].values[0])
    yhat_upper = nearest_index_value['yhat_upper'].values[0]
    yhat_value = nearest_index_value['yhat'].values[0]
    lower_std = abs(yhat_value - yhat_lower)
    upper_std = abs(yhat_upper - yhat_value)
    #yhat_lower = max(0, yhat_lower - (2 * lower_std))
    #yhat_upper += (2 * upper_std)
    return ((current_metric_value < yhat_lower) or (current_metric_value > yhat_upper)), yhat_lower, yhat_upper

async def update_metrics(inference_queue):
    current_interval_metrics = []
    future_forecast = None
    training_dataset = Queue(maxsize=TRAINING_METRIC_DATASET_LIMIT)
    training_dataset_df = pd.DataFrame([])
    while True:
        new_data = await inference_queue.get()
        logging.info("Got new payload")
        if new_data is None:
            break
        if len(new_data["metric_payload"]) == 0:
            continue
        payload_data_values = new_data["metric_payload"][0]["value"]
        date_time = datetime.fromtimestamp(float(payload_data_values[0]))
        current_metric_value = float(payload_data_values[1])
        metric_dict = {"ds": date_time.strftime("%Y-%m-%d %H:%M:%S"), "y": current_metric_value}
        current_interval_metrics.append(metric_dict)
        if len(training_dataset_df) > TIME_INTERVAL:
            nearest_index = future_forecast.index.get_loc(metric_dict["ds"], method="nearest")
            nearest_index_value = future_forecast.iloc[[nearest_index]]
            anomaly_flag, yhat_lower, yhat_upper = await is_anomaly(nearest_index_value, current_metric_value)
            json_payload = {"is_anomaly": str(anomaly_flag), "yhat_lower": yhat_lower, "yhat_upper": yhat_upper, "metric_name": "cluster:node_cpu:sum_rate5m" }
            logging.info("Published to Nats subject")
            await nw.publish(nats_subject="forecasted_metric_bounds", payload_df=json.dumps(json_payload).encode())
        if len(current_interval_metrics) == TIME_INTERVAL:
            if training_dataset.full():
                for i in range(TIME_INTERVAL):
                    training_dataset.get()
            for current_metric in current_interval_metrics:
                training_dataset.put(current_metric)
            training_dataset_df = pd.DataFrame(list(training_dataset.queue))
            future_forecast = await fit_predict_model(training_dataset_df)
            current_interval_metrics = []


async def init_nats():
    logging.info("Attempting to connect to NATS")
    await nw.connect()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    inference_queue = asyncio.Queue(loop=loop)
    prometheus_scraper_coroutine = scrape_prometheus_metrics(inference_queue)
    update_metrics_coroutine = update_metrics(inference_queue)

    task = loop.create_task(init_nats())
    loop.run_until_complete(task)

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