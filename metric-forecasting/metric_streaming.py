import asyncio
import logging
import pandas as pd
from datetime import datetime
from queue import Queue
import time
import json
import os
from prometheus_api_client import PrometheusConnect
from opni_nats import NatsWrapper
from metric_anomaly_detector import MetricAnomalyDetector

PROMETHEUS_ENDPOINT = os.getenv( "PROMETHEUS_ENDPOINT", "http://localhost:9090")

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

prom = PrometheusConnect(url=PROMETHEUS_ENDPOINT, disable_ssl=True)
LOOP_TIME_SECOND = float(60.0) # unit: second, type: float


async def update_metrics(inference_queue):
    nw = NatsWrapper()
    await nw.connect()
    ## orderedDict?

    mad= MetricAnomalyDetector()
    while True:
        new_data = await inference_queue.get()
        if len(new_data["metric_payload"]) == 0:
            continue
        logger.debug(f"Got new payload {new_data}")
        json_payload = mad.verify_new_data(new_data)
        await nw.publish(nats_subject="forecasted_metric_bounds", payload_df=json.dumps(json_payload).encode())
        mad.fit_model()
        

async def scrape_prometheus_metrics(inference_queue):
    starttime = time.time()
    while True:
        ## TODO: should have a preprocessing service to scrape metrics from prometheus every minute, and fillin missing values if necessarys
        current_cpu_usage_data = prom.custom_query(query='1 - (avg(irate({__name__=~"node_cpu_seconds_total|windows_cpu_time_total",mode="idle"}[2m])))')
        inference_queue_payload = {"metric_payload": current_cpu_usage_data}
        logger.debug(datetime.fromtimestamp(float(current_cpu_usage_data[0]["value"][0])).strftime("%Y-%m-%d %H:%M:%S"))
        await inference_queue.put(inference_queue_payload)
        await asyncio.sleep(LOOP_TIME_SECOND - ((time.time() - starttime) % LOOP_TIME_SECOND))

    

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    inference_queue = asyncio.Queue(loop=loop)
    prometheus_scraper_coroutine = scrape_prometheus_metrics(inference_queue)
    update_metrics_coroutine = update_metrics(inference_queue)

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