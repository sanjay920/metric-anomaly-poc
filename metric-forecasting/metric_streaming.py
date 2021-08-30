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
IS_LOCAL = True if PROMETHEUS_ENDPOINT == "http://localhost:9090" else False

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(LOGGING_LEVEL)

prom = PrometheusConnect(url=PROMETHEUS_ENDPOINT, disable_ssl=True)
LOOP_TIME_SECOND = float(60.0) # unit: second, type: float


async def update_metrics(inference_queue):
    if not IS_LOCAL:
        nw = NatsWrapper()
        await nw.connect()
    ## orderedDict?

    metrics_list = ["cpu_usage", "memory_usage", "disk_usage"] ## TODO: default metrics and their queries should be configured in a file.
    mad = {}
    for m in metrics_list:
        mad[m] = MetricAnomalyDetector(m)
    while True:
        new_data = await inference_queue.get()
        starttime = time.time()
        for m in metrics_list:
            if len(new_data[m]) == 0:
                continue
            json_payload = mad[m].verify_new_data(new_data[m])
            if not IS_LOCAL:
                await nw.publish(nats_subject="forecasted_metric_bounds", payload_df=json.dumps(json_payload).encode()) ## plan to change the subjact to "forecasted_metrics"
            mad[m].fit_model()
        logger.debug(f"anomaly detection time spent : {time.time() - starttime}")
        

async def scrape_prometheus_metrics(inference_queue):
    starttime = time.time()
    while True:
        thistime = time.time()
        ## TODO: should have a preprocessing service to scrape metrics from prometheus every minute, and fillin missing values if necessarys
        current_cpu_usage_data = prom.custom_query(query='1- (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])))')
        memory_usage = prom.custom_query(query='100 * (1 - sum(node_memory_MemAvailable_bytes) / sum(node_memory_MemTotal_bytes))')
        disk_usage = prom.custom_query(query='(sum(node_filesystem_size_bytes{device!~"rootfs|HarddiskVolume.+"})- sum(node_filesystem_free_bytes{device!~"rootfs|HarddiskVolume.+"})) / sum(node_filesystem_size_bytes{device!~"rootfs|HarddiskVolume.+"}) * 100 ')
        inference_queue_payload = {"cpu_usage": current_cpu_usage_data, "memory_usage" : memory_usage, "disk_usage" : disk_usage}
        await inference_queue.put(inference_queue_payload)
        logger.debug(f"query time spent: {time.time() - thistime}")
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