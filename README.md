# metric-anomaly-detection

How to run the development version of opni in your cluster:
- Clone rancher/opni 
- Switch to the branch `nats-metrics-listener`
- Download Tilt (https://tilt.dev)
- If you need a cluster, use the script `hack/create-k3d-cluster.sh`. Otherwise, if you are using an existing cluster, you must perform the following additional steps: 
  - Create a publically available repo in your docker hub account called "opni-manager"
  - Insert the following lines into the opni Tiltfile: 
    - Above the call to `docker_build_with_restart`: `default_registry('docker.io/your-user-name')`
    - Near the top of the file under the existing call to allow_k8s_contexts: `allow_k8s_contexts('your-context-name')` 
- Run `tilt up`
