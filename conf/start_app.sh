#!/bin/bash

scheduler_pid=$(pgrep -f "dask-scheduler")
[ -z "$scheduler_pid" ] && dask-scheduler --port 8786 & sleep 5

worker_pid=$(pgrep -f "dask-worker tcp://localhost:8786")
[ -z "$worker_pid" ] && dask-worker tcp://localhost:8786 --memory-limit 8GB & sleep 5

gunicorn tide_app:app -w 4 -k uvicorn.workers.UvicornWorker -b 127.0.0.1:8040 --keyfile conf/privkey.pem --certfile conf/fullchain.pem --timeout 180 --reload
