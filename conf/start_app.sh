#!/bin/bash

gunicorn tide_app:app -w 2 -k uvicorn.workers.UvicornWorker -b 127.0.0.1:8040 --keyfile conf/privkey.pem --certfile conf/fullchain.pem --timeout 180 --reload
