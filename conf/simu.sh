# install packages
pip3 install -r ./requirements.txt

# run dask-scheduler and dask-worker
dask-scheduler --port 8786 &
dask-worker tcp://localhost:8786 --memory-limit 8GB &

# run API server
## localhost: gunicorn tide_app:app -k uvicorn.workers.UvicornWorker -b 127.0.0.1:8040 --timeout 120
gunicorn tide_app:app -w 4 -k uvicorn.workers.UvicornWorker -b 127.0.0.1:8040 --keyfile conf/privkey.pem --certfile conf/fullchain.pem --reload --timeout 180

# kill process
ps -ef | grep 'tide_app' | grep -v grep | awk '{print $2}' | xargs -r kill -9

# kill dask
ps -ef | grep -w 'dask-scheduler' | grep -v grep | awk '{print $2}' | xargs -r kill -9 && ps -ef | grep -w 'dask-worker' | grep -v grep | awk '{print $2}' | xargs -r kill -9 && ps -ef | grep -w 'tide_app' | grep -v grep | awk '{print $2}' | xargs -r kill -9

# pm2 start
pm2 start ./conf/ecosystem.config.js

## use pipenv
pipenv update

# update requirements.txt
pipreqs --force ./

# make local development code updated without reinstalling the package
pip install -e .
