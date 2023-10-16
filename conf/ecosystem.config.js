module.exports = {
  apps : [
  {
    name: 'tide',
    script: '(scheduler_pid=$(pgrep -f "dask-scheduler"); [ -z "$scheduler_pid" ] && dask-scheduler --port 8786 & sleep 5) & (worker_pid=$(pgrep -f "dask-worker tcp://localhost:8786"); [ -z "$worker_pid" ] && dask-worker tcp://localhost:8786 --memory-limit 4GB & sleep 5) & gunicorn tide_app:app -w 4 -k uvicorn.workers.UvicornWorker -b 127.0.0.1:8040 --keyfile conf/privkey.pem --certfile conf/fullchain.pem --timeout 180 --reload',
    args: '',
    merge_logs: true,
    autorestart: true,
    log_file: "tmp/tide.outerr.log",
    out_file: "tmp/tide.log",
    error_file: "tmp/tide_err.log",
    log_date_format : "YYYY-MM-DD HH:mm Z",
    append_env_to_name: true,
    watch: false,
    max_memory_restart: '4G',
    pre_stop: "ps -ef | grep -w 'tide_app' | grep -v grep | awk '{print $2}' | xargs -r kill -9"
  }],
};
