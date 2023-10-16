module.exports = {
  apps : [
  {
    name: 'tide',
    script: './conf/start_app.sh',
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
