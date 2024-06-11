from subprocess import Popen
import os
import time

def start_mlflow_server():
    cmd = [
        'mlflow', 'server',
        '--backend-store-uri', 'sqlite:///mlflow.db',
        '--default-artifact-root', './mlruns',
        '--host', '127.0.0.1',
        '--port', '5000'
    ]
    process = Popen(cmd)
    return process

os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'

mlflow_server_process = start_mlflow_server()
time.sleep(5)