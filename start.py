import os
from datetime import datetime
import time

process_name = "wisee.py"
while datetime.now() < datetime(2021,7,6,20,30):
    tmp = os.popen("ps -Af").read()
    if "python " + process_name not in tmp[:]:
        print("The process is not running. Let's restart.")
        newprocess="nohup python %s >/dev/null 2>&1 &" % (process_name)
        os.system(newprocess) 
    else:
        print("running...")
        time.sleep(60)
