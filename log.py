from datetime import datetime


def FloatToDatetime(fl):
    return datetime.fromtimestamp(fl)

def DatetimeToFloat(d):
    return d.timestamp()

class Logger:

    def __init__(self):
        self.timeInit = datetime.now().timestamp()

    def PrintDebug(self, msg):
        et = datetime.now().timestamp() - self.timeInit
        etMS = int(et * 1000) % 1000
        etS = int(et % 60)
        etM = int((et / 60) % 60)
        etH = int(et / 3600)
        t = "%02d:%02d:%02d.%3d"%(etH,etM,etS,etMS)
        print(str(datetime.now())[5:-3] + " [" + t + "]:" + msg)

        # print(str(datetime.now()) + " : " + msg)

logger = Logger()