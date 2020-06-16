from datetime import datetime

rCol = '\033[0m'

tCol = dict()
tCol['k'] = '\033[30m'
tCol['black'] = '\033[30m'
tCol['r'] = '\033[31m'
tCol['red'] = '\033[31m'
tCol['g'] = '\033[32m'
tCol['green'] = '\033[32m'
tCol['y'] = '\033[33m'
tCol['yellow'] = '\033[33m'
tCol['b'] = '\033[34m'
tCol['blue'] = '\033[34m'
tCol['m'] = '\033[35m'
tCol['magenta'] = '\033[35m'
tCol['c'] = '\033[36m'
tCol['cyan'] = '\033[36m'
tCol['w'] = '\033[37m'
tCol['white'] = '\033[37m'
tCol['br'] = '\033[91m'
tCol['brightred'] = '\033[91m'
tCol['bg'] = '\033[92m'
tCol['brightgreen'] = '\033[92m'
tCol['by'] = '\033[93m'
tCol['brightyellow'] = '\033[93m'
tCol['bb'] = '\033[94m'
tCol['brightblue'] = '\033[94m'
tCol['bm'] = '\033[95m'
tCol['brightmagenta'] = '\033[95m'
tCol['bc'] = '\033[96m'
tCol['brightcyan'] = '\033[96m'
tCol['bw'] = '\033[97m'
tCol['brightwhite'] = '\033[97m'

bCol = dict()
bCol['k'] = '\033[40m'
bCol['black'] = '\033[40m'
bCol['r'] = '\033[41m'
bCol['red'] = '\033[41m'
bCol['g'] = '\033[42m'
bCol['green'] = '\033[42m'
bCol['y'] = '\033[43m'
bCol['yellow'] = '\033[43m'
bCol['b'] = '\033[44m'
bCol['blue'] = '\033[44m'
bCol['m'] = '\033[45m'
bCol['magenta'] = '\033[45m'
bCol['c'] = '\033[46m'
bCol['cyan'] = '\033[46m'
bCol['w'] = '\033[47m'
bCol['white'] = '\033[47m'
bCol['br'] = '\033[101m'
bCol['brightred'] = '\033[101m'
bCol['bg'] = '\033[102m'
bCol['brightgreen'] = '\033[102m'
bCol['by'] = '\033[103m'
bCol['brightyellow'] = '\033[103m'
bCol['bb'] = '\033[104m'
bCol['brightblue'] = '\033[104m'
bCol['bm'] = '\033[105m'
bCol['brightmagenta'] = '\033[105m'
bCol['bc'] = '\033[106m'
bCol['brightcyan'] = '\033[106m'
bCol['bw'] = '\033[107m'
bCol['brightwhite'] = '\033[107m'

def FloatToDatetime(fl):
    return datetime.fromtimestamp(fl)

def DatetimeToFloat(d):
    return d.timestamp()

class Logger:

    def __init__(self):
        self.timeInit = datetime.now().timestamp()

    def PrintDebug(self, msg, current = True, elapsed = True, col='', bg=''):
        t = ""
        if current:
            t += str(datetime.now())[5:-3]
        
        if elapsed:
            et = datetime.now().timestamp() - self.timeInit
            etMS = int(et * 1000) % 1000
            etS = int(et % 60)
            etM = int((et / 60) % 60)
            etH = int(et / 3600)
            t += "[%02d:%02d:%02d.%3d]"%(etH,etM,etS,etMS)
        
        if current or elapsed:
            t += ":"

        if col != '':
            t += tCol[col]

        if bg != '':
            t += bCol[bg]

        t += msg + rCol

        print(t)

logger = Logger()
