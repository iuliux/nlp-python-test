import os, json, requests
from threading import Thread


class ParseMultiJob(Thread):
    def __init__(self, cacheDir, jobs, server=0, baseport=3456):
        Thread.__init__(self)
        self.cacheDir = cacheDir
        self.jobs = jobs
        self.baseport = baseport
        self.server = server

        self.nlpResults = None

    def run(self):
        p = Parser("./json-cache", self.baseport + self.server)
        self.nlpResults = [(p.parse(fpath, text), g, fpath)
                            for (text, g, fpath) in self.jobs]


class Parser:
    cacheDir = ''

    def __init__(self, cacheDir, port=3456):
        self.cacheDir = cacheDir
        self.port = port

    def parse(self, filePath, text):
        fileName = filePath.split('/')[2]
        print '[' + str(self.port) + ']', fileName
        cacheFile = self.cacheDir + "/" + fileName
        if os.path.isfile(cacheFile) == False:
            payload = {'jsonrpc': '2.0', "method": "parse", "id": 1, "params": [text]}
            headers = {'content-type': 'application/json'}
            r = requests.post('http://localhost:' + str(self.port), data=json.dumps(payload), headers=headers)
            f = open(cacheFile, 'w')
            f.write(r.text)
            f.close()

            response = json.loads(r.text)
        else:
            #print "From cache"
            f = open(cacheFile, "r")
            text = f.read()
            f.close()
            response = json.loads(text)

        return json.loads(response['result'])