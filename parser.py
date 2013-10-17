import os, json, requests

class Parser:
    cacheDir = ''

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir

    def parse(self, fileName, text):
    
        cacheFile = self.cacheDir + "/" + fileName
        if os.path.isfile(cacheFile) == False:
            payload = {'jsonrpc': '2.0', "method": "parse", "id": 1, "params": [text]}
            headers = {'content-type': 'application/json'}
            r = requests.post('http://localhost:3456', data=json.dumps(payload), headers=headers)
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