#!/usr/bin/python3
# run from source directory
import http.client
import socket
import json
import ssl
import base64
import logging

# Defining certificate related stuff and host of endpoint
certificate_file = './certs/client/client.crt'
certificate_secret= './certs/client/client.key'
#host = 'localhost'
 
request_headers = {
    'Content-Type': 'application/json'
}
request_body_dict={
    'Temperature': 38,
    'Humidity': 80
}


class Connection:
    '''
    HTTPS/SSL connection class
    '''
    def __init__(self, host='localhost', port=9999):
        self.context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
        self.context.load_cert_chain(certfile=certificate_file, keyfile=certificate_secret)
        
        self.connection = http.client.HTTPSConnection(host=host, port=port, context=self.context, timeout=100)
        
        print("http.client Lib:", http.client.__file__)

        orig_connect = http.client.HTTPSConnection.connect
        def monkey_connect(self):            
            orig_connect(self)
            print(self)
        http.client.HTTPSConnection.connect = monkey_connect            
#             
#             print(dir(http.client.HTTPSConnection.socket_options))
# # #            self.sock.setsockopt(…)


        # Create a connection to submit HTTP requests

#        self.connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        self.get('/pools')
        self.get('/pools')
        self.get('/pools')
                
    def post(self, url):
        return self._request('POST', url)

    def get(self, url):
        return self._request('GET', url)

    def _request(self, method, url):
        hdr = headers = {"Content-Type":"application/x-www-form-urlencoded", "Connection":"Keep-Alive"}
        self.connection.request(method=method, url=url, headers=hdr)

        #, headers=request_headers, body=json.dumps(request_body_dict))
        response = self.connection.getresponse()
        print(response)
        return json.loads(response.read())
    
def build_url_put(pool_handle, key, value ):
    # e.g., /put?pool=939392092&key=foobar&value=Zim 
    url = '/put?pool=' + pool_handle['session'] + '&key=' + key + '&value=' + value
    print(url)
    return url
    
from multiprocessing import Process

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    c = Connection()
    print(c.get('/pools'))
    pool = c.post('/pools/mypool?sizemb=128')
    print(pool)
    print(c.get('/pools'))
#    c.post(build_url_put(pool, 'K', 'Hello'))

    
    


    # pids = []

    # while(len(pids) < 1000):
    #     p = Process(target=issue_post)
    #     p.start()
    #     pids.append(p)

    # for p in pids:
    #     p.join()
