#!/usr/bin/python3
# run from source directory
import http.client
import json
import ssl
 
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

        # Create a connection to submit HTTP requests
        self.connection = http.client.HTTPSConnection(host=host, port=port, context=self.context)

    def post(self, url):
        return self._request('POST', url)

    def get(self, url):
        return self._request('GET', url)

    def _request(self, method, url):
        self.connection.request(method="POST", url=url, headers=request_headers, body=json.dumps(request_body_dict))
        response = self.connection.getresponse()
        return response.read()
    
def issue_post():
    # Define the client certificate settings for https connection
    context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
    context.load_cert_chain(certfile=certificate_file, keyfile=certificate_secret)
 
    # Create a connection to submit HTTP requests
    connection = http.client.HTTPSConnection(host, port=9999, context=context)
 
    # Use connection to submit a HTTP POST request
    connection.request(method="POST", url='/pool/foobar', headers=request_headers, body=json.dumps(request_body_dict))
 
    # Print the HTTP response from the IOT service endpoint
    response = connection.getresponse()
    print(response.status, response.reason)
    data = response.read()
    print(data)



from multiprocessing import Process

if __name__ == '__main__':
    c = Connection()
    r = c.get('/pools')
    print(r)
    
    


    # pids = []

    # while(len(pids) < 1000):
    #     p = Process(target=issue_post)
    #     p.start()
    #     pids.append(p)

    # for p in pids:
    #     p.join()
