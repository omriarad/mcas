#!/usr/bin/python3
import http.client
import json
import ssl
 
# Defining certificate related stuff and host of endpoint
certificate_file = '/home/danielwaddington/mcas/src/server/rest-mcas/certs/client/client.crt'
certificate_secret= '/home/danielwaddington/mcas/src/server/rest-mcas/certs/client/client.key'
host = 'localhost'
 
request_headers = {
    'Content-Type': 'application/json'
}
request_body_dict={
    'Temperature': 38,
    'Humidity': 80
}


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
    pids = []

    while(len(pids) < 1000):
        p = Process(target=issue_post)
        p.start()
        pids.append(p)

    for p in pids:
        p.join()
