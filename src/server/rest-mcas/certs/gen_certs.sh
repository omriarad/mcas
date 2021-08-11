#!/bin/bash

# This script was executed from inside the cas directory seen in the C++ code above

# Generate root CA
echo "Generating rootCA"
mkdir rootCA
cd rootCA
openssl genrsa -out rootCA.key 4096
openssl req -x509 -new -nodes -key rootCA.key -subj "/C=NL/ST=UTRECHT/O=Some company/CN=rootca" -sha256 -days 1024 -out rootCA.crt
cd ..

# Generate server certificate
echo "\nGenerating server certificate"
mkdir server
cd server
openssl genrsa -out server.key 2048
openssl req -new -sha256 -key server.key -subj "/C=NL/ST=UTRECHT/O=Some company/CN=Some ip" -out server.csr
openssl x509 -req -in server.csr -CA ../rootCA/rootCA.crt -CAkey ../rootCA/rootCA.key -CAcreateserial -out server.crt -days 500 -sha256
cd ..

# Generate client certificate
echo "\nGenerating client certificate"
mkdir client
cd client
openssl genrsa -out client.key 2048
openssl req -new -sha256 -key client.key -subj "/C=NL/ST=UTRECHT/O=Some company/CN=Some ip" -out client.csr
openssl x509 -req -in client.csr -CA ../rootCA/rootCA.crt -CAkey ../rootCA/rootCA.key -CAcreateserial -out client.crt -days 500 -sha256
cd ..
