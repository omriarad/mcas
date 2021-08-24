#!/bin/bash

# This script was executed from inside the cas directory seen in the C++ code above

# Generate root CA
echo "Generating rootCA"
mkdir rootCA
cd rootCA
openssl genrsa -out rootCA.key 4096
openssl req -x509 -new -nodes -key rootCA.key -subj "/C=US/ST=CA/O=IBM/CN=rootca" -sha256 -days 1024 -out rootCA.crt
cd ..

# Generate server certificate
echo "Generating server certificate"
mkdir server
cd server
openssl genrsa -out server.key 2048
openssl req -new -sha256 -key server.key -subj "/C=US/ST=CA/O=IBM" -out server.csr
openssl x509 -req -in server.csr -CA ../rootCA/rootCA.crt -CAkey ../rootCA/rootCA.key -CAcreateserial -out server.crt -days 500 -sha256
cd ..

#/CN=Some ip
# Generate client certificate
echo "Generating client certificate"
mkdir client
cd client
openssl genrsa -out client.key 2048
openssl req -new -sha256 -key client.key -subj "/C=US/ST=CA/O=IBM" -out client.csr
openssl x509 -req -in client.csr -CA ../rootCA/rootCA.crt -CAkey ../rootCA/rootCA.key -CAcreateserial -out client.crt -days 500 -sha256
cd ..


# verify
openssl x509 -in server/server.crt -text -noout
