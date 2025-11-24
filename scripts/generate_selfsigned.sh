#!/usr/bin/env bash
set -euo pipefail
mkdir -p reverse-proxy/ssl
openssl req -x509 -nodes -newkey rsa:2048 -days 90 \
  -keyout reverse-proxy/ssl/privkey.pem \
  -out   reverse-proxy/ssl/fullchain.pem \
  -subj "/CN=pizia.disi.unitn.it"
echo "Self-signed certs written to reverse-proxy/ssl/"
