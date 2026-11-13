#!/bin/zsh
# K-EXAONE 2.5 Stage 1 (c0 + adult, 1,250 프롬프트 x k; 기본 10) 본실행. 끊기면 같은 명령으로 이어서 돈다.
cd "$(dirname "$0")"
set -a; . ./.env.local; set +a
exec caffeinate -i uv run --no-project --python 3.12 --with certifi sh -c \
  "SSL_CERT_FILE=\$(python3 -c 'import certifi;print(certifi.where())') python3 kgss_exaone_run.py --k ${K:-10} --workers 150"
