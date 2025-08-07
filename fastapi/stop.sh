#!/bin/bash

# 실행 중인 uvicorn 프로세스 찾기 및 종료
pids=$(ps aux | grep '[u]vicorn main:app' | awk '{print $2}')

if [ -z "$pids" ]; then
  echo "No uvicorn process found."
else
  echo "Stopping uvicorn processes: $pids"
  kill -9 $pids
fi
