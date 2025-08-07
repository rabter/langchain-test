#!/bin/bash

# 기존에 실행 중인 uvicorn 프로세스 종료
pids=$(ps aux | grep '[u]vicorn main:app' | awk '{print $2}')

if [ -n "$pids" ]; then
  echo "Stopping existing uvicorn processes: $pids"
  kill -9 $pids
fi

# uvicorn 서버 백그라운드로 실행, 로그는 app.log에 저장
nohup uvicorn main:app --host 0.0.0.0 --port 8888 > app.log 2>&1 &

echo "Uvicorn server started with PID $!"
