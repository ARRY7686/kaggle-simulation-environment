#!/usr/bin/env bash

ENV_URL=$1

echo "🔍 Checking HF Space..."
curl -s -X POST "$ENV_URL/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_churn"}' > /dev/null

if [ $? -eq 0 ]; then
  echo "✅ HF Space reachable"
else
  echo "❌ HF Space failed"
  exit 1
fi

echo "🐳 Checking Docker build..."
docker build -t test-env . > /dev/null

if [ $? -eq 0 ]; then
  echo "✅ Docker build passed"
else
  echo "❌ Docker build failed"
  exit 1
fi

echo "🧠 Checking OpenEnv spec..."
openenv validate

if [ $? -eq 0 ]; then
  echo "✅ OpenEnv validation passed"
else
  echo "❌ OpenEnv validation failed"
  exit 1
fi

echo "🎉 ALL CHECKS PASSED"