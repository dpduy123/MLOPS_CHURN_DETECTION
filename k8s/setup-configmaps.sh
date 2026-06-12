#!/bin/bash
# Script để tạo các K8s ConfigMaps từ các file cấu hình nằm trong thư mục deployments/monitoring
# Bạn nên chạy script này trước khi apply các file yaml (kubectl apply -f k8s/)

# Di chuyển vị trí thực thi lệnh về thư mục gốc của project (nơi chứa file README.md)
cd "$(dirname "$0")/.."

echo "Tạo ConfigMaps cho Prometheus..."
kubectl create configmap prometheus-config --from-file=deployments/monitoring/prometheus/prometheus.yml -o yaml --dry-run=client | kubectl apply -f -
kubectl create configmap prometheus-rules --from-file=deployments/monitoring/prometheus/rules -o yaml --dry-run=client | kubectl apply -f -

echo "Tạo ConfigMaps cho Loki..."
kubectl create configmap loki-config --from-file=deployments/monitoring/loki/loki-config.yaml -o yaml --dry-run=client | kubectl apply -f -

echo "Tạo ConfigMaps cho Promtail..."
kubectl create configmap promtail-config --from-file=deployments/monitoring/promtail/promtail-config.yaml -o yaml --dry-run=client | kubectl apply -f -

echo "Tạo ConfigMaps cho Grafana..."
kubectl create configmap grafana-provisioning --from-file=deployments/monitoring/grafana/provisioning -o yaml --dry-run=client | kubectl apply -f -
kubectl create configmap grafana-dashboards --from-file=deployments/monitoring/grafana/dashboards -o yaml --dry-run=client | kubectl apply -f -

echo "Tạo ConfigMaps thành công!"
