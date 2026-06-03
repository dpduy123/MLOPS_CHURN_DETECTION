# Monitoring Stack Implementation Plan

## Context

Project structure:

```text
deployments/
├── docker-compose.yml                 # To create
└── monitoring/
    ├── grafana/                       # To create
    ├── loki/
    │   └── loki-config.yaml
    ├── prometheus/
    │   └── prometheus.yml
    ├── promtail/
    │   └── promtail-config.yaml
    └── evidently/                     # Ignore completely
```

Available services:

### MLflow

Already running externally:

```text
host.docker.internal:5100
```

Do NOT deploy another MLflow instance.

### Churn Prediction Service

Docker image already exists:

```text
churn-prediction:1.0.0
```

Expected runtime:

```text
Port: 3000
Metrics endpoint: /metrics
```

Prometheus metrics already exposed:

* churn_prediction_requests_total
* churn_prediction_labels_total
* churn_prediction_confidence
* churn_prediction_latency_seconds
* churn_prediction_batch_size
* churn_low_confidence_streak

The metric:

```text
churn_low_confidence_streak
```

must be monitored and used for alerting.

---

# Objectives

Build a complete monitoring stack consisting of:

* Prometheus
* Grafana
* Loki
* Promtail

Requirements:

1. Verify existing configurations.
2. Fix any incorrect configurations.
3. Create Grafana provisioning.
4. Create Grafana dashboards.
5. Create alerting when:

```text
churn_low_confidence_streak >= 200
```

6. Send email notifications to configurable recipients.
7. Create a single docker-compose file to run everything.

---

# Important Constraints

## Ignore Evidently

The folder:

```text
deployments/monitoring/evidently
```

must be ignored entirely.

Do not:

* deploy Evidently
* scrape Evidently metrics
* create Evidently dashboards
* create Evidently alerts

---

# Phase 1 — Configuration Review

Review the following files:

## Loki

```text
deployments/monitoring/loki/loki-config.yaml
```

Verify:

* storage configuration
* schema configuration
* filesystem paths
* ruler configuration
* container compatibility

Check whether:

```yaml
alertmanager_url: http://localhost:9093
```

is still needed.

If Grafana Alerting will be used, remove unnecessary Alertmanager dependencies.

---

## Prometheus

Review:

```text
deployments/monitoring/prometheus/prometheus.yml
```

Verify:

### Churn service scrape

```yaml
job_name: bentoml_churn
targets:
  - bentoml:3000
```

Ensure target matches the final docker-compose service name.

Review:

* scrape interval
* metrics path
* labels
* rule files

Remove dead configurations if necessary.

---

## Promtail

Review:

```text
deployments/monitoring/promtail/promtail-config.yaml
```

Verify:

* docker discovery
* docker socket access
* relabeling
* compose project filters

Special attention:

```yaml
com_docker_compose_project
```

must match the final compose project name.

If filtering is too restrictive, simplify it.

Verify logs will reach Loki.

---

# Phase 2 — Docker Compose

Create:

```text
deployments/docker-compose.yml
```

Services:

## churn-prediction

Image:

```yaml
image: churn-prediction:1.0.0
```

Port:

```yaml
3000:3000
```

---

## prometheus

Official Prometheus image.

Mount:

```text
monitoring/prometheus/prometheus.yml
```

and rules directory.

---

## loki

Official Loki image.

Mount:

```text
monitoring/loki/loki-config.yaml
```

Persist data.

---

## promtail

Official Promtail image.

Mount:

* docker socket
* promtail config

---

## grafana

Official Grafana image.

Persist:

* dashboards
* datasources
* alerting configuration

Expose:

```text
3001
```

if desired.

---

# Phase 3 — Grafana Provisioning

Create:

```text
deployments/monitoring/grafana/
```

Structure:

```text
grafana/
├── provisioning/
│   ├── datasources/
│   ├── dashboards/
│   └── alerting/
└── dashboards/
```

---

## Datasources

Automatically provision:

### Prometheus

URL:

```text
http://prometheus:9090
```

### Loki

URL:

```text
http://loki:3100
```

No manual UI setup should be required.

---

# Phase 4 — Dashboard Creation

Create a dashboard named:

```text
Churn Prediction Monitoring
```

Required panels:

---

## Request Volume

Metric:

```promql
sum(rate(churn_prediction_requests_total[5m]))
```

---

## Predictions by Label

Metric:

```promql
sum by(label)(
  rate(churn_prediction_labels_total[5m])
)
```

---

## Confidence Distribution

Metric:

```promql
histogram_quantile(...)
```

based on:

```text
churn_prediction_confidence
```

---

## Latency

Display:

* p50
* p95
* p99

using:

```text
churn_prediction_latency_seconds
```

---

## Batch Size

Based on:

```text
churn_prediction_batch_size
```

---

## Current Low Confidence Streak

Metric:

```promql
churn_low_confidence_streak
```

Display as Stat panel.

---

## Logs Panel

Datasource:

```text
Loki
```

Filter logs from:

```text
churn-prediction
```

Allow filtering by:

* level
* service

---

# Phase 5 — Alerting

Implement alerting for:

```promql
churn_low_confidence_streak >= 200
```

Preferred approach:

## Grafana Unified Alerting

Do NOT introduce Alertmanager unless absolutely necessary.

Create alert rule:

```promql
churn_low_confidence_streak >= 200
```

Recommended:

```text
for: 5m
```

to avoid false positives.

Labels:

```yaml
severity: critical
service: churn-prediction
```

Annotations:

* alert title
* current value
* dashboard link

---

# Phase 6 — Email Notifications

Configure Grafana SMTP.

Recipients must be configurable via environment variables.

Expected variables:

```env
SMTP_HOST=
SMTP_PORT=
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=
ALERT_EMAILS=
```

Agent should document:

* Gmail setup
* App Password setup
* Verification steps

---

# Phase 7 — Validation

After implementation verify:

## Metrics

Open:

```text
http://localhost:3000/metrics
```

Ensure metrics are exposed.

---

## Prometheus

Open:

```text
http://localhost:9090/targets
```

Verify:

```text
bentoml_churn = UP
```

---

## Grafana

Verify:

* Prometheus datasource healthy
* Loki datasource healthy
* Dashboard loads correctly

---

## Logs

Generate application logs.

Verify logs appear in Loki and Grafana.

---

## Alert Test

Temporarily lower threshold:

```promql
churn_low_confidence_streak >= 2
```

Generate alert.

Verify email delivery.

---

# Deliverables

Agent must produce:

1. Updated Loki configuration
2. Updated Prometheus configuration
3. Updated Promtail configuration
4. Complete docker-compose.yml
5. Grafana datasource provisioning
6. Grafana dashboard provisioning
7. Grafana alerting configuration
8. Alert rule for low confidence streak
9. Documentation explaining how to run and verify the stack

The final result must work with a single command:

```bash
docker compose up -d
```
