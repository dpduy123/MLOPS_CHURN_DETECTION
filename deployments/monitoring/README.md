Monitoring stack (Prometheus, Grafana, Loki, Promtail)

Run the stack from the `deployments` directory:

```bash
cd deployments
docker compose up -d
```

Environment variables for SMTP (used by Grafana):

- `SMTP_HOST` - SMTP server host
- `SMTP_PORT` - SMTP server port
- `SMTP_USER` - SMTP username
- `SMTP_PASSWORD` - SMTP password
- `SMTP_FROM` - From address for alerts
- `ALERT_EMAILS` - Comma-separated recipient emails

Verification:

- Prometheus UI: http://localhost:9090/targets
- Grafana UI: http://localhost:3001/ (admin/admin)
- Churn service metrics: http://localhost:3000/metrics
