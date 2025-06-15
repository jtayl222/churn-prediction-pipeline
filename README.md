# 📈 Churn Prediction Pipeline

<p align="center">
  <img src="https://raw.githubusercontent.com/jtayl222/assets/main/logos/mlops_churn.svg" width="350" alt="Churn Pipeline"/>
</p>

> **TL;DR** – This repo shows how to turn a single CSV file into a production‑ready, continuously‑deployed ML service.  It is the *starting point* for the fully automated Argo workflow in [`churn-prediction-pipeline-ArgoWF`](https://github.com/jtayl222/churn-prediction-pipeline-ArgoWF) and plugs directly into the K3s cluster defined in [`k3s-homelab`](https://github.com/jtayl222/k3s-homelab).

> **⚠️ Work in Progress** – This is an active demonstration project showcasing MLOps patterns. While functional, it's continuously evolving to incorporate production-grade features. See [Development Roadmap](#development-roadmap) for planned improvements.

---

## 1. Why this project matters

Recruiters look for **evidence** that you can ship ML to production, not just train notebooks.  This repo demonstrates:

* **Reproducible data & code** – deterministic preprocessing, version‑pinned dependencies, and optional DVC storage so every experiment can be replayed.
* **CI/CD for models** – GitHub Actions tests the pipeline and pushes an OCI image to GHCR on every PR.
* **Cloud + On‑Prem parity** – the exact training script runs unmodified in AWS SageMaker **or** on a local K3s cluster.
* **Observability & governance** – built‑in MLflow tracking and Prometheus metrics exported by Seldon Core (see Next Steps).

For the big picture, read my Medium story:
➡️ [*From DevOps to MLOps: Why Employers Care and How I Built a Fortune 500 Stack in My Spare Bedroom*](https://jeftaylo.medium.com/from-devops-to-mlops-why-employers-care-and-how-i-built-a-fortune-500-stack-in-my-spare-bedroom-ce0d06dd3c61)

---

## 2. High‑level architecture

```mermaid
flowchart LR
    subgraph AWS[SageMaker]
        S3[(Telco Churn CSV)] --> PreProcess
        PreProcess((Processing Job)) --> Train
        Train((XGBoost Training)) --> ModelArtifacts>model.tar.gz]
    end
    ModelArtifacts --> |"build & push"| Kaniko{{Kaniko}}
    Kaniko --> |"registries"| ECR[(ECR / GHCR)]
    ECR --> |"deploy"| Seldon>>Seldon Core<<
    Seldon --> |"/predict"| Users[[Call API]]
    Seldon --> |"metrics"| Prometheus
    Prometheus --> Grafana
```

*Need a Kubernetes‑native workflow?* Jump to the Argo version ➡️ [`churn-prediction-pipeline-ArgoWF`](https://github.com/jtayl222/churn-prediction-pipeline-ArgoWF).

---

## 3. Quick Start

### 3.1 Local (no AWS account required)

```bash
# 1 — Clone
$ git clone https://github.com/jtayl222/churn-prediction-pipeline.git && cd churn-prediction-pipeline

# 2 — Create env
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# 3 — Run minimal pipeline
$ python minimal_churn_pipeline.py  
# → outputs metrics to ./artifacts and MLflow (if MLFLOW_TRACKING_URI is set)
```

### 3.2 AWS SageMaker

```bash
# upload data
aws s3 cp data/WA_Fn-UseC_-Telco-Customer-Churn.csv s3://$S3_BUCKET/telco.csv

# kick off pipeline
python churn_prediction_pipeline.py \
  --s3-bucket $S3_BUCKET \
  --role-arn  arn:aws:iam::$ACCOUNT:role/ChurnPredictionEC2Role
```

> **Cost tip:** Processing + single‑instance training ≈ **\$0.70/run** on spot instances.

---

## 4. Repository layout

```text
.
├── data/                  # raw dataset (small sample committed; full set fetched at runtime)
├── scripts/               # helper utilities (upload_to_s3, evaluation)
├── iac/                   # Terraform for IAM + S3 boilerplate
├── churn_prediction_pipeline.py   # SageMaker pipeline definition
├── minimal_churn_pipeline.py      # pure‑python reference
└── requirements.txt
```

---

## 5. Next Steps & Development Roadmap

### 5.1 Immediate Extensions
| Goal                         | Repository                                                                                                                                                |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🐳 **Containerize model**    | [`churn-prediction-pipeline-ArgoWF`](https://github.com/jtayl222/churn-prediction-pipeline-ArgoWF) uses **Kaniko** to build OCI images inside the cluster |
| ☁️ **Deploy on K3s homelab** | [`k3s-homelab`](https://github.com/jtayl222/k3s-homelab) provisions a 5‑node cluster with MetalLB & Longhorn                                              |
| 🎛 **End‑to‑end demo**       | [`homelab-mlops-demo`](https://github.com/jtayl222/homelab-mlops-demo) walks through training ➜ serving ➜ monitoring                                      |
| 📊 **Dashboards & alerts**   | Grafana JSON & Alertmanager rules in `homelab-mlops-demo/grafana/`                                                                                        |

### 5.2 Production Readiness (Planned)
#### 🔒 **Security & Compliance**
- [ ] IAM roles with least-privilege principles
- [ ] Secrets management with AWS Secrets Manager / K8s secrets
- [ ] Network policies and VPC security groups
- [ ] Container image vulnerability scanning
- [ ] Data encryption at rest and in transit

#### 🧪 **Testing & Quality Assurance**
- [ ] **Unit Tests**: `pytest` suite for preprocessing and evaluation logic
- [ ] **Integration Tests**: End-to-end pipeline validation with test data
- [ ] **Model Tests**: Bias detection, performance regression, data drift tests
- [ ] **Infrastructure Tests**: Terraform validation, K8s resource health checks
- [ ] **Load Testing**: Performance benchmarks for inference endpoints

#### 📊 **Observability & Monitoring**
- [ ] **SLIs/SLOs**: Define service level indicators (latency, accuracy, uptime)
- [ ] **Alerting Runbooks**: Automated incident response procedures
- [ ] **Distributed Tracing**: Request flow tracking across microservices
- [ ] **Log Aggregation**: Centralized logging with structured formats
- [ ] **Capacity Planning**: Auto-scaling based on traffic patterns

#### 🏗️ **Performance & Scale**
- [ ] **Throughput Optimization**: Handle 10K+ requests/second
- [ ] **Cost Management**: Spot instances, resource right-sizing, auto-scaling
- [ ] **Data Pipeline Scale**: Process TB-scale datasets efficiently
- [ ] **Streaming Integration**: Real-time inference with Kafka/Kinesis
- [ ] **Multi-region Deployment**: Geographic load distribution

#### 🔄 **Data & Model Governance**
- [ ] **Schema Evolution**: Backward-compatible data format changes
- [ ] **Data Lineage**: Track data provenance and transformations
- [ ] **Model Versioning**: A/B testing and gradual rollout strategies
- [ ] **Compliance Auditing**: GDPR, SOX, regulatory reporting
- [ ] **Automated Retraining**: Drift detection and model refresh triggers

#### 🚨 **Incident Response**
- [ ] **Rollback Procedures**: Automated model version reversion
- [ ] **Debugging Guides**: Systematic troubleshooting workflows
- [ ] **Disaster Recovery**: Cross-region backup and failover
- [ ] **Change Management**: Approval workflows for production changes
- [ ] **Post-mortem Templates**: Structured incident analysis

---

## 6. Contributing

Pull requests are welcome!  See [CONTRIBUTING.md](CONTRIBUTING.md) *(coming soon)* for coding standards and how to run the pre‑commit hooks.

**Current Development Focus:**
- Improving test coverage for preprocessing logic
- Adding model drift detection capabilities
- Implementing automated security scanning
- Documentation for production deployment patterns

---

## 7. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 8. Acknowledgments

This project is part of a broader MLOps learning initiative. While designed for demonstration purposes, it incorporates real-world patterns used in production environments. Feedback and contributions help improve its educational value
