# SMSML Juditha-Danuvanya

Repository ini berisi keseluruhan pipeline *Machine Learning Operations* (MLOps) untuk dataset Breast Cancer Wisconsin, sebagai pemenuhan Kriteria 2, 3, dan 4 kelas Membangun Sistem Machine Learning Dicoding.

## Struktur Folder
- `Membangun_model/`: Berisi *script* pelatihan model (Basic dan Advanced), serta file `.env` dan `requirements.txt`. Pelatihan model terhubung dengan DagsHub MLflow.
- `Workflow-CI/`: Berisi konfigurasi `MLProject` dan `conda.yaml` untuk dijalankan secara terisolasi.
- `.github/workflows/`: Berisi konfigurasi CI/CD untuk otomatisasi pelatihan model dan pembuatan *Docker Image* yang dikirim ke Docker Hub.
- `Monitoring dan Logging/`: Berisi konfigurasi `docker-compose.yml` untuk Prometheus dan Grafana, serta *script* `inference.py` dan `prometheus_exporter.py` untuk mensimulasikan *traffic* dan merekam metrik performa model.
- `Eksperimen_SML_Juditha-Danuvanya.txt`: Tautan ke repository Kriteria 1.
- `Workflow-CI.txt`: Tautan ke repository ini (Kriteria 3).
