# Workflow-CI

Folder ini berisi konfigurasi CI untuk re-training model menggunakan MLflow Project dan build/push Docker image.

## Struktur
- `.github/workflows/ci.yml` : workflow GitHub Action untuk menjalankan MLflow Project dan membangun Docker image.
- `.workflow/ci.yml` : salinan struktur workflow sesuai kriteria advance.
- `MLProject/` : folder MLflow Project berisi `modelling.py`, `conda.yaml`, `MLproject`, dataset, dan model artefak.

## Cara kerja
- Ketika ada push ke branch `main`, workflow akan:
  1. checkout kode
  2. install dependencies
  3. jalankan `mlflow run ./MLProject --env-manager local`
  4. build image Docker dengan `mlflow models build-docker`
  5. push image ke Docker Hub
