## Linebot on Cloud Run

### How to deploy

```bash
$ gcloud config set app/cloud_build_timeout 3600
$ gcloud run deploy cloud-run-seq2seq-linebot --source . --region us-west1
```
