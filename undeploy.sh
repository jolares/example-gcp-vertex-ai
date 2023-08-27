# To undeploy a model 456 from an endpoint 123 under project example in region us-central1, run:
# Refer to https://cloud.google.com/sdk/gcloud/reference/ai/endpoints/undeploy-model

## Environment Variables
# MODEL_ID
# GCP_PROJECT_ID
# GCP_REGION
# DEPLOYED_MODEL_ID

gcloud ai endpoints undeploy-model $MODEL_ID \
    --project=$GCP_PROJECT_ID \
    --region=$GCP_REGION \
    --deployed-model-id=$DEPLOYED_MODEL_ID