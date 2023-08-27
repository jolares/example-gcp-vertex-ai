# To deploy a model 456 to an endpoint 123 under project example in region us-central1, run:
# Refer to https://cloud.google.com/sdk/gcloud/reference/ai/endpoints/deploy-model

## Environment Variables
# MODEL_ID
# MODEL_DISPLAY_NAME
# GCP_PROJECT_ID
# GCP_REGION
# GCP_AIP_ENDPOINT_ID

gcloud ai endpoints deploy-model $GCP_AIP_ENDPOINT_ID \
    --project=$GCP_PROJECT_ID \
    --region=$GCP_REGION \
    --model=$MODEL_ID \
    --display-name=$MODEL_DISPLAY_NAME