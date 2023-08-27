# Uploads a custom container for prediction
# Refer to https://cloud.google.com/vertex-ai/docs/predictions/use-custom-container

gcloud ai models upload \
  --region=$GCP_REGION \
  --display-name=$MODEL_DISPLAY_NAME \
  --container-image-uri=$IMAGE_URI \
  --artifact-uri=$MODEL_ARTIFACT_DIRPATH
  # Optional
  --container-command=$COMMAND \
  --container-args=$ARGS \
  --container-ports=$PORTS \
  --container-env-vars=$ENV \
  --container-health-route=$HEALTH_ROUTE \
  --container-predict-route=$PREDICT_ROUTE \
