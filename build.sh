# - Builds a trainer container
# - Runs trainer container locally to ensure its working correctly.
# - Uploads/Publishes trainer container to GCP Cloud Storage bucket

## Env variables
# GCP_PROJECT_ID
# GCP_REGION
# GCP_CS_TRAIN_OUTPUT_BUCKET_URI


# TODO: conditionally create bucket if it does not exist already
# Creates a new bucket in GCP Cloud Storage
gsutil mb -l ${GCP_REGION} ${GCP_CS_TRAIN_OUTPUT_BUCKET_URI}

# Defines the URI of the container image to build
IMAGE_URI="gcr.io/$GCP_PROJECT_ID/mpg:v1"

# Builds the container
docker build ./ -t $

# Runs the container within notebook instance to ensure it's working correctly.
docker run $IMAGE_URI

# Pushes the container to Google Container Registry:
docker push $IMAGE_URI