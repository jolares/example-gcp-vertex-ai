# !pip3 install google-cloud-aiplatform --upgrade --user
import os
from google.cloud import aiplatform

GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
GCP_REGION=os.environ.get('GCP_REGION')
GCP_AIP_ENDPOINT_ID=os.environ.get('GCP_AIP_ENDPOINT_ID')

endpoint = aiplatform.Endpoint(
    endpoint_name=f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/endpoints/{GCP_AIP_ENDPOINT_ID}"
)

test_mpg = [
    1.4838871833555929,
    1.8659883497083019,
    2.234620276849616,
    1.0187816540094903,
    -2.530890710602246,
    -1.6046416850441676,
    -0.4651483719733302,
    -0.4952254087173721,
    0.7746763768735953
]

response = endpoint.predict([test_mpg])

print('API response: ', response)

print('Predicted MPG: ', response.predictions[0][0])