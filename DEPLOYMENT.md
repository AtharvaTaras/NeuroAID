# Cloud Run Deployment Guide

This guide explains how to deploy the NeuroAID Streamlit app to Google Cloud Run.

## Prerequisites

1. Google Cloud Platform account with billing enabled
2. Google Cloud SDK (gcloud) installed and configured
3. Docker installed (for local testing)
4. Project ID in GCP

## Environment Variables

Before deploying, you need to set the following environment variables in Cloud Run.

**For local development**, copy `env.example` to `.env` and fill in your values:
```bash
cp env.example .env
# Then edit .env with your actual credentials
```

The required environment variables are:

- `USERNAME` - Login username
- `PASSWORD` - Login password  
- `NAME` - User's name
- `AWS_ACCESS_KEY_ID` - AWS S3 access key (if using S3 storage)
- `AWS_SECRET_ACCESS_KEY` - AWS S3 secret key (if using S3 storage)

## Deployment Steps

### 1. Build and Test Locally (Optional)

```bash
# Build the Docker image
docker build -t neuroaid:latest .

# Run locally to test
docker run -p 8501:8501 \
  -e USERNAME=your_username \
  -e PASSWORD=your_password \
  -e NAME=YourName \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  neuroaid:latest
```

### 2. Build and Push to Google Container Registry

```bash
# Set your project ID
export PROJECT_ID=your-gcp-project-id
export SERVICE_NAME=neuroaid

# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker

# Build and push the image
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME:latest
```

### 3. Deploy to Cloud Run

```bash
# Deploy the service
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --set-env-vars USERNAME=your_username \
  --set-env-vars PASSWORD=your_password \
  --set-env-vars NAME=YourName \
  --set-env-vars AWS_ACCESS_KEY_ID=your_key \
  --set-env-vars AWS_SECRET_ACCESS_KEY=your_secret
```

### 4. Alternative: Deploy with Environment Variables from Secret Manager (Recommended)

For better security, store sensitive values in Secret Manager:

```bash
# Create secrets
echo -n "your_username" | gcloud secrets create username --data-file=-
echo -n "your_password" | gcloud secrets create password --data-file=-
echo -n "YourName" | gcloud secrets create name --data-file=-
echo -n "your_aws_key" | gcloud secrets create aws-access-key-id --data-file=-
echo -n "your_aws_secret" | gcloud secrets create aws-secret-access-key --data-file=-

# Deploy with secrets
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --update-secrets USERNAME=username:latest,PASSWORD=password:latest,NAME=name:latest,AWS_ACCESS_KEY_ID=aws-access-key-id:latest,AWS_SECRET_ACCESS_KEY=aws-secret-access-key:latest
```

## Resource Recommendations

- **Memory**: 2Gi (minimum) - Models and image processing require significant memory
- **CPU**: 2 vCPU (minimum) - For faster model inference
- **Timeout**: 3600 seconds - Model downloads and inference can take time
- **Max Instances**: Adjust based on expected traffic

## Updating the Deployment

To update the app after making changes:

```bash
# Rebuild and push
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

# Update the service
gcloud run services update $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --region us-central1
```

## Accessing the App

After deployment, Cloud Run will provide a URL like:
```
https://neuroaid-xxxxx-uc.a.run.app
```

## Troubleshooting

1. **Port binding issues**: Ensure the startup script correctly uses the PORT environment variable
2. **Memory errors**: Increase memory allocation if models fail to load
3. **Timeout errors**: Increase timeout for initial model downloads
4. **Model download failures**: Check network connectivity and Google Drive permissions

## Notes

- Models are downloaded from Google Drive at runtime on first use
- The app uses `@st.cache_resource` to cache models in memory
- Consider pre-downloading models to reduce cold start time
- Cloud Run automatically scales to zero when not in use

