import boto3
import os
from dotenv import load_dotenv
from botocore.config import Config

load_dotenv(".env")

bucket = os.getenv("THUNDER_R2_BUCKET")
endpoint = os.getenv("THUNDER_R2_ENDPOINT")
access_key = os.getenv("THUNDER_R2_ACCESS_KEY")
secret_key = os.getenv("THUNDER_R2_SECRET_KEY")

# Remove bucket name from endpoint if present
if endpoint and bucket and endpoint.endswith(bucket):
    endpoint = endpoint.replace("/" + bucket, "")
elif endpoint and bucket and endpoint.endswith(bucket + "/"):
    endpoint = endpoint.replace("/" + bucket + "/", "")

print(f"Bucket: {bucket}")
print(f"Endpoint: {endpoint}")

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name="auto",
    config=Config(signature_version="s3v4")
)

try:
    print("Listing buckets...")
    res = s3.list_buckets()
    print("Buckets found:", [b['Name'] for b in res['Buckets']])
    
    print(f"Listing contents of {bucket}...")
    res = s3.list_objects_v2(Bucket=bucket, MaxKeys=10)
    if 'Contents' in res:
        for obj in res['Contents']:
            print(f" - {obj['Key']}")
    else:
        print("Bucket is empty or not found.")
except Exception as e:
    print(f"Error: {e}")
