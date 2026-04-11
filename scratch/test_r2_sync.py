import sys
import os
import time
import shutil

# Mocking boto3 to avoid dependency issues during test if not installed
try:
    import boto3
except ImportError:
    print("⚠️ boto3 not found, test will mock it.")
    class MockS3Client:
        def upload_file(self, local_path, bucket, s3_path):
            print(f"[MOCK S3] Uploaded {local_path} to {bucket}/{s3_path}")
    
    import unittest.mock as mock
    sys.modules["boto3"] = mock.Mock()
    sys.modules["boto3"].client.return_value = MockS3Client()
    sys.modules["botocore"] = mock.Mock()
    sys.modules["botocore.config"] = mock.Mock()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.storage import ObjectStorageManager

def test_storage():
    print("🧪 Testing ObjectStorageManager...")
    
    test_config = {
        "storage": {
            "enabled": True,
            "bucket": "test-bucket",
            "endpoint_url": "https://test.r2.cloudflarestorage.com",
            "access_key_id": "test-key",
            "secret_access_key": "test-secret"
        }
    }
    
    # Create a dummy checkpoint directory
    ckpt_dir = "scratch/dummy_checkpoint"
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "model.pt"), "w") as f:
        f.write("dummy weights")
    with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
        f.write('{"step": 100}')
        
    storage = ObjectStorageManager(test_config)
    
    print(f"🚀 Triggering async upload for {ckpt_dir}...")
    storage.upload_checkpoint_async(ckpt_dir)
    
    # Wait for background thread
    print("⏳ Waiting for upload to complete...")
    time.sleep(2)
    
    storage.close()
    print("✅ Test finished.")
    
    # Cleanup
    shutil.rmtree(ckpt_dir)

if __name__ == "__main__":
    test_storage()
