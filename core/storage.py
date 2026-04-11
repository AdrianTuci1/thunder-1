import os
import logging
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

class ObjectStorageManager:
    """
    Manages asynchronous uploads to S3-compatible object storage (Cloudflare R2).
    Optimized for heavy checkpoint syncing without blocking the main training loop.
    """
    
    def __init__(self, config: dict):
        self.storage_config = config.get("storage", {})
        self.enabled = self.storage_config.get("enabled", False)
        
        if not self.enabled:
            return

        self.bucket_name = self.storage_config.get("bucket")
        self.endpoint_url = self.storage_config.get("endpoint_url")
        self.region = self.storage_config.get("region", "auto")
        
        # Priority: Config keys -> Environment Variables
        self.access_key = self.storage_config.get("access_key_id") or os.getenv("THUNDER_R2_ACCESS_KEY")
        self.secret_key = self.storage_config.get("secret_access_key") or os.getenv("THUNDER_R2_SECRET_KEY")

        if not all([self.access_key, self.secret_key, self.endpoint_url, self.bucket_name]):
            logger.warning("⚠️ Storage enabled but missing credentials or endpoint. Syncing disabled.")
            self.enabled = False
            return

        # Initialize S3 client for R2
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
            config=Config(signature_version="s3v4")
        )
        
        # Background executor for non-blocking uploads
        self.executor = ThreadPoolExecutor(max_workers=4)

    def upload_checkpoint_async(self, local_dir: str):
        """
        Triggers a background upload of the entire checkpoint directory.
        """
        if not self.enabled:
            return
        
        self.executor.submit(self._sync_directory, local_dir)

    def _sync_directory(self, local_dir: str):
        """
        Recursive upload of a directory to S3.
        """
        base_name = os.path.basename(local_dir.rstrip("/"))
        
        try:
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    # Create a relative path for S3 key
                    relative_path = os.path.relpath(local_path, os.path.dirname(local_dir))
                    s3_path = relative_path.replace("\\", "/") # Ensure forward slashes for S3
                    
                    logger.info(f"📤 Uploading {local_path} to s3://{self.bucket_name}/{s3_path}")
                    self.s3_client.upload_file(local_path, self.bucket_name, s3_path)
            
            print(f"✅ Checkpoint synchronized to R2: {base_name}")
        except Exception as e:
            print(f"❌ Error syncing {base_name} to R2: {str(e)}")
            logger.error(f"Failed to upload checkpoint {local_dir} to R2: {e}")

    def close(self):
        """
        Wait for all pending uploads to complete.
        """
        if self.enabled:
            self.executor.shutdown(wait=True)
