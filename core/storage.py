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

        self.bucket_name = self.storage_config.get("bucket") or os.getenv("THUNDER_R2_BUCKET")
        self.endpoint_url = self.storage_config.get("endpoint_url") or os.getenv("THUNDER_R2_ENDPOINT")
        self.region = self.storage_config.get("region", "auto") or os.getenv("THUNDER_R2_REGION", "auto")
        
        # Priority: Config keys -> Environment Variables
        self.access_key = self.storage_config.get("access_key_id") or os.getenv("THUNDER_R2_ACCESS_KEY")
        self.secret_key = self.storage_config.get("secret_access_key") or os.getenv("THUNDER_R2_SECRET_KEY")

        if not all([self.access_key, self.secret_key, self.endpoint_url, self.bucket_name]):
            logger.warning("⚠️ Storage enabled but missing credentials or endpoint. Syncing disabled.")
            self.enabled = False
            return

        # Fix: Strip bucket name from endpoint if it was accidentally included (common R2 mistake)
        if self.endpoint_url and self.bucket_name:
            # Remove bucket from end of URL if present
            clean_endpoint = self.endpoint_url.rstrip("/")
            if clean_endpoint.endswith(self.bucket_name):
                self.endpoint_url = clean_endpoint[:-(len(self.bucket_name))].rstrip("/")
                logger.info(f"🔧 Normalized R2 endpoint: {self.endpoint_url}")

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

    def download_checkpoint(self, checkpoint_name: str, local_dest: str):
        """
        Synchronously downloads a checkpoint folder from R2.
        """
        if not self.enabled:
            logger.error("Attempted to download from R2 while storage is disabled.")
            return False

        try:
            os.makedirs(local_dest, exist_ok=True)
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            # Ensure prefix ends with / if it's a folder
            prefix = checkpoint_name.rstrip('/') + '/'
            print(f"📥 Downloading checkpoint '{checkpoint_name}' from R2 to {local_dest}...")
            
            download_count = 0
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        # Calculate local path
                        relative_path = os.path.relpath(s3_key, prefix)
                        if relative_path == ".": continue # Skip the folder key itself if it exists
                        
                        local_file_path = os.path.join(local_dest, relative_path)
                        
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        
                        print(f"  - {relative_path}")
                        self.s3_client.download_file(self.bucket_name, s3_key, local_file_path)
                        download_count += 1
            
            if download_count == 0:
                # Try without trailing slash just in case it's a prefix rather than a folder
                prefix_no_slash = checkpoint_name.rstrip('/')
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix_no_slash):
                    if 'Contents' in page:
                         for obj in page['Contents']:
                            s3_key = obj['Key']
                            relative_path = os.path.relpath(s3_key, os.path.dirname(prefix_no_slash))
                            local_file_path = os.path.join(os.path.dirname(local_dest), relative_path)
                            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                            print(f"  - {relative_path}")
                            self.s3_client.download_file(self.bucket_name, s3_key, local_file_path)
                            download_count += 1

            if download_count == 0:
                print(f"⚠️ No files found for checkpoint '{checkpoint_name}' in R2.")
                return False
                
            print(f"✅ Download complete. {download_count} files retrieved.")
            return True
        except Exception as e:
            print(f"❌ Error downloading from R2: {str(e)}")
            return False

    def cleanup_remotely(self, keep_limit: int):
        """
        Lists all checkpoints in R2 and deletes the oldest ones to respect keep_limit.
        """
        if not self.enabled:
            return

        try:
            # List all objects to find all checkpoint directories (handles nested or mis-prefixed keys)
            paginator = self.s3_client.get_paginator('list_objects_v2')
            checkpoints = set()
            
            for page in paginator.paginate(Bucket=self.bucket_name):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if "checkpoint-" in key:
                            # Extract the checkpoint folder path (e.g. "checkpoint-1000" or "runs/checkpoint-1000")
                            parts = key.split('/')
                            for i, part in enumerate(parts):
                                if part.startswith("checkpoint-"):
                                    checkpoint_prefix = "/".join(parts[:i+1])
                                    checkpoints.add(checkpoint_prefix)
                                    break

            if len(checkpoints) <= keep_limit:
                return

            # Sort checkpoints by step number numerically
            sorted_checkpoints = sorted(
                list(checkpoints), 
                key=lambda x: int(x.split('-')[-1]) if '-' in x and x.split('-')[-1].isdigit() else 0
            )

            to_delete = sorted_checkpoints[:-keep_limit]
            print(f"🧹 R2 Pruning: Found {len(checkpoints)} checkpoints, keeping last {keep_limit}. Deleting {len(to_delete)}...")
            for folder in to_delete:
                print(f"  - Deleting {folder}")
                self._delete_folder(folder)
            
        except Exception as e:
            logger.error(f"Error during R2 cleanup: {e}")

    def get_latest_checkpoint_name(self) -> Optional[str]:
        """
        Queries R2 to find the checkpoint folder with the highest step number.
        Returns the full path/prefix of the checkpoint.
        """
        if not self.enabled:
            return None

        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            checkpoints = set()
            
            for page in paginator.paginate(Bucket=self.bucket_name):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if "checkpoint-" in key:
                            parts = key.split('/')
                            for i, part in enumerate(parts):
                                if part.startswith("checkpoint-"):
                                    checkpoint_prefix = "/".join(parts[:i+1])
                                    checkpoints.add(checkpoint_prefix)
                                    break
            
            if not checkpoints:
                return None

            # Return the one with the highest step number
            latest = max(
                list(checkpoints), 
                key=lambda x: int(x.split('-')[-1]) if '-' in x and x.split('-')[-1].isdigit() else 0
            )
            return latest
        except Exception as e:
            logger.error(f"Error finding latest checkpoint in R2: {e}")
            return None

    def _delete_folder(self, prefix: str):
        """
        Deletes all objects under a given prefix in R2.
        """
        try:
            # Paginator to handle more than 1000 objects if needed
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    delete_keys = [{'Key': obj['Key']} for obj in page['Contents']]
                    self.s3_client.delete_objects(Bucket=self.bucket_name, Delete={'Objects': delete_keys})
            print(f"🗑️ Successfully deleted {prefix} from R2.")
        except Exception as e:
            logger.error(f"Failed to delete folder {prefix} from R2: {e}")

    def close(self):
        """
        Wait for all pending uploads to complete.
        """
        if self.enabled:
            self.executor.shutdown(wait=True)
