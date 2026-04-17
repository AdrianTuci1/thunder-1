import os
import sys
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config_manager import THUNDER_CONFIG
from core.storage import ObjectStorageManager

def manual_remote_cleanup(limit=5, dry_run=False):
    load_dotenv(".env")
    
    # Force enable storage for this script
    THUNDER_CONFIG["storage"]["enabled"] = True
    storage = ObjectStorageManager(THUNDER_CONFIG)
    
    if not storage.enabled:
        print("❌ R2 Storage is not correctly configured in .env or config_manager.py")
        return

    print(f"🧹 Starting {'DRY RUN ' if dry_run else ''}R2 Cleanup (Keeping last {limit} checkpoints)...")
    
    try:
        # List all objects to find all checkpoint directories
        paginator = storage.s3_client.get_paginator('list_objects_v2')
        checkpoints = set()
        
        for page in paginator.paginate(Bucket=storage.bucket_name):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if "checkpoint-" in key:
                        parts = key.split('/')
                        for i, part in enumerate(parts):
                            if part.startswith("checkpoint-"):
                                checkpoints.add("/".join(parts[:i+1]))
                                break
        
        if not checkpoints:
            print("ℹ️ No checkpoints found in R2.")
            return

        print(f"🔍 Found {len(checkpoints)} checkpoints in R2.")
        
        # Sort by step number
        sorted_checkpoints = sorted(
            list(checkpoints), 
            key=lambda x: int(x.split('-')[-1]) if '-' in x and x.split('-')[-1].isdigit() else 0
        )
        
        if len(sorted_checkpoints) <= limit:
            print(f"✅ Only {len(sorted_checkpoints)} checkpoints found. No cleanup needed.")
            return
            
        to_delete = sorted_checkpoints[:-limit]
        print(f"🗑️ Identified {len(to_delete)} checkpoints for deletion.")
        
        for folder in to_delete:
            if dry_run:
                print(f" [DRY RUN] Would delete: {folder}")
            else:
                print(f" 💣 Deleting: {folder}...")
                storage._delete_folder(folder)
                
        print(f"\n✨ Cleanup completed. {'(Nothing actually deleted)' if dry_run else ''}")
        
    except Exception as e:
        print(f"❌ Error during manual cleanup: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prune old checkpoints from Cloudflare R2")
    parser.add_argument("--limit", type=int, default=5, help="Number of checkpoints to keep (default: 5)")
    parser.add_argument("--execute", action="store_true", help="Actually delete files (default: dry run)")
    
    args = parser.parse_args()
    
    manual_remote_cleanup(limit=args.limit, dry_run=not args.execute)
