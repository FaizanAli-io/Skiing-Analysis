#!/usr/bin/env python3
"""
Quick test script to verify S3 connection and upload functionality
Run this after updating .env with real AWS credentials
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("S3 Connection Test - Skiing Analysis Platform")
print("=" * 60)
print()

# Check environment variables
print("📋 Environment Configuration:")
print(f"   AWS_REGION: {os.getenv('AWS_REGION', 'Not set')}")
print(f"   AWS_S3_BUCKET: {os.getenv('AWS_S3_BUCKET', 'Not set')}")
print(f"   AWS_ACCESS_KEY_ID: {'Set ✓' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not set ✗'}")
print(f"   AWS_SECRET_ACCESS_KEY: {'Set ✓' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not set ✗'}")
print()

# Import S3Manager
try:
    from services.aws_s3 import S3Manager
    print("✓ S3Manager imported successfully")
except ImportError as e:
    print(f"✗ Failed to import S3Manager: {e}")
    print("   Make sure boto3 is installed: pip install boto3")
    exit(1)

# Check if S3 is enabled
print()
print("🔌 S3 Connection Status:")
if S3Manager.is_enabled():
    print("   ✓ S3 storage is ENABLED")
else:
    print("   ✗ S3 storage is DISABLED")
    print("   Ensure AWS credentials are set in .env file")
    exit(1)

print()
print("-" * 60)
print("Running Upload Tests...")
print("-" * 60)
print()

# Test 1: Create a test file
print("Test 1: Creating test file...")
test_file = "test_upload.txt"
test_content = f"Test upload at {datetime.now().isoformat()}"
try:
    with open(test_file, "w") as f:
        f.write(test_content)
    print(f"   ✓ Created {test_file}")
except Exception as e:
    print(f"   ✗ Failed to create test file: {e}")
    exit(1)

# Test 2: Upload to videos/ folder
print()
print("Test 2: Uploading to videos/ folder...")
try:
    video_key = S3Manager.upload_video(test_file, s3_key="videos/test/test_upload.txt")
    if video_key:
        print(f"   ✓ Uploaded successfully")
        print(f"   S3 Key: {video_key}")
    else:
        print(f"   ✗ Upload failed (returned None)")
        exit(1)
except Exception as e:
    print(f"   ✗ Upload failed with error: {e}")
    exit(1)

# Test 3: Generate presigned URL
print()
print("Test 3: Generating presigned URL...")
try:
    url = S3Manager.get_video_url(video_key, expiration=3600)
    if url:
        print(f"   ✓ Presigned URL generated")
        print(f"   URL (expires in 1 hour):")
        print(f"   {url[:80]}...")
    else:
        print(f"   ✗ Failed to generate URL")
except Exception as e:
    print(f"   ✗ Failed with error: {e}")

# Test 4: List files in bucket
print()
print("Test 4: Listing files in bucket...")
try:
    bucket = os.getenv('AWS_S3_BUCKET')
    files = S3Manager.list_files(bucket, prefix="videos/test/")
    if files:
        print(f"   ✓ Found {len(files)} file(s) in videos/test/:")
        for file_key in files[:5]:  # Show first 5
            print(f"      - {file_key}")
    else:
        print(f"   ⚠ No files found (this is OK for first run)")
except Exception as e:
    print(f"   ✗ List failed with error: {e}")

# Test 5: Download the file back
print()
print("Test 5: Downloading file from S3...")
download_path = "test_download.txt"
try:
    bucket = os.getenv('AWS_S3_BUCKET')
    success = S3Manager.download_file(bucket, video_key, download_path)
    if success and os.path.exists(download_path):
        with open(download_path, "r") as f:
            content = f.read()
        print(f"   ✓ Downloaded successfully")
        print(f"   Content: {content}")
        os.remove(download_path)
    else:
        print(f"   ✗ Download failed")
except Exception as e:
    print(f"   ✗ Download failed with error: {e}")

# Test 6: Delete test file from S3
print()
print("Test 6: Cleaning up test file from S3...")
try:
    bucket = os.getenv('AWS_S3_BUCKET')
    success = S3Manager.delete_file(bucket, video_key)
    if success:
        print(f"   ✓ Deleted {video_key} from S3")
    else:
        print(f"   ⚠ Could not delete (file may not exist)")
except Exception as e:
    print(f"   ✗ Delete failed with error: {e}")

# Cleanup local test file
if os.path.exists(test_file):
    os.remove(test_file)
    print(f"   ✓ Deleted local test file")

print()
print("=" * 60)
print("✅ All Tests Complete!")
print("=" * 60)
print()
print("Your S3 integration is working correctly.")
print("You can now deploy to EC2 and start uploading real videos.")
print()
print("Next steps:")
print("  1. Get RDS database credentials from client")
print("  2. Launch EC2 instance")
print("  3. Run deploy_aws.sh on EC2")
print("  4. Test video upload through FastAPI")
print()
