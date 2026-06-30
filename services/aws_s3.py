"""
AWS S3 Integration for Video and Report Storage
Handles upload, download, and signed URL generation
"""

import os
import boto3
from botocore.exceptions import ClientError
from typing import Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# AWS Configuration from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")  # Single bucket for videos and reports

# Initialize S3 client (only if credentials are provided)
s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    logger.info(f"S3 client initialized for region: {AWS_REGION}")
else:
    logger.warning("AWS credentials not found. S3 storage disabled. Files will be stored locally.")


class S3Manager:
    """Manages S3 uploads, downloads, and signed URLs"""
    
    @staticmethod
    def is_enabled() -> bool:
        """Check if S3 storage is enabled"""
        return s3_client is not None and AWS_S3_BUCKET is not None
    
    @staticmethod
    def upload_video(local_path: str, s3_key: Optional[str] = None) -> Optional[str]:
        """
        Upload a video file to S3
        
        Args:
            local_path: Local file path
            s3_key: S3 key (path in bucket). If None, uses filename
            
        Returns:
            S3 key on success, None on failure
        """
        if not S3Manager.is_enabled():
            logger.warning("S3 not enabled. File remains local.")
            return None
        
        try:
            if s3_key is None:
                s3_key = f"videos/{datetime.now().strftime('%Y/%m/%d')}/{os.path.basename(local_path)}"
            
            s3_client.upload_file(
                local_path,
                AWS_S3_BUCKET,
                s3_key,
                ExtraArgs={
                    'ContentType': 'video/mp4',
                    'ServerSideEncryption': 'AES256'  # Encrypt at rest
                }
            )
            logger.info(f"Uploaded video to s3://{AWS_S3_BUCKET}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Failed to upload video to S3: {e}")
            return None
    
    @staticmethod
    def upload_report(local_path: str, s3_key: Optional[str] = None) -> Optional[str]:
        """
        Upload a PDF report to S3
        
        Args:
            local_path: Local file path
            s3_key: S3 key (path in bucket). If None, uses filename
            
        Returns:
            S3 key on success, None on failure
        """
        if not S3Manager.is_enabled():
            logger.warning("S3 not enabled. File remains local.")
            return None
        
        try:
            if s3_key is None:
                s3_key = f"reports/{datetime.now().strftime('%Y/%m/%d')}/{os.path.basename(local_path)}"
            
            s3_client.upload_file(
                local_path,
                AWS_S3_BUCKET,
                s3_key,
                ExtraArgs={
                    'ContentType': 'application/pdf',
                    'ServerSideEncryption': 'AES256'
                }
            )
            logger.info(f"Uploaded report to s3://{AWS_S3_BUCKET}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Failed to upload report to S3: {e}")
            return None
    
    @staticmethod
    def upload_image(local_path: str, s3_key: Optional[str] = None) -> Optional[str]:
        """
        Upload a snapshot image to S3
        
        Args:
            local_path: Local file path
            s3_key: S3 key (path in bucket). If None, uses filename
            
        Returns:
            S3 key on success, None on failure
        """
        if not S3Manager.is_enabled():
            return None
        
        try:
            if s3_key is None:
                s3_key = f"snapshots/{datetime.now().strftime('%Y/%m/%d')}/{os.path.basename(local_path)}"
            
            s3_client.upload_file(
                local_path,
                AWS_S3_BUCKET,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/jpeg',
                    'ServerSideEncryption': 'AES256'
                }
            )
            logger.info(f"Uploaded image to s3://{AWS_S3_BUCKET}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Failed to upload image to S3: {e}")
            return None
    
    @staticmethod
    def generate_presigned_url(bucket: str, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for temporary access to S3 object
        
        Args:
            bucket: S3 bucket name
            s3_key: S3 key (path in bucket)
            expiration: URL expiration time in seconds (default 1 hour)
            
        Returns:
            Presigned URL on success, None on failure
        """
        if not S3Manager.is_enabled():
            return None
        
        try:
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    @staticmethod
    def get_video_url(s3_key: str, expiration: int = 3600) -> Optional[str]:
        """Get presigned URL for a video"""
        if not AWS_S3_BUCKET:
            return None
        return S3Manager.generate_presigned_url(AWS_S3_BUCKET, s3_key, expiration)
    
    @staticmethod
    def get_report_url(s3_key: str, expiration: int = 3600) -> Optional[str]:
        """Get presigned URL for a report"""
        if not AWS_S3_BUCKET:
            return None
        return S3Manager.generate_presigned_url(AWS_S3_BUCKET, s3_key, expiration)
    
    @staticmethod
    def download_file(bucket: str, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3
        
        Args:
            bucket: S3 bucket name
            s3_key: S3 key (path in bucket)
            local_path: Local destination path
            
        Returns:
            True on success, False on failure
        """
        if not S3Manager.is_enabled():
            return False
        
        try:
            s3_client.download_file(bucket, s3_key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            return False
    
    @staticmethod
    def delete_file(bucket: str, s3_key: str) -> bool:
        """
        Delete a file from S3
        
        Args:
            bucket: S3 bucket name
            s3_key: S3 key (path in bucket)
            
        Returns:
            True on success, False on failure
        """
        if not S3Manager.is_enabled():
            return False
        
        try:
            s3_client.delete_object(Bucket=bucket, Key=s3_key)
            logger.info(f"Deleted s3://{bucket}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete from S3: {e}")
            return False
    
    @staticmethod
    def list_files(bucket: str, prefix: str = "") -> list:
        """
        List files in S3 bucket with optional prefix
        
        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix to filter results
            
        Returns:
            List of S3 keys
        """
        if not S3Manager.is_enabled():
            return []
        
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' not in response:
                return []
            return [obj['Key'] for obj in response['Contents']]
        except ClientError as e:
            logger.error(f"Failed to list S3 files: {e}")
            return []


# Convenience functions for common operations
def upload_analysis_files(
    video_path: str,
    report_path: Optional[str] = None,
    snapshot_path: Optional[str] = None
) -> dict:
    """
    Upload all analysis output files to S3
    
    Returns:
        Dictionary with S3 keys and URLs
    """
    result = {
        'video_s3_key': None,
        'report_s3_key': None,
        'snapshot_s3_key': None,
        'video_url': None,
        'report_url': None,
        'snapshot_url': None
    }
    
    # Upload video
    if video_path and os.path.exists(video_path):
        video_key = S3Manager.upload_video(video_path)
        if video_key:
            result['video_s3_key'] = video_key
            result['video_url'] = S3Manager.get_video_url(video_key, expiration=86400)  # 24 hours
    
    # Upload report
    if report_path and os.path.exists(report_path):
        report_key = S3Manager.upload_report(report_path)
        if report_key:
            result['report_s3_key'] = report_key
            result['report_url'] = S3Manager.get_report_url(report_key, expiration=86400)
    
    # Upload snapshot
    if snapshot_path and os.path.exists(snapshot_path):
        snapshot_key = S3Manager.upload_image(snapshot_path)
        if snapshot_key:
            result['snapshot_s3_key'] = snapshot_key
            result['snapshot_url'] = S3Manager.get_video_url(snapshot_key, expiration=86400)
    
    return result
