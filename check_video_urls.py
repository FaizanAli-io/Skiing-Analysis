#!/usr/bin/env python3
"""
Quick script to check what URLs are saved in the database
"""

from database import SessionLocal
from models.video_analysis import VideoAnalysis

db = SessionLocal()

print("=" * 80)
print("VIDEO ANALYSIS RECORDS - URL Check")
print("=" * 80)
print()

attempts = db.query(VideoAnalysis).order_by(VideoAnalysis.id.desc()).limit(5).all()

if not attempts:
    print("❌ No attempts found in database!")
else:
    for i, attempt in enumerate(attempts, 1):
        print(f"Attempt #{i} (ID: {attempt.id})")
        print(f"  Attempt Number: {attempt.attempt_number}")
        print(f"  Video Name: {attempt.video_name}")
        print(f"  Video Link: {attempt.video_link}")
        print(f"  Report Path: {attempt.report_path}")
        print(f"  Output Video Path: {attempt.output_video_path}")
        print(f"  S3 Video Key: {attempt.s3_video_key}")
        print(f"  S3 Report Key: {attempt.s3_report_key}")
        print()
        
        # Analysis
        if attempt.video_link:
            if attempt.video_link.startswith("http"):
                print(f"  ✅ Video Link is S3 URL")
            else:
                print(f"  ⚠️  Video Link is local path: {attempt.video_link}")
        else:
            print(f"  ❌ Video Link is NULL/Empty!")
        
        if attempt.s3_video_key:
            print(f"  ✅ S3 Video Key exists")
        else:
            print(f"  ⚠️  S3 Video Key is NULL")
        
        print("-" * 80)
        print()

db.close()

print()
print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print()
print("If Video Link starts with 'http' → Frontend should display S3 video")
print("If Video Link is local path → Frontend looks for local file")
print("If Video Link is NULL → Frontend shows nothing")
print()
print("For S3 to work:")
print("  1. video_link should be S3 presigned URL (https://...)")
print("  2. s3_video_key should be S3 key (videos/2026/07/01/...)")
print()
