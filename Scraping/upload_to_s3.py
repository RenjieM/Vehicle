import boto3
import logging
import os

""" Set up S3 access """

""" Upload to intake intake_from_scraping """
s3 = boto3.resource("s3")
def upload_to_intake(File_path, S3_path, Bucket = "intakebucket"): #Key is filename, Filename should be dir!@#$@$!
    s3 = boto3.client("s3")
    s3.upload_file(
        Filename = File_path,
        Bucket = 'intakebucket',
        Key=S3_path
    )
# Download files from S3
def download(Key, Filename, Bucket = "intakebucket"):
    s3 = boto3.client("s3")
    s3.download_file(
        Bucket = "intakebucket", Key = Key, Filename = Filename
    )