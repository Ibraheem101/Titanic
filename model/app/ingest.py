import boto3

s3 = boto3.client('s3')
bucket_name = 'my-titanic-project-bucket'

# Download model and data files
s3.download_file(bucket_name, 'data/train.csv', '../../data/train.csv')
s3.download_file(bucket_name, 'data/test.csv','../../data/test.csv')