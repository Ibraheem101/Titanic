# Define the IAM Role for EC2
resource "aws_iam_role" "ec2_role" {
  name = "ec2_role"

  # Terraform's "jsonencode" function converts a
  # Terraform expression result to valid JSON syntax.
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      },
    ]
  })

  tags = {
    tag-key = "tag-value"
  }
}

# Create S3 Bucket
resource "aws_s3_bucket" "project_bucket" {
  bucket = "my-titanic-project-bucket"
  
  tags = {
    Name        = var.bucket_name
    Environment = "Dev"
  }
}

