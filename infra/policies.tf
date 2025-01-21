# Create policy for titanic project user

resource "aws_iam_policy" "user_policy" {
  name        = "policy-for-titanic-user"
  description = "This policy gives permission to create IAM roles and S3 buckets."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "iam:CreateRole",
          "iam:DeleteRole",
          "iam:AttachRolePolicy",
          "iam:DetachRolePolicy",
          "iam:GetRole",
          "iam:PassRole",
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "ec2:*",
          "ssm:StartSession",
          "ssm:TerminateSession"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ],
        Resource = [
          "arn:aws:s3:::my-titanic-project-bucket",
          "arn:aws:s3:::my-titanic-project-bucket/*"
        ]
      }
      
    ]
  })
}

resource "aws_iam_user_policy_attachment" "attach_policies_to_user" {
  user       = "titanic-project-user"
  policy_arn = aws_iam_policy.user_policy.arn
}