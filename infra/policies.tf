# # Create policy for titanic project user

# resource "aws_iam_policy" "policy-for-titanic-user" {
#   name        = "policy-for-titanic-user"
#   description = "This policy gives permission to create IAM roles and S3 buckets.

# "

#   policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [
#       {
#         Effect = "Allow"
#         Action = [
#           "iam:*",
#           "s3:*"
#         ]
#         Resource = "*"
#       }
#     ]
#   })
# }

# resource "aws_iam_user_policy_attachment" "attach_full_iam_s3" {
#   user       = "titanic-project-user"
#   policy_arn = aws_iam_policy.full_iam_ec2_access.arn
# }

