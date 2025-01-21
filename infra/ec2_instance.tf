# Create an instance profile and attach to iam role

resource "aws_iam_instance_profile" "ec2_instance_profile" {
  name = "ec2_instance_Profile"
  role = aws_iam_role.ec2_role.name
}


resource "aws_instance" "ec2_instance" {
  ami                     = var.ami
  instance_type           = "t3.micro"
  security_groups         = [aws_security_group.allow_http_api_access.name]
  iam_instance_profile = aws_iam_instance_profile.ec2_instance_profile.name

  tags = {
    Name = "titanic_instance"
  }

}