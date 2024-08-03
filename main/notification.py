import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, NoRegionError


# How to use:
# 1. Install `awscli`
# 2. Go to AWS and setup Access Key and Secrets on IAM
# 3. Run `aws configure` on laptop with the information from previous step
# 4. Add your phone number on the AWS SNS sandbox
# 5. Run the method below with your phone number

def send_sms(phone_number, message):
    sns = boto3.client('sns', region_name='us-east-1')

    try:
        response = sns.publish(
            PhoneNumber=phone_number,
            Message=message,
        )
        return response

    except NoCredentialsError as e:
        return f"Credentials not available: {e}"
    except NoRegionError as e:
        return f"Region not specified: {e}"
    except PartialCredentialsError as e:
        return f"Incomplete credentials: {e}"


if __name__ == "__main__":
    # Replace var below with the phone number you want to send SMS to
    phone_number = "+61481021999"
    message = "Hello, this is a test message from AWS SNS!"

    response = send_sms(phone_number, message)
    print(response)
