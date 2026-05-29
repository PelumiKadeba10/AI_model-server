import boto3
from config import Config


s3 = boto3.client(
    "s3",
    endpoint_url=(
        f"https://{Config.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    ),
    aws_access_key_id=Config.R2_ACCESS_KEY_ID,
    aws_secret_access_key=Config.R2_SECRET_ACCESS_KEY,
    region_name="auto",
)



def upload_file_to_r2(
    local_path,
    remote_path,
):
    s3.upload_file(
        local_path,
        Config.R2_BUCKET_NAME,
        remote_path,
    )

    return (
        f"{Config.R2_PUBLIC_URL}/{remote_path}"
    )