import os
from dotenv import load_dotenv


load_dotenv()


class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_STUDENTS_TABLE = os.getenv("SUPABASE_STUDENTS_TABLE", "students")
    SUPABASE_ATTENDANCE_TABLE = os.getenv("SUPABASE_ATTENDANCE_TABLE", "attendance")

    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") in {"1", "true", "True", "yes", "on"}

    R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
    R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")
