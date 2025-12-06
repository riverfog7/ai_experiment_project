import os


def get_db_conn_str():
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_port = os.getenv("POSTGRES_PORT")
    postgres_db = os.getenv("POSTGRES_DB")
    if not all([postgres_user, postgres_password, postgres_host, postgres_port, postgres_db]):
        raise ValueError("One or more PostgreSQL environment variables are not set.")

    return f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

def get_study_name():
    study_name = os.getenv("OPTUNA_STUDY_NAME")
    if not study_name:
        raise ValueError("OPTUNA_STUDY_NAME environment variable is not set.")
    return study_name
