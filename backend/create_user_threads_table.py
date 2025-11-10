import psycopg2
from psycopg2 import sql

# Replace with your Postgres connection details
DB_HOST = "your_host"
DB_NAME = "your_db"
DB_USER = "your_user"
DB_PASS = "your_pass"

def create_user_threads_table():
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
    cursor = conn.cursor()

    # Create table if not exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS user_threads (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        thread_id VARCHAR(255) NOT NULL UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        -- Add foreign key if you have a users table: FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """
    cursor.execute(create_table_query)
    conn.commit()
    print("Table 'user_threads' created successfully.")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    create_user_threads_table()