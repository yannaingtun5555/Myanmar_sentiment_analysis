# quick_test.py
import yaml
import psycopg2
from pathlib import Path

# Load config
config_path = Path("../../configs/config.yaml").resolve()
print(f"Loading config from: {config_path}")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database']
print(f"Connecting to: {db_config['database']} on {db_config['host']}:{db_config['port']}")
print(f"Username: {db_config['user']}")

try:
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database'],
        connect_timeout=5
    )
    print("✅ SUCCESS! Connection working!")
    
    # Test a query
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public'")
    table_count = cursor.fetchone()[0]
    print(f"Current tables in database: {table_count}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    
    # Try to connect to default postgres database instead
    print("\nTrying to connect to 'postgres' database...")
    try:
        conn2 = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database='postgres',
            connect_timeout=5
        )
        print("✅ Can connect to 'postgres' database")
        print("⚠️  Database 'youtube_analysis' doesn't exist!")
        print("   Run: CREATE DATABASE youtube_analysis;")
        conn2.close()
    except Exception as e2:
        print(f"❌ Cannot connect to PostgreSQL at all: {e2}")