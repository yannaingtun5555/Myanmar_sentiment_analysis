# src/db/connection.py

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager

import pymysql
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# CONFIGURATION
# =====================================================

class DatabaseConfig:
    """Database configuration management"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize DB config from multiple sources:
        1. config.yaml file
        2. Environment variables
        3. Default values
        """
        self.config = self._load_config(config_file)
        
    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        
        # Default config
        config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),  # 5436 for PostgreSQL, 3306 for MySQL
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'youtube_analysis'),
            'db_type': os.getenv('DB_TYPE', 'postgresql')  # 'postgresql' or 'mysql'
        }
        
        # Load from YAML if provided
        if config_file and Path(config_file).exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if 'database' in yaml_config:
                        config.update(yaml_config['database'])
            except ImportError:
                logger.warning("PyYAML not installed, using environment variables only")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        return config
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get database connection parameters"""
        return self.config
    
    def get_connection_string(self) -> str:
        """Get SQLAlchemy connection string"""
        if self.config['db_type'] == 'postgresql':
            return f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        else:
            return f"mysql+pymysql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"


# =====================================================
# DATABASE CONNECTION MANAGER
# =====================================================

class DatabaseManager:
    """Main database manager for the sentiment analysis pipeline"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine = None
        self._conn = None
        
    def get_engine(self) -> Engine:
        """Get SQLAlchemy engine"""
        if self._engine is None:
            connection_string = self.config.get_connection_string()
            self._engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
        return self._engine
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if self.config.config['db_type'] == 'postgresql':
            conn = psycopg2.connect(
                host=self.config.config['host'],
                port=self.config.config['port'],
                user=self.config.config['user'],
                password=self.config.config['password'],
                database=self.config.config['database']
            )
        else:
            conn = pymysql.connect(
                host=self.config.config['host'],
                port=self.config.config['port'],
                user=self.config.config['user'],
                password=self.config.config['password'],
                database=self.config.config['database'],
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
        
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database cursors"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict]:
        """Execute SELECT query and return results"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor) if self.config.config['db_type'] == 'postgresql' else conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(query, params or ())
            results = cursor.fetchall()
            cursor.close()
            return results
    
    def execute_insert(self, query: str, params: Optional[Tuple] = None) -> int:
        """Execute INSERT query and return last inserted ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            last_id = cursor.lastrowid if hasattr(cursor, 'lastrowid') else None
            
            # For PostgreSQL
            if not last_id and self.config.config['db_type'] == 'postgresql':
                cursor.execute("SELECT LASTVAL();")
                last_id = cursor.fetchone()[0]
            
            cursor.close()
            return last_id
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute batch INSERT/UPDATE"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            affected = cursor.rowcount
            cursor.close()
            return affected


# =====================================================
# SCHEMA LOADER AND INITIALIZER
# =====================================================

class SchemaLoader:
    """Load and execute SQL schema files"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.project_root = Path(__file__).parent.parent.parent
        
    def find_schema_file(self) -> Optional[Path]:
        """Find the schema.sql file in the project"""
        possible_paths = [
            self.project_root / "src" / "db" / "schema.sql",
            self.project_root / "schema.sql",
            Path(__file__).parent / "schema.sql",
            self.project_root / "configs" / "schema.sql"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found schema file: {path}")
                return path
        
        logger.error("Schema file not found!")
        return None
    
    def read_schema_file(self, schema_path: Path) -> str:
        """Read and preprocess SQL file"""
        with open(schema_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Remove comments and split statements
        statements = self._split_sql_statements(sql_content)
        
        return statements
    
    def _split_sql_statements(self, sql_content: str) -> List[str]:
        """Split SQL file into individual statements"""
        import re
        
        # Remove SQL comments
        sql_content = re.sub(r'--.*$', '', sql_content, flags=re.MULTILINE)
        sql_content = re.sub(r'/\*.*?\*/', '', sql_content, flags=re.DOTALL)
        
        # Split by semicolon
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        return statements
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table already exists"""
        db_type = self.db.config.config['db_type']
        
        if db_type == 'postgresql':
            query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """
        else:  # MySQL
            query = """
                SELECT COUNT(*) as count
                FROM information_schema.tables 
                WHERE table_schema = DATABASE()
                AND table_name = %s;
            """
        
        result = self.db.execute_query(query, (table_name,))
        
        if db_type == 'postgresql':
            return result[0]['exists'] if result else False
        else:
            return result[0]['count'] > 0 if result else False
    
    def create_schema(self, force: bool = False, dry_run: bool = False) -> bool:
        """
        Create database schema from schema.sql file
        
        Args:
            force: Drop existing tables before creating
            dry_run: Only print what would be executed, don't actually run
        
        Returns:
            True if successful, False otherwise
        """
        schema_path = self.find_schema_file()
        if not schema_path:
            logger.error("Cannot proceed without schema.sql file")
            return False
        
        statements = self.read_schema_file(schema_path)
        
        if dry_run:
            logger.info(f"DRY RUN: Would execute {len(statements)} SQL statements")
            for i, stmt in enumerate(statements[:5], 1):  # Show first 5
                logger.info(f"Statement {i}: {stmt[:100]}...")
            return True
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # If force, drop existing tables first
                if force:
                    logger.info("Force mode: Dropping existing tables...")
                    drop_statements = [
                        "DROP TABLE IF EXISTS labeled_dataset CASCADE;",
                        "DROP TABLE IF EXISTS review_queue CASCADE;",
                        "DROP TABLE IF EXISTS final_results CASCADE;",
                        "DROP TABLE IF EXISTS comparison_results CASCADE;",
                        "DROP TABLE IF EXISTS predictions CASCADE;",
                        "DROP TABLE IF EXISTS preprocessed_comments CASCADE;",
                        "DROP TABLE IF EXISTS raw_comments CASCADE;",
                        "DROP TABLE IF EXISTS requests CASCADE;",
                        "DROP FUNCTION IF EXISTS update_request_status CASCADE;",
                        "DROP VIEW IF EXISTS v_pending_reviews CASCADE;",
                        "DROP VIEW IF EXISTS v_disagreements CASCADE;",
                        "DROP VIEW IF EXISTS v_training_data_export CASCADE;",
                    ]
                    
                    for drop_stmt in drop_statements:
                        try:
                            cursor.execute(drop_stmt)
                            logger.debug(f"Executed: {drop_stmt[:50]}...")
                        except Exception as e:
                            logger.warning(f"Drop statement failed (may not exist): {e}")
                
                # Execute schema statements
                for i, statement in enumerate(statements, 1):
                    try:
                        cursor.execute(statement)
                        logger.debug(f"Executed statement {i}/{len(statements)}")
                    except Exception as e:
                        logger.error(f"Error executing statement {i}: {e}")
                        logger.error(f"Statement: {statement[:200]}")
                        raise
                
                conn.commit()
                logger.info(f"Successfully created schema from {schema_path.name}")
                
                # Verify tables were created
                self._verify_schema()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False
    
    def _verify_schema(self):
        """Verify that all required tables were created"""
        expected_tables = [
            'requests', 'raw_comments', 'preprocessed_comments',
            'predictions', 'comparison_results', 'final_results',
            'review_queue', 'labeled_dataset'
        ]
        
        existing_tables = []
        for table in expected_tables:
            if self.check_table_exists(table):
                existing_tables.append(table)
        
        logger.info(f"Tables created: {len(existing_tables)}/{len(expected_tables)}")
        
        missing = set(expected_tables) - set(existing_tables)
        if missing:
            logger.warning(f"Missing tables: {missing}")
        else:
            logger.info("All tables created successfully!")
    
    def reset_sequences(self):
        """Reset all sequences (PostgreSQL only)"""
        if self.db.config.config['db_type'] != 'postgresql':
            logger.warning("reset_sequences only supported for PostgreSQL")
            return
        
        queries = [
            "SELECT setval('requests_req_id_seq', COALESCE((SELECT MAX(req_id) FROM requests), 1), false);",
            "SELECT setval('predictions_id_seq', COALESCE((SELECT MAX(id) FROM predictions), 1), false);",
            "SELECT setval('comparison_results_id_seq', COALESCE((SELECT MAX(id) FROM comparison_results), 1), false);",
            "SELECT setval('final_results_id_seq', COALESCE((SELECT MAX(id) FROM final_results), 1), false);",
            "SELECT setval('review_queue_id_seq', COALESCE((SELECT MAX(id) FROM review_queue), 1), false);",
            "SELECT setval('labeled_dataset_id_seq', COALESCE((SELECT MAX(id) FROM labeled_dataset), 1), false);",
        ]
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for query in queries:
                try:
                    cursor.execute(query)
                except Exception as e:
                    logger.warning(f"Sequence reset failed: {e}")
            conn.commit()


# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def init_database(config_file: Optional[str] = None, force: bool = False, dry_run: bool = False) -> DatabaseManager:
    """
    Initialize database connection and create schema
    
    Args:
        config_file: Path to config.yaml file
        force: Drop existing tables and recreate
        dry_run: Only show what would be executed
    
    Returns:
        DatabaseManager instance
    """
    db_manager = DatabaseManager(DatabaseConfig(config_file))
    schema_loader = SchemaLoader(db_manager)
    
    if schema_loader.create_schema(force=force, dry_run=dry_run):
        logger.info("Database initialization completed")
    else:
        logger.error("Database initialization failed")
        if not dry_run:
            sys.exit(1)
    
    return db_manager


def get_db() -> DatabaseManager:
    """Singleton pattern for database connection"""
    global _db_instance
    if '_db_instance' not in globals():
        _db_instance = DatabaseManager()
    return _db_instance


# =====================================================
# USAGE EXAMPLES
# =====================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Database management for sentiment analysis')
    parser.add_argument('--init', action='store_true', help='Initialize database schema')
    parser.add_argument('--force', action='store_true', help='Drop existing tables before creating')
    parser.add_argument('--dry-run', action='store_true', help='Print SQL without executing')
    parser.add_argument('--config', type=str, help='Path to config.yaml file')
    parser.add_argument('--test', action='store_true', help='Test database connection')
    
    args = parser.parse_args()
    
    if args.test:
        # Test connection
        db = DatabaseManager(DatabaseConfig(args.config))
        try:
            result = db.execute_query("SELECT 1 as test")
            print("✅ Database connection successful!")
            print(f"   Database type: {db.config.config['db_type']}")
            print(f"   Host: {db.config.config['host']}:{db.config.config['port']}")
            print(f"   Database: {db.config.config['database']}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            sys.exit(1)
    
    elif args.init:
        # Initialize database
        db = init_database(
            config_file=args.config,
            force=args.force,
            dry_run=args.dry_run
        )
        if not args.dry_run:
            print("✅ Database schema created successfully!")
    
    else:
        # Example usage
        print("Database Connection Manager for Myanmar Sentiment Analysis")
        print("=" * 50)
        print("Usage:")
        print("  python connection.py --test              # Test connection")
        print("  python connection.py --init              # Create schema")
        print("  python connection.py --init --force      # Reset schema")
        print("  python connection.py --init --dry-run    # Preview SQL")