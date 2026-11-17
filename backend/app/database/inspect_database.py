"""
Database Inspection Script - Read-Only Analysis

This script inspects the SQLite personal finance database and provides
comprehensive information about its structure and content.

IMPORTANT: This script only reads data. No modifications are made.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Database path
DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "database" / "personal_finance.db"


class DatabaseInspector:
    """Inspect SQLite database without modifying it."""
    
    def __init__(self, db_path: str):
        """Initialize database inspector."""
        self.db_path = db_path
        self.connection = None
        self.cursor = None
    
    def connect(self) -> bool:
        """Connect to database (read-only)."""
        try:
            # Open connection in read-only mode
            self.connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            self.cursor = self.connection.cursor()
            print(f"✓ Connected to database: {self.db_path}\n")
            return True
        except sqlite3.Error as e:
            print(f"✗ Error connecting to database: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("\n✓ Database connection closed.")
    
    def get_tables(self) -> List[str]:
        """Get list of all tables in database."""
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
        self.cursor.execute(query)
        tables = [row[0] for row in self.cursor.fetchall()]
        return tables
    
    def get_table_schema(self, table_name: str) -> List[Tuple[str, str, bool, Any]]:
        """Get schema information for a table."""
        query = f"PRAGMA table_info({table_name})"
        self.cursor.execute(query)
        schema = self.cursor.fetchall()
        return schema
    
    def get_row_count(self, table_name: str) -> int:
        """Get number of rows in a table."""
        query = f"SELECT COUNT(*) FROM {table_name}"
        self.cursor.execute(query)
        count = self.cursor.fetchone()[0]
        return count
    
    def get_first_n_rows(self, table_name: str, n: int = 5) -> List[Tuple]:
        """Get first N rows from table."""
        query = f"SELECT * FROM {table_name} LIMIT {n}"
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        return rows
    
    def get_column_names(self, table_name: str) -> List[str]:
        """Get column names for a table."""
        schema = self.get_table_schema(table_name)
        return [col[1] for col in schema]
    
    def count_unique_values(self, table_name: str, column_name: str) -> int:
        """Count unique values in a column."""
        query = f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name}"
        self.cursor.execute(query)
        count = self.cursor.fetchone()[0]
        return count
    
    def count_null_values(self, table_name: str, column_name: str) -> int:
        """Count NULL values in a column."""
        query = f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NULL"
        self.cursor.execute(query)
        count = self.cursor.fetchone()[0]
        return count
    
    def get_unique_categories(self) -> List[str]:
        """Get unique transaction categories."""
        try:
            query = "SELECT DISTINCT category FROM transactions ORDER BY category"
            self.cursor.execute(query)
            categories = [row[0] for row in self.cursor.fetchall()]
            return categories
        except sqlite3.OperationalError:
            return []
    
    def get_transaction_types(self) -> Dict[str, int]:
        """Get count of income vs expense transactions."""
        counts = {}
        
        # Try different column name variations
        type_columns = ["transaction_type", "type", "category_type"]
        
        for col in type_columns:
            try:
                query = f"SELECT DISTINCT {col} FROM transactions"
                self.cursor.execute(query)
                types = [row[0] for row in self.cursor.fetchall()]
                
                for t in types:
                    query = f"SELECT COUNT(*) FROM transactions WHERE {col} = '{t}'"
                    self.cursor.execute(query)
                    count = self.cursor.fetchone()[0]
                    counts[f"{t}"] = count
                
                if counts:
                    return counts
            except sqlite3.OperationalError:
                continue
        
        return counts
    
    def get_transaction_date_range(self) -> Tuple[str, str]:
        """Get minimum and maximum transaction dates."""
        try:
            query = """
            SELECT 
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM transactions
            """
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            return result if result[0] else ("N/A", "N/A")
        except sqlite3.OperationalError:
            return ("N/A", "N/A")
    
    def count_duplicates(self, table_name: str) -> int:
        """Count potential duplicate transactions."""
        try:
            # Check for duplicates based on common transaction fields
            query = f"""
            SELECT COUNT(*) FROM (
                SELECT 
                    COUNT(*) as cnt
                FROM {table_name}
                GROUP BY date, amount, category
                HAVING COUNT(*) > 1
            )
            """
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            return result[0] if result else 0
        except sqlite3.OperationalError:
            return 0
    
    def count_users(self) -> int:
        """Count number of unique users."""
        try:
            query = "SELECT COUNT(DISTINCT user_id) FROM transactions"
            self.cursor.execute(query)
            count = self.cursor.fetchone()[0]
            return count
        except sqlite3.OperationalError:
            return 0
    
    def print_separator(self, char: str = "=", length: int = 80):
        """Print a separator line."""
        print(char * length)
    
    def print_section(self, title: str):
        """Print a section title."""
        self.print_separator()
        print(f"\n{title}\n")
    
    def inspect_and_report(self):
        """Run complete database inspection and print report."""
        if not self.connect():
            return
        
        try:
            print("\n")
            self.print_separator("=", 90)
            print("AI-POWERED PERSONAL FINANCE - DATABASE INSPECTION REPORT")
            print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.print_separator("=", 90)
            
            # 1. Database tables
            self.print_section("1. DATABASE TABLES")
            tables = self.get_tables()
            if tables:
                print(f"Number of tables: {len(tables)}\n")
                for i, table in enumerate(tables, 1):
                    print(f"  {i}. {table}")
            else:
                print("No tables found.")
            
            # 2. Detailed table information
            self.print_section("2. TABLE SCHEMA & STRUCTURE")
            for table in tables:
                print(f"\n{'─' * 80}")
                print(f"TABLE: {table.upper()}")
                print(f"{'─' * 80}")
                
                # Get schema
                schema = self.get_table_schema(table)
                row_count = self.get_row_count(table)
                
                print(f"Row Count: {row_count:,}\n")
                print("Columns:")
                print(f"{'  Column Name':<30} {'Data Type':<15} {'Null':<10}")
                print(f"  {'-' * 75}")
                
                for col_id, col_name, col_type, col_notnull, col_default, col_pk in schema:
                    notnull = "NOT NULL" if col_notnull else "NULL"
                    print(f"  {col_name:<30} {col_type:<15} {notnull:<10}")
                
                # Sample data
                if row_count > 0:
                    print(f"\nFirst 5 Rows:")
                    print(f"  {'-' * 75}")
                    rows = self.get_first_n_rows(table, 5)
                    col_names = self.get_column_names(table)
                    
                    for row_idx, row in enumerate(rows, 1):
                        print(f"\n  Row {row_idx}:")
                        for col_name, value in zip(col_names, row):
                            # Truncate long values
                            value_str = str(value)
                            if len(value_str) > 50:
                                value_str = value_str[:47] + "..."
                            print(f"    {col_name}: {value_str}")
            
            # 3. User statistics
            self.print_section("3. USER STATISTICS")
            num_users = self.count_users()
            print(f"Total unique users: {num_users:,}")
            
            # 4. Transaction statistics
            self.print_section("4. TRANSACTION STATISTICS")
            
            # Total transactions
            try:
                query = "SELECT COUNT(*) FROM transactions"
                self.cursor.execute(query)
                total_transactions = self.cursor.fetchone()[0]
                print(f"Total transactions: {total_transactions:,}")
            except sqlite3.OperationalError:
                print("Total transactions: N/A")
            
            # Transaction types
            trans_types = self.get_transaction_types()
            if trans_types:
                print(f"\nTransaction Type Breakdown:")
                for trans_type, count in sorted(trans_types.items()):
                    print(f"  • {trans_type}: {count:,}")
            
            # Date range
            min_date, max_date = self.get_transaction_date_range()
            print(f"\nTransaction Date Range:")
            print(f"  • Earliest: {min_date}")
            print(f"  • Latest: {max_date}")
            
            # 5. Categories
            self.print_section("5. TRANSACTION CATEGORIES")
            categories = self.get_unique_categories()
            if categories:
                print(f"Total unique categories: {len(categories)}\n")
                for i, category in enumerate(categories, 1):
                    # Get count for this category
                    try:
                        query = "SELECT COUNT(*) FROM transactions WHERE category = ?"
                        self.cursor.execute(query, (category,))
                        count = self.cursor.fetchone()[0]
                        print(f"  {i}. {category:<30} ({count:,} transactions)")
                    except:
                        print(f"  {i}. {category}")
            else:
                print("No categories found or category column doesn't exist.")
            
            # 6. Missing values analysis
            self.print_section("6. MISSING VALUES ANALYSIS")
            print("Checking for NULL values in important columns...\n")
            
            for table in tables:
                columns = self.get_column_names(table)
                null_counts = {}
                
                for col in columns:
                    null_count = self.count_null_values(table, col)
                    if null_count > 0:
                        null_counts[col] = null_count
                
                if null_counts:
                    print(f"Table: {table}")
                    for col, count in sorted(null_counts.items()):
                        total = self.get_row_count(table)
                        percentage = (count / total * 100) if total > 0 else 0
                        print(f"  • {col}: {count:,} NULL values ({percentage:.1f}%)")
                    print()
                else:
                    print(f"Table: {table} - No NULL values found\n")
            
            # 7. Duplicate analysis
            self.print_section("7. DUPLICATE TRANSACTION ANALYSIS")
            
            try:
                dup_count = self.count_duplicates("transactions")
                print(f"Potential duplicate transactions (same date, amount, category): {dup_count}")
            except:
                print("Could not analyze duplicates (transactions table may not exist)")
            
            # 8. Summary statistics
            self.print_section("8. DATABASE SUMMARY")
            print(f"Database file: {self.db_path}")
            print(f"Database file size: {os.path.getsize(self.db_path) / (1024*1024):.2f} MB")
            print(f"Total tables: {len(tables)}")
            
            total_rows = sum(self.get_row_count(table) for table in tables)
            print(f"Total rows across all tables: {total_rows:,}")
            
            self.print_separator("=", 90)
            print("\n✓ Database inspection completed successfully.\n")
            
        except Exception as e:
            print(f"\n✗ Error during inspection: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.close()


def main():
    """Main entry point."""
    print("\n" + "="*90)
    print("PERSONAL FINANCE DATABASE INSPECTOR")
    print("="*90 + "\n")
    
    if not DB_PATH.exists():
        print(f"✗ Database file not found at: {DB_PATH}")
        print(f"  Expected location: data/database/personal_finance.db")
        return
    
    inspector = DatabaseInspector(str(DB_PATH))
    inspector.inspect_and_report()


if __name__ == "__main__":
    main()
