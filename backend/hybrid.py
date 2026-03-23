"""
Hybrid Vulnerability Retrieval System
======================================
Combines PostgreSQL structured filtering with FAISS vector similarity search
for efficient CVE-to-device feature mapping.

Optimized for 200,000+ CVE records.
"""

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import faiss
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DBConfig:
    """PostgreSQL connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "cve_db"
    user: str = "postgres"
    password: str = "postgres"


@dataclass
class IndexConfig:
    """FAISS index configuration."""
    index_path: str = "faiss_cve.index"
    metadata_path: str = "faiss_metadata.pkl"
    embedding_dim: int = 384  # all-MiniLM-L6-v2 dimension


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CVERecord:
    """Parsed CVE record with essential fields."""
    cve_id: str
    description: str
    product: str
    problem_type: str
    cvss_score: float
    severity: str


@dataclass
class DeviceFeatures:
    """Device feature structure."""
    vendor: Optional[str] = None
    model: Optional[str] = None
    hardware_components: List[str] = None
    software_components: List[str] = None
    communication_protocols: List[str] = None
    functional_capabilities: List[str] = None
    
    def __post_init__(self):
        self.hardware_components = self.hardware_components or []
        self.software_components = self.software_components or []
        self.communication_protocols = self.communication_protocols or []
        self.functional_capabilities = self.functional_capabilities or []


@dataclass
class VulnerabilityMatch:
    """A matched vulnerability with risk score."""
    cve_id: str
    cvss_score: float
    similarity: float
    risk_score: float
    description: str
    severity: str
    problem_type: str


# =============================================================================
# Step 1 — Parse CVE Records
# =============================================================================

def load_cve_json(file_path: str) -> List[Dict]:
    """
    Load CVE records from a JSON file.
    
    Supports single record, array of records, or newline-delimited JSON.
    
    Args:
        file_path: Path to CVE JSON file
    
    Returns:
        List of raw CVE dictionaries
    """
    records = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Try parsing as standard JSON first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
    except json.JSONDecodeError:
        # Try newline-delimited JSON
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return records


def load_cve_directory(directory_path: str) -> List[Dict]:
    """
    Load all CVE JSON files from a directory recursively.
    
    Args:
        directory_path: Path to directory containing CVE JSON files
    
    Returns:
        List of raw CVE dictionaries
    """
    all_records = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    records = load_cve_json(file_path)
                    all_records.extend(records)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(all_records)} CVE records from {directory_path}")
    return all_records


def parse_cve_record(cve_raw: Dict) -> Optional[CVERecord]:
    """
    Extract required fields from a single CVE 5.x JSON record.
    
    Args:
        cve_raw: Raw CVE dictionary in CVE 5.x format
    
    Returns:
        CVERecord if valid, None if missing required fields
    """
    try:
        # Extract CVE ID
        cve_id = cve_raw.get("cveMetadata", {}).get("cveId", "")
        if not cve_id:
            return None
        
        # Extract description from containers.cna.descriptions
        descriptions = cve_raw.get("containers", {}).get("cna", {}).get("descriptions", [])
        description = ""
        for desc in descriptions:
            lang = desc.get("lang", "en")
            if lang == "en" and desc.get("value"):
                description = desc["value"]
                break
        if not description and descriptions:
            description = descriptions[0].get("value", "")
        
        # Skip records with missing descriptions
        if not description:
            return None
        
        # Extract product from containers.cna.affected
        affected = cve_raw.get("containers", {}).get("cna", {}).get("affected", [])
        product = ""
        if affected:
            product = affected[0].get("product", "")
            # Also try vendor
            if not product:
                product = affected[0].get("vendor", "")
        
        # Extract CVSS score and severity from containers.adp[].metrics
        cvss_score = 0.0
        severity = "UNKNOWN"
        
        adp_list = cve_raw.get("containers", {}).get("adp", [])
        for adp in adp_list:
            metrics = adp.get("metrics", [])
            if isinstance(metrics, list):
                for m in metrics:
                    if "cvssV3_1" in m:
                        cvss_score = float(m["cvssV3_1"].get("baseScore", 0.0))
                        severity = m["cvssV3_1"].get("baseSeverity", "UNKNOWN")
                        break
                    elif "cvssV3_0" in m:
                        cvss_score = float(m["cvssV3_0"].get("baseScore", 0.0))
                        severity = m["cvssV3_0"].get("baseSeverity", "UNKNOWN")
                        break
            elif isinstance(metrics, dict):
                if "cvssV3_1" in metrics:
                    cvss_score = float(metrics["cvssV3_1"].get("baseScore", 0.0))
                    severity = metrics["cvssV3_1"].get("baseSeverity", "UNKNOWN")
            if cvss_score > 0:
                break
        
        # Fallback: check cna.metrics
        if cvss_score == 0.0:
            cna_metrics = cve_raw.get("containers", {}).get("cna", {}).get("metrics", [])
            for m in cna_metrics:
                if "cvssV3_1" in m:
                    cvss_score = float(m["cvssV3_1"].get("baseScore", 0.0))
                    severity = m["cvssV3_1"].get("baseSeverity", "UNKNOWN")
                    break
                elif "cvssV3_0" in m:
                    cvss_score = float(m["cvssV3_0"].get("baseScore", 0.0))
                    severity = m["cvssV3_0"].get("baseSeverity", "UNKNOWN")
                    break
        
        # Extract problem type (CWE)
        problem_type = ""
        problem_types = cve_raw.get("containers", {}).get("cna", {}).get("problemTypes", [])
        for pt in problem_types:
            descs = pt.get("descriptions", [])
            for d in descs:
                if d.get("description"):
                    problem_type = d["description"]
                    break
            if problem_type:
                break
        
        # Also check adp for problem types
        if not problem_type:
            for adp in adp_list:
                problem_types = adp.get("problemTypes", [])
                for pt in problem_types:
                    descs = pt.get("descriptions", [])
                    for d in descs:
                        if d.get("description"):
                            problem_type = d["description"]
                            break
                    if problem_type:
                        break
                if problem_type:
                    break
        
        return CVERecord(
            cve_id=cve_id,
            description=description,
            product=product,
            problem_type=problem_type,
            cvss_score=cvss_score,
            severity=severity
        )
    
    except Exception as e:
        return None


def parse_cve_dataset(source: str) -> List[CVERecord]:
    """
    Parse CVE records from a file or directory.
    
    Args:
        source: Path to JSON file or directory
    
    Returns:
        List of parsed CVERecord objects
    """
    if os.path.isdir(source):
        raw_data = load_cve_directory(source)
    else:
        raw_data = load_cve_json(source)
    
    records = []
    for cve_raw in raw_data:
        record = parse_cve_record(cve_raw)
        if record:
            records.append(record)
    
    print(f"Parsed {len(records)} valid CVE records from {len(raw_data)} total")
    return records


# =============================================================================
# Repository-based CVE Loading
# =============================================================================

def clone_cve_repository(
    repo_url: str,
    local_path: str = "cvelistV5",
    shallow: bool = True
) -> str:
    """
    Clone a CVE repository from GitHub.
    
    Args:
        repo_url: GitHub repository URL (e.g., https://github.com/CVEProject/cvelistV5)
        local_path: Local directory to clone into
        shallow: Use shallow clone (--depth 1) to save space/time
    
    Returns:
        Path to cloned repository
    """
    import subprocess
    
    if os.path.exists(local_path):
        print(f"Repository already exists at {local_path}")
        # Pull latest changes
        try:
            subprocess.run(
                ["git", "-C", local_path, "pull"],
                check=True,
                capture_output=True
            )
            print("Pulled latest changes")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not pull updates: {e}")
        return local_path
    
    # Clone repository
    print(f"Cloning repository from {repo_url}...")
    cmd = ["git", "clone"]
    if shallow:
        cmd.extend(["--depth", "1"])
    cmd.extend([repo_url, local_path])
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully cloned to {local_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone repository: {e}")
    
    return local_path


def process_cve_repository(
    repo_url: str,
    db_config: DBConfig,
    local_path: str = "cvelistV5",
    cves_dir: str = "cves",
    start_year: int = 1999,
    end_year: int = 2026,
    batch_size: int = 1000,
    skip_clone: bool = False
) -> int:
    """
    Process CVE repository from GitHub and store all CVEs in database.
    
    Handles the standard CVEProject/cvelistV5 structure:
    cves/
    ├── 1999/
    │   ├── 0xxx/
    │   │   ├── CVE-1999-0001.json
    │   │   └── ...
    │   └── 1xxx/
    │       └── ...
    ├── 2024/
    │   └── ...
    └── ...
    
    Args:
        repo_url: GitHub repository URL
        db_config: PostgreSQL configuration
        local_path: Local directory for cloned repo
        cves_dir: Name of CVE directory in repo (usually "cves")
        start_year: First year to process (default: 1999)
        end_year: Last year to process (default: 2026)
        batch_size: Records per database batch insert
        skip_clone: Skip cloning if repo already exists locally
    
    Returns:
        Total number of CVE records stored
    """
    print("=" * 60)
    print("CVE REPOSITORY PROCESSING")
    print("=" * 60)
    
    # Step 1: Clone or update repository
    if not skip_clone:
        print(f"\n[1] Cloning/updating repository...")
        clone_cve_repository(repo_url, local_path)
    else:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Repository not found at {local_path}")
        print(f"\n[1] Using existing repository at {local_path}")
    
    # Step 2: Find CVEs directory
    cves_path = os.path.join(local_path, cves_dir)
    if not os.path.exists(cves_path):
        raise FileNotFoundError(f"CVEs directory not found at {cves_path}")
    
    print(f"\n[2] Processing CVEs from {cves_path}")
    
    # Step 3: Create database table
    print(f"\n[3] Preparing database...")
    create_cve_table(db_config)
    
    # Step 4: Process each year directory
    total_processed = 0
    total_stored = 0
    all_records = []
    
    print(f"\n[4] Processing years {start_year} to {end_year}...")
    
    for year in range(start_year, end_year + 1):
        year_path = os.path.join(cves_path, str(year))
        
        if not os.path.exists(year_path):
            continue
        
        year_count = 0
        
        # Walk through all subdirectories (e.g., 0xxx, 1xxx, etc.)
        for root, dirs, files in os.walk(year_path):
            for file in files:
                if file.endswith('.json') and file.startswith('CVE-'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        raw_records = load_cve_json(file_path)
                        for cve_raw in raw_records:
                            record = parse_cve_record(cve_raw)
                            if record:
                                all_records.append(record)
                                year_count += 1
                                total_processed += 1
                                
                                # Batch insert when we reach batch_size
                                if len(all_records) >= batch_size:
                                    store_cve_postgres_batch(all_records, db_config)
                                    total_stored += len(all_records)
                                    all_records = []
                                    
                    except Exception as e:
                        # Skip problematic files silently
                        continue
        
        if year_count > 0:
            print(f"  Year {year}: {year_count} CVEs")
    
    # Insert remaining records
    if all_records:
        store_cve_postgres_batch(all_records, db_config)
        total_stored += len(all_records)
    
    print(f"\n[5] Complete!")
    print(f"  Total processed: {total_processed}")
    print(f"  Total stored: {total_stored}")
    
    print("\n" + "=" * 60)
    print("REPOSITORY PROCESSING COMPLETE")
    print("=" * 60)
    
    return total_stored


def store_cve_postgres_batch(
    records: List[CVERecord],
    config: DBConfig
):
    """
    Insert a batch of CVE records into PostgreSQL.
    
    Args:
        records: List of CVERecord objects
        config: Database configuration
    """
    if not records:
        return
    
    insert_sql = """
    INSERT INTO cve_records (cve_id, description, product, problem_type, cvss_score, severity)
    VALUES %s
    ON CONFLICT (cve_id) DO UPDATE SET
        description = EXCLUDED.description,
        product = EXCLUDED.product,
        problem_type = EXCLUDED.problem_type,
        cvss_score = EXCLUDED.cvss_score,
        severity = EXCLUDED.severity
    """
    
    conn = get_db_connection(config)
    try:
        with conn.cursor() as cur:
            values = [
                (r.cve_id, r.description, r.product, r.problem_type, r.cvss_score, r.severity)
                for r in records
            ]
            execute_values(cur, insert_sql, values)
        conn.commit()
    finally:
        conn.close()


def load_cves_from_repo(
    repo_url: str,
    db_config: DBConfig = None,
    index_config: IndexConfig = None,
    local_path: str = "cvelistV5",
    start_year: int = 1999,
    end_year: int = 2026,
    build_index: bool = True,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[int, Optional['FAISSIndex']]:
    """
    Complete pipeline to load CVEs from GitHub repository.
    
    1. Clones/updates the repository
    2. Parses all CVE files
    3. Stores in PostgreSQL
    4. Optionally builds FAISS index
    
    Args:
        repo_url: GitHub repository URL (e.g., https://github.com/CVEProject/cvelistV5)
        db_config: PostgreSQL configuration
        index_config: FAISS index configuration
        local_path: Local directory for cloned repo
        start_year: First year to process
        end_year: Last year to process
        build_index: Whether to build FAISS index after loading
        model_name: Embedding model name
    
    Returns:
        Tuple of (total_records, FAISSIndex or None)
    
    Example:
        # Load from official CVE repository
        total, index = load_cves_from_repo(
            "https://github.com/CVEProject/cvelistV5",
            db_config=DBConfig(host="localhost", database="cve_db"),
            start_year=2020,
            end_year=2024
        )
    """
    db_config = db_config or DBConfig()
    index_config = index_config or IndexConfig()
    
    # Process repository and store in database
    total = process_cve_repository(
        repo_url=repo_url,
        db_config=db_config,
        local_path=local_path,
        start_year=start_year,
        end_year=end_year
    )
    
    faiss_index = None
    
    if build_index and total > 0:
        print("\n" + "=" * 60)
        print("BUILDING FAISS INDEX")
        print("=" * 60)
        
        # Load all records from database
        print("\n[1] Loading records from database...")
        records = load_all_cves_from_db(db_config)
        
        # Build index
        print("\n[2] Generating embeddings and building index...")
        model = EmbeddingModel(model_name)
        faiss_index = build_faiss_index(records, model, index_config)
        
        print("\n" + "=" * 60)
        print("INDEX BUILD COMPLETE")
        print("=" * 60)
    
    return total, faiss_index


def load_all_cves_from_db(
    config: DBConfig,
    limit: int = None
) -> List[CVERecord]:
    """
    Load all CVE records from PostgreSQL database.
    
    Args:
        config: Database configuration
        limit: Maximum records to load (None for all)
    
    Returns:
        List of CVERecord objects
    """
    query = """
    SELECT cve_id, description, product, problem_type, cvss_score, severity
    FROM cve_records
    ORDER BY cvss_score DESC
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    conn = get_db_connection(config)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
        
        records = [
            CVERecord(
                cve_id=row["cve_id"],
                description=row["description"] or "",
                product=row["product"] or "",
                problem_type=row["problem_type"] or "",
                cvss_score=row["cvss_score"] or 0.0,
                severity=row["severity"] or "UNKNOWN"
            )
            for row in rows
        ]
        
        print(f"Loaded {len(records)} records from database")
        return records
    finally:
        conn.close()


def get_cve_stats_from_db(config: DBConfig) -> Dict:
    """
    Get CVE statistics from the database.
    
    Args:
        config: Database configuration
    
    Returns:
        Dictionary with statistics
    """
    conn = get_db_connection(config)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Total count
            cur.execute("SELECT COUNT(*) as total FROM cve_records")
            total = cur.fetchone()["total"]
            
            # Count by severity
            cur.execute("""
                SELECT severity, COUNT(*) as count 
                FROM cve_records 
                GROUP BY severity 
                ORDER BY count DESC
            """)
            severity_counts = {row["severity"]: row["count"] for row in cur.fetchall()}
            
            # Count by year (extract from CVE ID)
            cur.execute("""
                SELECT 
                    SUBSTRING(cve_id FROM 5 FOR 4) as year,
                    COUNT(*) as count
                FROM cve_records
                GROUP BY year
                ORDER BY year DESC
                LIMIT 10
            """)
            year_counts = {row["year"]: row["count"] for row in cur.fetchall()}
            
            # Average CVSS
            cur.execute("SELECT AVG(cvss_score) as avg_cvss FROM cve_records WHERE cvss_score > 0")
            avg_cvss = cur.fetchone()["avg_cvss"] or 0.0
        
        return {
            "total_cves": total,
            "severity_distribution": severity_counts,
            "recent_years": year_counts,
            "average_cvss": round(avg_cvss, 2)
        }
    finally:
        conn.close()


# =============================================================================
# Step 2 — Store Structured Metadata in PostgreSQL
# =============================================================================

def get_db_connection(config: DBConfig):
    """
    Create a PostgreSQL database connection.
    
    Args:
        config: Database configuration
    
    Returns:
        psycopg2 connection object
    """
    return psycopg2.connect(
        host=config.host,
        port=config.port,
        database=config.database,
        user=config.user,
        password=config.password
    )


def create_cve_table(config: DBConfig):
    """
    Create the cve_records table if it doesn't exist.
    
    Args:
        config: Database configuration
    """
    create_sql = """
    CREATE TABLE IF NOT EXISTS cve_records (
        cve_id TEXT PRIMARY KEY,
        description TEXT,
        product TEXT,
        problem_type TEXT,
        cvss_score FLOAT,
        severity TEXT
    );
    
    -- Create indexes for efficient filtering
    CREATE INDEX IF NOT EXISTS idx_cve_product ON cve_records (product);
    CREATE INDEX IF NOT EXISTS idx_cve_severity ON cve_records (severity);
    CREATE INDEX IF NOT EXISTS idx_cve_cvss ON cve_records (cvss_score);
    
    -- Full-text search index on product
    CREATE INDEX IF NOT EXISTS idx_cve_product_gin ON cve_records USING gin(to_tsvector('english', product));
    """
    
    conn = get_db_connection(config)
    try:
        with conn.cursor() as cur:
            cur.execute(create_sql)
        conn.commit()
        print("Created cve_records table and indexes")
    finally:
        conn.close()


def store_cve_postgres(
    records: List[CVERecord],
    config: DBConfig,
    batch_size: int = 1000
):
    """
    Insert CVE records into PostgreSQL in batches.
    
    Args:
        records: List of CVERecord objects
        config: Database configuration
        batch_size: Number of records per batch insert
    """
    # Create table if needed
    create_cve_table(config)
    
    insert_sql = """
    INSERT INTO cve_records (cve_id, description, product, problem_type, cvss_score, severity)
    VALUES %s
    ON CONFLICT (cve_id) DO UPDATE SET
        description = EXCLUDED.description,
        product = EXCLUDED.product,
        problem_type = EXCLUDED.problem_type,
        cvss_score = EXCLUDED.cvss_score,
        severity = EXCLUDED.severity
    """
    
    conn = get_db_connection(config)
    try:
        with conn.cursor() as cur:
            # Process in batches
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                values = [
                    (r.cve_id, r.description, r.product, r.problem_type, r.cvss_score, r.severity)
                    for r in batch
                ]
                execute_values(cur, insert_sql, values)
                
                if (i + batch_size) % 10000 == 0:
                    print(f"Inserted {min(i + batch_size, len(records))}/{len(records)} records")
        
        conn.commit()
        print(f"Successfully stored {len(records)} CVE records in PostgreSQL")
    finally:
        conn.close()


def query_cves_by_product(
    search_terms: List[str],
    config: DBConfig,
    limit: int = 1000
) -> List[CVERecord]:
    """
    Query CVEs by product name using ILIKE.
    
    Args:
        search_terms: List of product/vendor terms to search
        config: Database configuration
        limit: Maximum results
    
    Returns:
        List of matching CVERecord objects
    """
    if not search_terms:
        return []
    
    # Build OR conditions for each search term
    conditions = " OR ".join(["product ILIKE %s" for _ in search_terms])
    query = f"""
    SELECT cve_id, description, product, problem_type, cvss_score, severity
    FROM cve_records
    WHERE {conditions}
    ORDER BY cvss_score DESC
    LIMIT %s
    """
    
    params = [f"%{term}%" for term in search_terms] + [limit]
    
    conn = get_db_connection(config)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        
        return [
            CVERecord(
                cve_id=row["cve_id"],
                description=row["description"],
                product=row["product"] or "",
                problem_type=row["problem_type"] or "",
                cvss_score=row["cvss_score"] or 0.0,
                severity=row["severity"] or "UNKNOWN"
            )
            for row in rows
        ]
    finally:
        conn.close()


# =============================================================================
# Step 3 — Prepare Embedding Text
# =============================================================================

def prepare_embedding_text(record: CVERecord) -> str:
    """
    Construct embedding text for a CVE record.
    
    Format: "CVE ID + product + description + problem type"
    
    Args:
        record: CVERecord object
    
    Returns:
        Combined text for embedding
    """
    parts = [record.cve_id]
    
    if record.product:
        parts.append(record.product)
    
    parts.append(record.description)
    
    if record.problem_type:
        parts.append(record.problem_type)
    
    return " ".join(parts)


def prepare_batch_embedding_texts(records: List[CVERecord]) -> List[str]:
    """
    Prepare embedding texts for multiple CVE records.
    
    Args:
        records: List of CVERecord objects
    
    Returns:
        List of embedding texts
    """
    return [prepare_embedding_text(r) for r in records]


# =============================================================================
# Step 4 — Create Embeddings
# =============================================================================

class EmbeddingModel:
    """Wrapper for the sentence transformer embedding model."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize_embeddings: L2 normalize embeddings
        
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings
        )
        
        return embeddings


def generate_embeddings(
    records: List[CVERecord],
    model: EmbeddingModel,
    batch_size: int = 64
) -> np.ndarray:
    """
    Generate embeddings for CVE records.
    
    Args:
        records: List of CVERecord objects
        model: EmbeddingModel instance
        batch_size: Batch size for encoding
    
    Returns:
        Numpy array of embeddings
    """
    texts = prepare_batch_embedding_texts(records)
    embeddings = model.encode(texts, batch_size=batch_size)
    print(f"Generated embeddings: {embeddings.shape}")
    return embeddings


# =============================================================================
# Step 5 — Build FAISS Vector Index
# =============================================================================

class FAISSIndex:
    """FAISS index for CVE vector similarity search."""
    
    def __init__(self, config: IndexConfig = None):
        """
        Initialize FAISS index.
        
        Args:
            config: Index configuration
        """
        self.config = config or IndexConfig()
        self.index = None
        self.metadata = []  # List of {cve_id, cvss_score, severity}
    
    def build(
        self,
        embeddings: np.ndarray,
        records: List[CVERecord],
        use_ivf: bool = True,
        nlist: int = 100
    ):
        """
        Build FAISS index from embeddings.
        
        For large datasets (200k+), uses IVF index for efficiency.
        
        Args:
            embeddings: Numpy array of embeddings (n, dim)
            records: List of CVERecord objects (same order as embeddings)
            use_ivf: Use IVF index for large datasets
            nlist: Number of clusters for IVF
        """
        n_vectors, dim = embeddings.shape
        
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))
        
        if use_ivf and n_vectors > 10000:
            # Use IVF index for large datasets
            # IVF with inner product (for normalized vectors, equivalent to cosine)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, min(nlist, n_vectors // 10))
            
            # Train the index
            print(f"Training IVF index with {n_vectors} vectors...")
            self.index.train(embeddings)
            
            # Add vectors
            self.index.add(embeddings)
            
            # Set search parameters
            self.index.nprobe = 10  # Number of clusters to search
        else:
            # Use flat index for smaller datasets
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
        
        # Store metadata
        self.metadata = [
            {
                "cve_id": r.cve_id,
                "cvss_score": r.cvss_score,
                "severity": r.severity,
                "description": r.description,
                "problem_type": r.problem_type
            }
            for r in records
        ]
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def save(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.config.index_path)
        
        with open(self.config.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Saved index to {self.config.index_path}")
        print(f"Saved metadata to {self.config.metadata_path}")
    
    def load(self):
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(self.config.index_path):
            raise FileNotFoundError(f"Index file not found: {self.config.index_path}")
        
        self.index = faiss.read_index(self.config.index_path)
        
        with open(self.config.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding (1, dim) or (dim,)
            k: Number of results
        
        Returns:
            List of (index, similarity_score) tuples
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = np.ascontiguousarray(query_embedding.astype('float32'))
        
        # Search
        similarities, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx >= 0:  # FAISS returns -1 for empty results
                results.append((int(idx), float(sim)))
        
        return results
    
    def get_metadata(self, index: int) -> Dict:
        """Get metadata for a vector index."""
        if 0 <= index < len(self.metadata):
            return self.metadata[index]
        return {}


def build_faiss_index(
    records: List[CVERecord],
    model: EmbeddingModel,
    config: IndexConfig = None,
    batch_size: int = 64
) -> FAISSIndex:
    """
    Build and save FAISS index for CVE records.
    
    Args:
        records: List of CVERecord objects
        model: EmbeddingModel instance
        config: Index configuration
        batch_size: Batch size for embedding generation
    
    Returns:
        FAISSIndex instance
    """
    config = config or IndexConfig()
    
    # Generate embeddings
    embeddings = generate_embeddings(records, model, batch_size)
    
    # Build index
    faiss_index = FAISSIndex(config)
    faiss_index.build(embeddings, records)
    
    # Save to disk
    faiss_index.save()
    
    return faiss_index


# =============================================================================
# Step 6 — Construct Device Feature Query
# =============================================================================

def construct_device_query(device: DeviceFeatures) -> str:
    """
    Convert device features into search query text.
    
    Args:
        device: DeviceFeatures object
    
    Returns:
        Search query text
    """
    parts = []
    
    # Add vendor and model
    if device.vendor:
        parts.append(device.vendor)
    if device.model:
        parts.append(device.model)
    
    # Add hardware components
    for comp in device.hardware_components:
        if comp:
            parts.append(comp)
    
    # Add software components
    for comp in device.software_components:
        if comp:
            parts.append(comp)
    
    # Add communication protocols with context
    for proto in device.communication_protocols:
        if proto:
            parts.append(f"{proto} protocol")
    
    # Add functional capabilities with context
    for func in device.functional_capabilities:
        if func:
            parts.append(f"{func} device")
    
    query_text = " ".join(parts)
    return query_text


def load_device_features(data: Dict) -> DeviceFeatures:
    """
    Load device features from a dictionary.
    
    Args:
        data: Dictionary with device feature fields
    
    Returns:
        DeviceFeatures object
    """
    return DeviceFeatures(
        vendor=data.get("vendor"),
        model=data.get("model"),
        hardware_components=data.get("hardware_components", []),
        software_components=data.get("software_components", []),
        communication_protocols=data.get("communication_protocols", []),
        functional_capabilities=data.get("functional_capabilities", [])
    )


# =============================================================================
# Step 7 — Retrieval Pipeline
# =============================================================================

def retrieve_similar_cves(
    device: DeviceFeatures,
    model: EmbeddingModel,
    faiss_index: FAISSIndex,
    db_config: DBConfig = None,
    use_structured_filter: bool = True,
    k: int = 20
) -> List[Tuple[Dict, float]]:
    """
    Retrieve similar CVEs using hybrid retrieval.
    
    Stage 1: Structured filtering via PostgreSQL (if enabled)
    Stage 2: Vector similarity via FAISS
    
    Args:
        device: DeviceFeatures object
        model: EmbeddingModel instance
        faiss_index: FAISSIndex instance
        db_config: PostgreSQL configuration (optional)
        use_structured_filter: Enable structured pre-filtering
        k: Number of results
    
    Returns:
        List of (cve_metadata, similarity_score) tuples
    """
    results = []
    
    # Stage 1: Structured filtering (optional)
    structured_cves = set()
    if use_structured_filter and db_config:
        search_terms = []
        if device.vendor:
            search_terms.append(device.vendor)
        if device.model:
            search_terms.append(device.model)
        search_terms.extend(device.hardware_components)
        search_terms.extend(device.software_components)
        
        if search_terms:
            filtered_records = query_cves_by_product(search_terms, db_config, limit=1000)
            structured_cves = {r.cve_id for r in filtered_records}
            print(f"Structured filter matched {len(structured_cves)} CVEs")
    
    # Stage 2: Vector similarity search
    query_text = construct_device_query(device)
    query_embedding = model.encode([query_text], show_progress=False)[0]
    
    # Search FAISS
    search_results = faiss_index.search(query_embedding, k=k * 2)  # Get extra for filtering
    
    for idx, similarity in search_results:
        metadata = faiss_index.get_metadata(idx)
        if not metadata:
            continue
        
        # If structured filtering is enabled, boost or filter
        if use_structured_filter and structured_cves:
            if metadata["cve_id"] in structured_cves:
                # Boost similarity for structured matches
                similarity = min(1.0, similarity * 1.2)
        
        results.append((metadata, similarity))
    
    # Sort by similarity and return top k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


# =============================================================================
# Step 8 — Risk Scoring
# =============================================================================

def compute_risk_score(similarity: float, cvss_score: float) -> float:
    """
    Compute risk score for a CVE match.
    
    risk_score = similarity * cvss_score
    
    Args:
        similarity: Cosine similarity score (0-1)
        cvss_score: CVSS base score (0-10)
    
    Returns:
        Risk score (0-10)
    """
    return round(similarity * cvss_score, 2)


def rank_vulnerabilities(
    matches: List[Tuple[Dict, float]]
) -> List[VulnerabilityMatch]:
    """
    Rank vulnerabilities by risk score.
    
    Args:
        matches: List of (cve_metadata, similarity_score) tuples
    
    Returns:
        List of VulnerabilityMatch objects sorted by risk score
    """
    ranked = []
    
    for metadata, similarity in matches:
        cvss = metadata.get("cvss_score", 0.0)
        risk = compute_risk_score(similarity, cvss)
        
        ranked.append(VulnerabilityMatch(
            cve_id=metadata["cve_id"],
            cvss_score=cvss,
            similarity=round(similarity, 4),
            risk_score=risk,
            description=metadata.get("description", ""),
            severity=metadata.get("severity", "UNKNOWN"),
            problem_type=metadata.get("problem_type", "")
        ))
    
    # Sort by risk score descending
    ranked.sort(key=lambda x: x.risk_score, reverse=True)
    return ranked


# =============================================================================
# Step 9 — Generate Vulnerability Assessment
# =============================================================================

def generate_vulnerability_report(
    device: DeviceFeatures,
    ranked_vulnerabilities: List[VulnerabilityMatch],
    top_n: int = 10
) -> Dict:
    """
    Generate structured vulnerability assessment report.
    
    Args:
        device: DeviceFeatures object
        ranked_vulnerabilities: List of ranked VulnerabilityMatch objects
        top_n: Number of top vulnerabilities to include
    
    Returns:
        Vulnerability assessment report dictionary
    """
    top_vulns = ranked_vulnerabilities[:top_n]
    
    report = {
        "device": {
            "vendor": device.vendor or "",
            "model": device.model or ""
        },
        "assessment_summary": {
            "total_matches": len(ranked_vulnerabilities),
            "critical_count": sum(1 for v in top_vulns if v.severity == "CRITICAL"),
            "high_count": sum(1 for v in top_vulns if v.severity == "HIGH"),
            "medium_count": sum(1 for v in top_vulns if v.severity == "MEDIUM"),
            "low_count": sum(1 for v in top_vulns if v.severity == "LOW"),
            "max_risk_score": top_vulns[0].risk_score if top_vulns else 0.0,
            "avg_risk_score": round(
                sum(v.risk_score for v in top_vulns) / len(top_vulns), 2
            ) if top_vulns else 0.0
        },
        "top_vulnerabilities": [
            {
                "cve_id": v.cve_id,
                "cvss_score": v.cvss_score,
                "similarity": v.similarity,
                "risk_score": v.risk_score,
                "severity": v.severity,
                "description": v.description[:300] + "..." if len(v.description) > 300 else v.description,
                "problem_type": v.problem_type
            }
            for v in top_vulns
        ]
    }
    
    return report


# =============================================================================
# Main Pipeline Functions
# =============================================================================

def run_etl_pipeline(
    cve_source: str,
    db_config: DBConfig = None,
    index_config: IndexConfig = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Run complete ETL pipeline: parse CVEs, store in PostgreSQL, build FAISS index.
    
    Args:
        cve_source: Path to CVE JSON file or directory
        db_config: PostgreSQL configuration
        index_config: FAISS index configuration
        model_name: Embedding model name
    """
    print("=" * 60)
    print("CVE ETL PIPELINE")
    print("=" * 60)
    
    db_config = db_config or DBConfig()
    index_config = index_config or IndexConfig()
    
    # Step 1: Parse CVE records
    print("\n[Step 1] Parsing CVE dataset...")
    records = parse_cve_dataset(cve_source)
    
    if not records:
        raise ValueError("No valid CVE records found")
    
    # Step 2: Store in PostgreSQL
    print("\n[Step 2] Storing in PostgreSQL...")
    store_cve_postgres(records, db_config)
    
    # Step 3-5: Generate embeddings and build FAISS index
    print("\n[Step 3-5] Building vector index...")
    model = EmbeddingModel(model_name)
    build_faiss_index(records, model, index_config)
    
    print("\n" + "=" * 60)
    print("ETL PIPELINE COMPLETE")
    print("=" * 60)


def run_vulnerability_assessment(
    device_features: Dict,
    db_config: DBConfig = None,
    index_config: IndexConfig = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_structured_filter: bool = True,
    k: int = 20,
    top_n: int = 10
) -> Dict:
    """
    Run vulnerability assessment for a device.
    
    Args:
        device_features: Device feature dictionary
        db_config: PostgreSQL configuration
        index_config: FAISS index configuration
        model_name: Embedding model name
        use_structured_filter: Enable PostgreSQL pre-filtering
        k: Number of candidates from vector search
        top_n: Number of top vulnerabilities in report
    
    Returns:
        Vulnerability assessment report
    """
    print("=" * 60)
    print("VULNERABILITY ASSESSMENT")
    print("=" * 60)
    
    index_config = index_config or IndexConfig()
    
    # Load device features
    device = load_device_features(device_features)
    print(f"\nDevice: {device.vendor} {device.model}")
    
    # Load embedding model
    print("\n[1] Loading embedding model...")
    model = EmbeddingModel(model_name)
    
    # Load FAISS index
    print("\n[2] Loading vector index...")
    faiss_index = FAISSIndex(index_config)
    faiss_index.load()
    
    # Retrieve similar CVEs
    print("\n[3] Retrieving similar CVEs...")
    matches = retrieve_similar_cves(
        device=device,
        model=model,
        faiss_index=faiss_index,
        db_config=db_config if use_structured_filter else None,
        use_structured_filter=use_structured_filter,
        k=k
    )
    print(f"Found {len(matches)} matches")
    
    # Rank by risk score
    print("\n[4] Computing risk scores...")
    ranked = rank_vulnerabilities(matches)
    
    # Generate report
    print("\n[5] Generating report...")
    report = generate_vulnerability_report(device, ranked, top_n)
    
    print("\n" + "=" * 60)
    print("ASSESSMENT COMPLETE")
    print("=" * 60)
    
    return report


# =============================================================================
# In-Memory Mode (No PostgreSQL Required)
# =============================================================================

class InMemoryVulnerabilitySystem:
    """
    In-memory vulnerability assessment system.
    
    Use this when PostgreSQL is not available.
    Stores CVE metadata in memory alongside FAISS index.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_config: IndexConfig = None
    ):
        """
        Initialize the in-memory system.
        
        Args:
            model_name: Embedding model name
            index_config: FAISS index configuration
        """
        self.model = None
        self.faiss_index = None
        self.records = []
        self.model_name = model_name
        self.index_config = index_config or IndexConfig()
    
    def load_model(self):
        """Load the embedding model."""
        if self.model is None:
            self.model = EmbeddingModel(self.model_name)
    
    def build_index(self, cve_source: str, batch_size: int = 64):
        """
        Build index from CVE source.
        
        Args:
            cve_source: Path to CVE JSON file or directory
            batch_size: Batch size for embedding generation
        """
        self.load_model()
        
        # Parse CVEs
        self.records = parse_cve_dataset(cve_source)
        
        # Build FAISS index
        self.faiss_index = build_faiss_index(
            self.records,
            self.model,
            self.index_config,
            batch_size
        )
    
    def load_index(self):
        """Load pre-built index from disk."""
        self.load_model()
        self.faiss_index = FAISSIndex(self.index_config)
        self.faiss_index.load()
    
    def assess_device(
        self,
        device_features: Dict,
        k: int = 20,
        top_n: int = 10
    ) -> Dict:
        """
        Assess device vulnerabilities.
        
        Args:
            device_features: Device feature dictionary
            k: Number of candidates
            top_n: Top vulnerabilities to return
        
        Returns:
            Vulnerability assessment report
        """
        if self.faiss_index is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        self.load_model()
        device = load_device_features(device_features)
        
        # Retrieve similar CVEs
        matches = retrieve_similar_cves(
            device=device,
            model=self.model,
            faiss_index=self.faiss_index,
            db_config=None,
            use_structured_filter=False,
            k=k
        )
        
        # Rank and generate report
        ranked = rank_vulnerabilities(matches)
        report = generate_vulnerability_report(device, ranked, top_n)
        
        return report


# =============================================================================
# Main — Extract features from relay.pdf & assess against official CVE repo
# =============================================================================

if __name__ == "__main__":
    import json
    from feature_extraction import extract_device_features

    # ─── Configuration ───────────────────────────────────────────────
    PDF_PATH = "relay.pdf"
    CVE_REPO_URL = "https://github.com/CVEProject/cvelistV5"
    CVE_LOCAL_PATH = "cvelistV5"
    DB_CONFIG = DBConfig()                 # default: localhost / cve_db
    INDEX_CONFIG = IndexConfig()           # default: faiss_cve.index

    START_YEAR = 1999
    END_YEAR = 2026

    print("=" * 60)
    print("VULNERABILITY ASSESSMENT PIPELINE")
    print("=" * 60)

    # ─── Step 1: Extract device features from relay.pdf ──────────────
    print("\n[Step 1] Extracting device features from", PDF_PATH, "...")
    device_features = extract_device_features(PDF_PATH)
    print("\nDevice features:")
    print(json.dumps(device_features, indent=2))

    # ─── Step 2: Clone / update the official CVE repository ──────────
    print("\n[Step 2] Cloning / updating official CVE repository...")
    total_stored, faiss_idx = load_cves_from_repo(
        repo_url=CVE_REPO_URL,
        db_config=DB_CONFIG,
        index_config=INDEX_CONFIG,
        local_path=CVE_LOCAL_PATH,
        start_year=START_YEAR,
        end_year=END_YEAR,
        build_index=True,
    )
    print(f"Total CVEs stored: {total_stored}")

    # ─── Step 3: Run vulnerability assessment ────────────────────────
    print("\n[Step 3] Running vulnerability assessment...")
    report = run_vulnerability_assessment(
        device_features=device_features,
        db_config=DB_CONFIG,
        index_config=INDEX_CONFIG,
        use_structured_filter=True,
        k=20,
        top_n=10,
    )

    # ─── Step 4: Print report ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VULNERABILITY ASSESSMENT REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    # Save report to file
    report_path = "vulnerability_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
