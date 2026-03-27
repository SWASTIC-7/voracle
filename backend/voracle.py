
import torch, gc

gc.collect()
torch.cuda.empty_cache()
import os

for root, dirs, files in os.walk('/kaggle/input'):
    for file in files:
        print(os.path.join(root, file))
import pdfplumber

def extract_text_from_pdf(file_path: str) -> str:

    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    print(f"Extracted {len(full_text)} characters from {file_path}")
    return full_text

# loading a model to extract features from the text generated from pdf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_extraction_model(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
):
    """
    Load a quantised causal-LM and its tokenizer.

    Returns:
        (model, tokenizer) tuple.
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
    )
    print(f"Loaded model: {model_name}")
    return model, tokenizer

# make chunks of the text
def chunk_text_by_tokens(text, tokenizer, max_tokens=1200):
    tokens = tokenizer(text)["input_ids"]

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_ids = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append(chunk_text)

    return chunks

# creates the json from chunks
import json
from typing import Dict, List, Optional

_EXTRACTION_PROMPT = """\
TASK: Extract industrial device features from the provided text.

WHAT TO EXTRACT (BE SELECTIVE):

vendor: Company/manufacturer name
- Example: "ABB", "Siemens", "General Electric"

model: Specific device model/part number
- Example: "RH(G) 110", "SIPROTEC 5 7SJ85"

hardware_components: ONLY core component/module model numbers
- Include: Specific chip/processor/relay/module models (e.g., "RH(G) 113", "ARM Cortex-A53")
- Exclude: Generic terms ("CPU", "Relay", "coil"), voltage ranges, materials ("Silver"), descriptions
- Extract model numbers and variants, NOT specifications

software_components: Firmware/OS version if mentioned
- Example: "firmware v3.42", "Embedded Linux 5.4"

communication_protocols: Industrial protocols by name
- Example: "Modbus TCP", "IEC 61850 MMS", "IEC 61850 GOOSE", "DNP3"
- Use EMPTY LIST [] if none found (never use "none")

functional_capabilities: Device functions (brief, non-redundant)
- Example: "distance protection", "latching relay operation", "relay switching"
- Exclude: Specifications, voltage ranges, mounting details
- Keep concise and avoid repetition

EXTRACTION RULES:
1. Extract ONLY explicit information from text
2. Use null for fields not mentioned, [] for empty lists
3. NO specification details (voltages, impedances, resistances) in lists
4. NO redundant/near-duplicate items
5. NO generic terms or material names

OUTPUT FORMAT:
{{
  "vendor": null,
  "model": null,
  "hardware_components": [],
  "software_components": [],
  "communication_protocols": [],
  "functional_capabilities": []
}}

TEXT TO ANALYZE:
{chunk}

YOUR EXTRACTION:
"""


def _extract_valid_json(text: str) -> Optional[Dict]:
    """Return the last valid JSON object found in *text*, or None."""
    candidates = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def _clean_extracted_item(item: str) -> Optional[str]:
    """Filter out noise from individual extracted items."""
    if not item:
        return None
    
    item = item.strip()
    
    # Filter out common noise patterns
    noise_patterns = [
        # Electrical specs
        r"^\d+\.\d+.*V",  # Voltage ranges like "32.9... 61.8 V"
        r"^\d+\s*Ω",  # Impedance/resistance
        r"^\d+\s*%",  # Percentages
        r"^\d+\s*(µ|m)?H",  # Henri/milli-Henri
        # Generic/vague terms
        r"^(CPU|Relay|coil|Terminal|Rack|A1|B1|Silver|silver|none|Contact)$",
        r"^(Mounting|Power supply|Protection|Indication|Socket|Slot|Port)$",
        # Specifications
        r"^(at\s+\d+°?C|\d+%|approx|max|min|max\.|about)",  # Temperature/tolerance specs
        # Physical descriptions (not capabilities)
        r"^(Rear view|Side view|Front view|Top view|diagram|drawing|example|picture|photo)",
        r"^(center|coil connection|make contact|break contact|changeover point)",
        # Assembly/installation instructions
        r"^(simply|snap|screw|bolt|fastening|locking|mounting|insert|place)",
        r"^(plug|unplug|remove|install|replace|attach)",
        # Diagram labels
        r"^[A-Z]\d+$|^(point|node|connection)",  # Like "A2", "B1"
        # Low-value identifiers
        r"^(D\s+Sp|Z-\d+|[A-Z]\d{5}|[A-Z]{1,2}\d{1,3}$)",  # Random codes
        # Redundant phrases
        r"^view$|^system$|^operation$",
    ]
    
    for pattern in noise_patterns:
        if re.search(pattern, item, re.IGNORECASE):
            return None
    
    return item


def _deduplicate_list(items: list) -> list:
    """Remove duplicates and near-duplicates, keep longest descriptive version."""
    if not items:
        return []
    
    # Normalize and clean
    cleaned = []
    seen_normalized = set()
    
    for item in items:
        clean_item = _clean_extracted_item(item)
        if clean_item:
            # Normalize for deduplication (lowercase, no extra spaces)
            normalized = " ".join(clean_item.lower().split())
            
            # Check for exact duplicate
            if normalized in seen_normalized:
                continue
            
            # Check for substring duplicates (e.g., "relay operation" vs "relay operation buffer")
            is_subset = False
            for existing in seen_normalized:
                # If this item is contained in an existing item (or vice versa), skip it
                # Unless it's significantly different
                if normalized in existing or existing in normalized:
                    # Only skip if it's a true substring, not just word order
                    norm_words = set(normalized.split())
                    exist_words = set(existing.split())
                    if norm_words < exist_words or exist_words < norm_words:
                        is_subset = True
                        break
            
            if not is_subset:
                seen_normalized.add(normalized)
                cleaned.append(clean_item)
    
    return sorted(cleaned)


def _clean_extraction_result(obj: Dict) -> Dict:
    """Clean and deduplicate extraction results."""
    if not obj:
        return obj
    
    # Ensure vendor and model are strings, not lists
    if isinstance(obj.get("vendor"), list):
        obj["vendor"] = obj["vendor"][0] if obj["vendor"] else None
    if isinstance(obj.get("model"), list):
        obj["model"] = obj["model"][0] if obj["model"] else None
    
    # Clean lists
    obj["hardware_components"] = _deduplicate_list(obj.get("hardware_components", []))
    obj["software_components"] = _deduplicate_list(obj.get("software_components", []))
    obj["communication_protocols"] = _deduplicate_list(obj.get("communication_protocols", []))
    obj["functional_capabilities"] = _deduplicate_list(obj.get("functional_capabilities", []))
    
    # Remove "none" protocol if it slipped through
    if obj.get("communication_protocols") == ["none"]:
        obj["communication_protocols"] = []
    
    return obj


def _is_hardcoded_example(obj: Dict) -> bool:
    """Check if the extracted object matches hardcoded examples or shows lazy behavior."""
    if not obj or not isinstance(obj, dict):
        return False
    
    # Safely extract string values, handling type mismatches
    vendor = obj.get("vendor")
    model = obj.get("model")
    
    # Ensure they're strings (not lists or other types)
    if isinstance(vendor, list):
        vendor = " ".join(vendor) if vendor else ""
    else:
        vendor = (vendor or "").lower().strip()
    
    if isinstance(model, list):
        model = " ".join(model) if model else ""
    else:
        model = (model or "").lower().strip()
    
    hw = str(obj.get("hardware_components", [])).lower()
    sw = str(obj.get("software_components", [])).lower()
    protocols = str(obj.get("communication_protocols", [])).lower()
    
    # Known example patterns to reject
    intel_atom_pattern = vendor == "intel" and "atom c3000" in model
    siemens_pattern = vendor == "siemens" and "siprotec" in model
    
    if intel_atom_pattern or siemens_pattern:
        return True
    
    # Reject if model looks suspiciously like an example (e.g., contains "5" or "7")
    # but vendor is clearly an example manufacturer
    if vendor in ["intel", "siemens", "abb", "ge"] and not model:
        return False  # Just vendor with no model is OK
    
    return False


def _run_extraction_on_chunk(
    chunk: str, model, tokenizer
) -> Optional[Dict]:
    """Run the LLM on a single chunk and return parsed JSON."""
    prompt = _EXTRACTION_PROMPT.format(chunk=chunk)

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    with torch.no_grad():
        # Use higher temp + sampling to force the model to generate different outputs
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=1.0,  # Higher temperature for more randomness
            top_p=0.95,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    extracted = _extract_valid_json(result)
    
    # Reject if hardcoded example
    if extracted and _is_hardcoded_example(extracted):
        print(f"  ⚠️  Rejected: Output matches known example pattern")
        return None
    
    # Clean and deduplicate results
    if extracted:
        extracted = _clean_extraction_result(extracted)
    
    # Debug: Show what was extracted
    if extracted:
        has_content = any([
            extracted.get("vendor"),
            extracted.get("model"),
            extracted.get("hardware_components"),
            extracted.get("software_components"),
            extracted.get("communication_protocols"),
            extracted.get("functional_capabilities"),
        ])
        if has_content:
            vendor = extracted.get('vendor', 'N/A')
            model = extracted.get('model', '')
            hw_count = len(extracted.get('hardware_components', []))
            print(f"  ✓ {vendor} {model} ({hw_count} hw)")
        else:
            print(f"  ○ Empty extraction")
            return extracted
    else:
        print(f"  ✗ No valid JSON")
        return None
    
    return extracted

import re

def _merge_parsed_results(parsed_results: List[Dict]) -> Dict:
    """Merge per-chunk extraction dicts into a single feature dict."""
    merged: Dict = {
        "vendor": None,
        "model": None,
        "hardware_components": set(),
        "software_components": set(),
        "communication_protocols": set(),
        "functional_capabilities": set(),
    }

    for item in parsed_results:
        if item.get("vendor") and not merged["vendor"]:
            merged["vendor"] = item["vendor"]
        if item.get("model") and not merged["model"]:
            merged["model"] = item["model"]

        for key in (
            "hardware_components",
            "software_components",
            "communication_protocols",
            "functional_capabilities",
        ):
            value = item.get(key)
            if isinstance(value, list):
                merged[key].update(value)

    # Convert sets to sorted lists
    for key in merged:
        if isinstance(merged[key], set):
            merged[key] = sorted(list(merged[key]))
    
    # Apply final cleanup
    merged = _clean_extraction_result(merged)
    
    # Convert None to empty string for vendor/model
    merged["vendor"] = merged["vendor"] or ""
    merged["model"] = merged["model"] or ""

    return merged

def extract_device_features(
    pdf_path: str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    max_tokens_per_chunk: int = 1200,
) -> Dict:
    """
    End-to-end feature extraction from a PDF datasheet.

    Args:
        pdf_path: Path to the device PDF.
        model_name: HuggingFace model name for extraction.
        max_tokens_per_chunk: Max tokens per text chunk.

    Returns:
        Dictionary with keys: vendor, model, hardware_components,
        software_components, communication_protocols, functional_capabilities.
    """
    # 1. Extract text
    full_text = extract_text_from_pdf(pdf_path)

    # 2. Load LLM
    llm, tokenizer = load_extraction_model(model_name)

    # 3. Chunk
    chunks = chunk_text_by_tokens(full_text, tokenizer, max_tokens_per_chunk)
    print(f"Split into {len(chunks)} chunks")

    # 4. Extract per chunk
    parsed_results = []
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx + 1}/{len(chunks)}")
        obj = _run_extraction_on_chunk(chunk, llm, tokenizer)
        if obj:
            parsed_results.append(obj)
    print(f"Valid parsed chunks: {len(parsed_results)}")

    # 5. Merge
    features = _merge_parsed_results(parsed_results)

    # 6. Cleanup
    del llm, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print("\nExtracted device features:")
    print(json.dumps(features, indent=2))
    return features

# if __name__ == "__main__":
#     import sys

#     pdf =  "relay.pdf"
#     features = extract_device_features(pdf)
#     print(json.dumps(features, indent=2))

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

# defining data structure to be used for mapping
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

# helper functions to load and parse data from CVE
def load_cve_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        return []


def load_cve_directory(directory_path: str) -> List[Dict]:
    """Load all CVE JSON files from a directory recursively."""
    all_records = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    records = load_cve_data(file_path)
                    all_records.extend(records)
                except Exception as e:
                    continue
    print(f"Loaded {len(all_records)} CVE records from {directory_path}")
    return all_records


def extract_cve_fields(cve_raw: Dict) -> Optional[CVERecord]:
    try:
        cve_id = cve_raw.get("cveMetadata", {}).get("cveId", "")

        descriptions = cve_raw.get("containers", {}).get("cna", {}).get("descriptions", [])
        description = ""
        for desc in descriptions:
            if desc.get("lang", "en") == "en" and desc.get("value"):
                description = desc["value"]
                break
        if not description and descriptions:
            description = descriptions[0].get("value", "")

        if not description:
            return None

        affected = cve_raw.get("containers", {}).get("cna", {}).get("affected", [])
        product = ""
        if affected:
            product = affected[0].get("product", "")

        cvss_score = 0.0
        adp_list = cve_raw.get("containers", {}).get("adp", [])
        for adp in adp_list:
            metrics = adp.get("metrics", {})
            if isinstance(metrics, list):
                for m in metrics:
                    if "cvssV3_1" in m:
                        cvss_score = float(m["cvssV3_1"].get("baseScore", 0.0))
                        break
            elif isinstance(metrics, dict):
                cvss_v3 = metrics.get("cvssV3_1", {})
                if cvss_v3:
                    cvss_score = float(cvss_v3.get("baseScore", 0.0))
            if cvss_score > 0:
                break

        if cvss_score == 0.0:
            cna_metrics = cve_raw.get("containers", {}).get("cna", {}).get("metrics", [])
            for m in cna_metrics:
                if "cvssV3_1" in m:
                    cvss_score = float(m["cvssV3_1"].get("baseScore", 0.0))
                    break
                elif "cvssV3_0" in m:
                    cvss_score = float(m["cvssV3_0"].get("baseScore", 0.0))
                    break

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

        # Extract severity (default to 'UNKNOWN' if not found)
        severity = "UNKNOWN"
        adp_list = cve_raw.get("containers", {}).get("adp", [])
        for adp in adp_list:
            metrics = adp.get("metrics", {})
            if isinstance(metrics, list):
                for m in metrics:
                    if "cvssV3_1" in m:
                        severity = m["cvssV3_1"].get("baseSeverity", "UNKNOWN")
                        break
                    elif "cvssV3_0" in m:
                        severity = m["cvssV3_0"].get("baseSeverity", "UNKNOWN")
                        break
            elif isinstance(metrics, dict):
                cvss_v3 = metrics.get("cvssV3_1", {})
                if cvss_v3:
                    severity = cvss_v3.get("baseSeverity", "UNKNOWN")
            if severity != "UNKNOWN":
                break

        if severity == "UNKNOWN":
            cna_metrics = cve_raw.get("containers", {}).get("cna", {}).get("metrics", [])
            for m in cna_metrics:
                if "cvssV3_1" in m:
                    severity = m["cvssV3_1"].get("baseSeverity", "UNKNOWN")
                    break
                elif "cvssV3_0" in m:
                    severity = m["cvssV3_0"].get("baseSeverity", "UNKNOWN")
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
        print(f"Error parsing CVE: {e}")
        return None


def parse_cve_dataset(file_path: str) -> List[CVERecord]:
    if os.path.isdir(file_path):
        raw_data = load_cve_directory(file_path)
    else:
        raw_data = load_cve_data(file_path)
    records = []

    for cve_raw in raw_data:
        record = extract_cve_fields(cve_raw)
        if record:
            records.append(record)

    print(f"Parsed {len(records)} valid CVE records from {len(raw_data)} total")
    return records

# function to clone cve repo
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

# function to create connection to postgres database
def get_db_connection(config: DBConfig):
    return psycopg2.connect(
        host=config.host,
        port=config.port,
        database=config.database,
        user=config.user,
        password=config.password
    )

# function to create table on postgres
def create_cve_table(config: DBConfig):
    conn = get_db_connection(config)
    try:
        with conn.cursor() as cur:
            cur.execute(create_sql)
        conn.commit()
        print("Created cve_records table and indexes")
    finally:
        conn.close()

# helper functions to store data in postgres and query
def store_cve_postgres(
    records: List[CVERecord],
    config: DBConfig,
    batch_size: int = 1000
):

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

    if not search_terms:
        return []

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

# stores cve records in batches
def store_cve_postgres_batch(
    records: List[CVERecord],
    config: DBConfig
):

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

# main function to process the cve data
# calls clone function -- find cves directory -- walks throught all the file in each year and store in postgres
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


    # Step 1: Clone or update repository
    if not skip_clone:
        print(f"\n[1] Cloning/updating repository...")
        clone_cve_repository(repo_url, local_path)
    else:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Repository not found at {local_path}")
        print(f"\n[1] Using existing repository at {local_path}")

    cves_path = os.path.join(local_path, cves_dir)
    if not os.path.exists(cves_path):
        raise FileNotFoundError(f"CVEs directory not found at {cves_path}")

    print(f"\n[2] Processing CVEs from {cves_path}")

    print(f"\n[3] Preparing database...")
    create_cve_table(db_config)

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

                    # using helper functions to parse cve json
                    try:
                        raw_records = load_cve_json(file_path)
                        for cve_raw in raw_records:
                            record = parse_cve_record(cve_raw)
                            if record:
                                all_records.append(record)
                                year_count += 1
                                total_processed += 1

                                if len(all_records) >= batch_size:
                                    store_cve_postgres_batch(all_records, db_config)
                                    total_stored += len(all_records)
                                    all_records = []

                    except Exception as e:
                        continue

        if year_count > 0:
            print(f"  Year {year}: {year_count} CVEs")

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

# to load cve from database
def load_all_cves_from_db(
    config: DBConfig,
    limit: int = None
) -> List[CVERecord]:

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

# get stats in a disctionary
def get_cve_stats_from_db(config: DBConfig) -> Dict:

    conn = get_db_connection(config)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:

            cur.execute("SELECT COUNT(*) as total FROM cve_records")
            total = cur.fetchone()["total"]


            cur.execute("""
                SELECT severity, COUNT(*) as count
                FROM cve_records
                GROUP BY severity
                ORDER BY count DESC
            """)
            severity_counts = {row["severity"]: row["count"] for row in cur.fetchall()}


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

class EmbeddingModel:
    """Wrapper for the sentence transformer embedding model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

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

def prepare_embedding_text(record: CVERecord) -> str:
    parts = [record.cve_id]

    if record.product:
        parts.append(record.product)

    parts.append(record.description)

    if record.problem_type:
        parts.append(record.problem_type)

    return " ".join(parts)


def prepare_batch_embedding_texts(records: List[CVERecord]) -> List[str]:
    return [prepare_embedding_text(r) for r in records]

def generate_embeddings(
    records: List[CVERecord],
    model: EmbeddingModel,
    batch_size: int = 64
) -> np.ndarray:
    texts = prepare_batch_embedding_texts(records)
    embeddings = model.encode(texts, batch_size=batch_size)
    print(f"Generated embeddings: {embeddings.shape}")
    return embeddings

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

# Complete pipeline to load CVEs from GitHub repository.

# 1. Clones/updates the repository
# 2. Parses all CVE files
# 3. Stores in PostgreSQL
# 4. Optionally builds FAISS index
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

# converts device feature to query
def construct_device_query(device: DeviceFeatures) -> str:
    parts = []

    if device.vendor:
        parts.append(device.vendor)
    if device.model:
        parts.append(device.model)

    for comp in device.hardware_components:
        if comp:
            parts.append(comp)

    for comp in device.software_components:
        if comp:
            parts.append(comp)

    for proto in device.communication_protocols:
        if proto:
            parts.append(f"{proto} protocol")

    for func in device.functional_capabilities:
        if func:
            parts.append(f"{func} device")

    query_text = " ".join(parts)
    return query_text


def load_device_features(data: Dict) -> DeviceFeatures:

    return DeviceFeatures(
        vendor=data.get("vendor"),
        model=data.get("model"),
        hardware_components=data.get("hardware_components", []),
        software_components=data.get("software_components", []),
        communication_protocols=data.get("communication_protocols", []),
        functional_capabilities=data.get("functional_capabilities", [])
    )

# retrieve simalal cves from postgress of vector search on faiss
def retrieve_similar_cves(
    device: DeviceFeatures,
    model: EmbeddingModel,
    faiss_index: FAISSIndex,
    db_config: DBConfig = None,
    use_structured_filter: bool = True,
    k: int = 20
) -> List[Tuple[Dict, float]]:

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

# defines score to a vulnerability
def compute_risk_score(similarity: float, cvss_score: float) -> float:

    return round(similarity * cvss_score, 2)


def rank_vulnerabilities(
    matches: List[Tuple[Dict, float]]
) -> List[VulnerabilityMatch]:

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

def generate_vulnerability_report(
    device: DeviceFeatures,
    ranked_vulnerabilities: List[VulnerabilityMatch],
    top_n: int = 10
) -> Dict:

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

# complete pipeline
def run_etl_pipeline(
    cve_source: str,
    db_config: DBConfig = None,
    index_config: IndexConfig = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):

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

# return report
def run_vulnerability_assessment(
    device_features: Dict,
    db_config: DBConfig = None,
    index_config: IndexConfig = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_structured_filter: bool = True,
    k: int = 20,
    top_n: int = 10
) -> Dict:

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

# in-memory support
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

if __name__ == "__main__":
    import json


    # \u2500\u2500\u2500 Configuration \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n    PDF_PATH = "relay.pdf"
    CVE_REPO_URL = "https://github.com/CVEProject/cvelistV5"
    CVE_LOCAL_PATH = "cvelistV5"
    # DB_CONFIG = DBConfig()                 # default: localhost / cve_db
    INDEX_CONFIG = IndexConfig()           # default: faiss_cve.index
    PDF_PATH = "relay.pdf"  # Path to the device specification PDF
    START_YEAR = 1999
    END_YEAR = 2026

    print("=" * 60)
    print("VULNERABILITY ASSESSMENT PIPELINE")
    print("=" * 60)

    # \u2500\u2500\u2500 Step 1: Extract device features from relay.pdf \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n    print("\n[Step 1] Extracting device features from", PDF_PATH, "...")
    device_features = extract_device_features(PDF_PATH)
    print("\nDevice features:")
    print(json.dumps(device_features, indent=2))
    clone_cve_repository(CVE_REPO_URL, CVE_LOCAL_PATH)
    # \u2500\u2500\u2500 Step 2: Initialize and build in-memory system \u2500\u2500\u2500\u2500\u2500\u2500\u2500\n    print("\n[Step 2] Initializing and building in-memory vulnerability system...")
    in_memory_system = InMemoryVulnerabilitySystem(index_config=INDEX_CONFIG)
    in_memory_system.build_index(cve_source="cvelistV5/cves") # Assuming CVEs are cloned locally
    # Note: For a fresh run without local CVEs, you might need to clone first:
    # from chv-c1zZ-x1Q import clone_cve_repository
    

    # \u2500\u2500\u2500 Step 3: Run vulnerability assessment \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n    print("\n[Step 3] Running vulnerability assessment...")
    report = in_memory_system.assess_device(
        device_features=device_features,
        k=20,
        top_n=10,
    )

    # \u2500\u2500\u2500 Step 4: Print report \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n    print("\n" + "=" * 60)
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