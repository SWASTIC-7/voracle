"""
Vulnerability Assessment Pipeline for Industrial Devices
=========================================================
This module maps device features to CVE records and computes vulnerability risk scores.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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


@dataclass
class DeviceFeatures:
    """Device feature structure."""
    vendor: Optional[str] = None
    model: Optional[str] = None
    hardware_components: List[str] = field(default_factory=list)
    software_components: List[str] = field(default_factory=list)
    communication_protocols: List[str] = field(default_factory=list)
    functional_capabilities: List[str] = field(default_factory=list)


@dataclass
class FeatureRisk:
    """Risk information for a single feature."""
    feature: str
    category: str
    risk_weight: float
    related_cves: List[Dict[str, Any]]


# =============================================================================
# Step 1 — Parse CVE Dataset
# =============================================================================

def load_cve_data(file_path: str) -> List[Dict]:
    """
    Load CVE records from a JSON file.
    
    Args:
        file_path: Path to CVE JSON file (single record or array)
    
    Returns:
        List of raw CVE dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        return []


def extract_cve_fields(cve_raw: Dict) -> Optional[CVERecord]:
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
        
        # Extract description from containers.cna.descriptions
        descriptions = cve_raw.get("containers", {}).get("cna", {}).get("descriptions", [])
        description = ""
        for desc in descriptions:
            if desc.get("lang", "en") == "en" and desc.get("value"):
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
        
        # Extract CVSS score from containers.adp[].metrics.cvssV3_1.baseScore
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
        
        # Fallback: check cna.metrics
        if cvss_score == 0.0:
            cna_metrics = cve_raw.get("containers", {}).get("cna", {}).get("metrics", [])
            for m in cna_metrics:
                if "cvssV3_1" in m:
                    cvss_score = float(m["cvssV3_1"].get("baseScore", 0.0))
                    break
                elif "cvssV3_0" in m:
                    cvss_score = float(m["cvssV3_0"].get("baseScore", 0.0))
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
            cvss_score=cvss_score
        )
    
    except Exception as e:
        print(f"Error parsing CVE: {e}")
        return None


def parse_cve_dataset(file_path: str) -> List[CVERecord]:
    """
    Load and parse all CVE records from a file.
    
    Args:
        file_path: Path to CVE JSON file
    
    Returns:
        List of parsed CVERecord objects
    """
    raw_data = load_cve_data(file_path)
    records = []
    
    for cve_raw in raw_data:
        record = extract_cve_fields(cve_raw)
        if record:
            records.append(record)
    
    print(f"Parsed {len(records)} valid CVE records from {len(raw_data)} total")
    return records


# =============================================================================
# Step 2 — Construct Feature Text
# =============================================================================

def construct_feature_text(device: DeviceFeatures) -> Tuple[List[str], List[str]]:
    """
    Convert device features into descriptive text sentences.
    
    Args:
        device: DeviceFeatures object
    
    Returns:
        Tuple of (feature_texts, feature_categories)
    """
    feature_texts = []
    categories = []
    
    # Hardware components
    for comp in device.hardware_components:
        if comp:
            feature_texts.append(f"{comp} hardware component")
            categories.append("hardware")
    
    # Software components
    for comp in device.software_components:
        if comp:
            feature_texts.append(f"{comp} software component")
            categories.append("software")
    
    # Communication protocols
    for proto in device.communication_protocols:
        if proto:
            feature_texts.append(f"{proto} communication protocol")
            categories.append("protocol")
    
    # Functional capabilities
    for func in device.functional_capabilities:
        if func:
            feature_texts.append(f"{func} functionality")
            categories.append("functional")
    
    return feature_texts, categories


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
# Step 3 — Generate Embeddings
# =============================================================================

def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load the sentence transformer model.
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        SentenceTransformer model
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        model: SentenceTransformer model
        texts: List of text strings
        batch_size: Batch size for encoding
        show_progress: Show progress bar
    
    Returns:
        Numpy array of embeddings (n_texts, embedding_dim)
    """
    if not texts:
        return np.array([])
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    return embeddings


# =============================================================================
# Step 4 — Compute Feature → CVE Similarity
# =============================================================================

def compute_similarity(
    feature_embeddings: np.ndarray,
    cve_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity matrix between features and CVEs.
    
    Args:
        feature_embeddings: Feature embedding matrix (n_features, dim)
        cve_embeddings: CVE embedding matrix (n_cves, dim)
    
    Returns:
        Similarity matrix (n_features, n_cves)
    """
    if feature_embeddings.size == 0 or cve_embeddings.size == 0:
        return np.array([])
    
    similarity_matrix = cosine_similarity(feature_embeddings, cve_embeddings)
    return similarity_matrix


# =============================================================================
# Step 5 — Filter Relevant CVEs
# =============================================================================

def filter_relevant_cves(
    similarity_matrix: np.ndarray,
    cve_records: List[CVERecord],
    threshold: float = 0.35,
    top_k: int = 5
) -> List[List[Tuple[int, float]]]:
    """
    Filter relevant CVEs for each feature based on similarity threshold.
    
    Args:
        similarity_matrix: Similarity matrix (n_features, n_cves)
        cve_records: List of CVE records
        threshold: Minimum similarity threshold
        top_k: Maximum number of CVEs per feature
    
    Returns:
        List of [(cve_index, similarity_score), ...] for each feature
    """
    n_features = similarity_matrix.shape[0]
    relevant_cves = []
    
    for i in range(n_features):
        feature_similarities = similarity_matrix[i]
        
        # Get indices where similarity > threshold
        above_threshold = np.where(feature_similarities > threshold)[0]
        
        # Sort by similarity descending
        sorted_indices = above_threshold[np.argsort(feature_similarities[above_threshold])[::-1]]
        
        # Take top k
        top_indices = sorted_indices[:top_k]
        
        # Store (index, similarity) tuples
        cve_pairs = [(int(idx), float(feature_similarities[idx])) for idx in top_indices]
        relevant_cves.append(cve_pairs)
    
    return relevant_cves


# =============================================================================
# Step 6 — Compute Vulnerability Weights
# =============================================================================

def compute_feature_weights(
    similarity_matrix: np.ndarray,
    cve_records: List[CVERecord],
    threshold: float = 0.35
) -> np.ndarray:
    """
    Calculate vulnerability weight for each feature.
    
    weight(feature_i) = Σ(sim(feature_i, cve_j) * cvss_score_j)
    
    Args:
        similarity_matrix: Similarity matrix (n_features, n_cves)
        cve_records: List of CVE records with CVSS scores
        threshold: Minimum similarity threshold for inclusion
    
    Returns:
        Normalized feature weights array
    """
    n_features = similarity_matrix.shape[0]
    weights = np.zeros(n_features)
    
    # Get CVSS scores
    cvss_scores = np.array([cve.cvss_score for cve in cve_records])
    
    for i in range(n_features):
        similarities = similarity_matrix[i]
        
        # Apply threshold mask
        mask = similarities > threshold
        
        # Compute weighted sum: sim * cvss
        weights[i] = np.sum(similarities[mask] * cvss_scores[mask])
    
    # Normalize weights to [0, 1]
    if weights.max() > 0:
        weights = weights / weights.max()
    
    return weights


# =============================================================================
# Step 7 — Device Risk Score
# =============================================================================

def compute_device_risk(
    feature_weights: np.ndarray,
    feature_categories: List[str]
) -> Dict[str, float]:
    """
    Compute overall device vulnerability score and category risks.
    
    Args:
        feature_weights: Normalized feature weight array
        feature_categories: Category for each feature
    
    Returns:
        Dictionary with risk scores
    """
    result = {
        "device_risk": 0.0,
        "hardware_risk": 0.0,
        "software_risk": 0.0,
        "protocol_risk": 0.0,
        "functional_risk": 0.0
    }
    
    if len(feature_weights) == 0:
        return result
    
    # Overall device risk = average of all feature weights
    result["device_risk"] = float(np.mean(feature_weights))
    
    # Category mapping
    category_map = {
        "hardware": "hardware_risk",
        "software": "software_risk",
        "protocol": "protocol_risk",
        "functional": "functional_risk"
    }
    
    # Compute category risks
    for category, risk_key in category_map.items():
        category_indices = [i for i, cat in enumerate(feature_categories) if cat == category]
        if category_indices:
            category_weights = feature_weights[category_indices]
            result[risk_key] = float(np.mean(category_weights))
    
    return result


# =============================================================================
# Step 8 — Generate Vulnerability Report
# =============================================================================

def generate_vulnerability_report(
    device: DeviceFeatures,
    feature_texts: List[str],
    feature_categories: List[str],
    feature_weights: np.ndarray,
    relevant_cves: List[List[Tuple[int, float]]],
    cve_records: List[CVERecord],
    risk_scores: Dict[str, float],
    high_risk_threshold: float = 0.3
) -> Dict:
    """
    Generate structured vulnerability report JSON.
    
    Args:
        device: Device features
        feature_texts: List of feature text descriptions
        feature_categories: Category for each feature
        feature_weights: Normalized feature weights
        relevant_cves: Relevant CVEs for each feature
        cve_records: All CVE records
        risk_scores: Device and category risk scores
        high_risk_threshold: Threshold for high-risk features
    
    Returns:
        Structured vulnerability report dictionary
    """
    # Build high-risk features list
    high_risk_features = []
    
    for i, (feature_text, category, weight) in enumerate(
        zip(feature_texts, feature_categories, feature_weights)
    ):
        if weight >= high_risk_threshold:
            related = []
            for cve_idx, similarity in relevant_cves[i]:
                cve = cve_records[cve_idx]
                related.append({
                    "cve_id": cve.cve_id,
                    "cvss": cve.cvss_score,
                    "similarity": round(similarity, 4),
                    "description": cve.description[:200] + "..." if len(cve.description) > 200 else cve.description,
                    "problem_type": cve.problem_type
                })
            
            high_risk_features.append({
                "feature": feature_text,
                "category": category,
                "risk_weight": round(float(weight), 4),
                "related_cves": related
            })
    
    # Sort by risk weight descending
    high_risk_features.sort(key=lambda x: x["risk_weight"], reverse=True)
    
    report = {
        "device": {
            "vendor": device.vendor or "",
            "model": device.model or ""
        },
        "risk_score": round(risk_scores["device_risk"], 4),
        "category_risks": {
            "hardware": round(risk_scores["hardware_risk"], 4),
            "software": round(risk_scores["software_risk"], 4),
            "protocol": round(risk_scores["protocol_risk"], 4),
            "functional": round(risk_scores["functional_risk"], 4)
        },
        "total_features_analyzed": len(feature_texts),
        "high_risk_features": high_risk_features,
        "summary": {
            "total_cves_matched": sum(len(cves) for cves in relevant_cves),
            "high_risk_feature_count": len(high_risk_features)
        }
    }
    
    return report


# =============================================================================
# Step 9 — Device Similarity (Advanced)
# =============================================================================

def compute_device_similarity(
    devices: List[DeviceFeatures],
    feature_weights_list: List[Dict[str, float]]
) -> np.ndarray:
    """
    Compute device similarity using weighted Jaccard similarity.
    
    similarity(p, q) = Σ min(w_p, w_q) / Σ max(w_p, w_q)
    
    Args:
        devices: List of device features
        feature_weights_list: List of {feature: weight} dictionaries per device
    
    Returns:
        Device similarity matrix (n_devices, n_devices)
    """
    n_devices = len(devices)
    similarity_matrix = np.zeros((n_devices, n_devices))
    
    for i in range(n_devices):
        for j in range(n_devices):
            if i == j:
                similarity_matrix[i][j] = 1.0
            elif i < j:
                weights_i = feature_weights_list[i]
                weights_j = feature_weights_list[j]
                
                # Get all features from both devices
                all_features = set(weights_i.keys()) | set(weights_j.keys())
                
                if not all_features:
                    continue
                
                # Compute weighted Jaccard
                min_sum = 0.0
                max_sum = 0.0
                
                for feature in all_features:
                    w_i = weights_i.get(feature, 0.0)
                    w_j = weights_j.get(feature, 0.0)
                    min_sum += min(w_i, w_j)
                    max_sum += max(w_i, w_j)
                
                if max_sum > 0:
                    sim = min_sum / max_sum
                else:
                    sim = 0.0
                
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
    
    return similarity_matrix


# =============================================================================
# Main Pipeline Function
# =============================================================================

def run_vulnerability_assessment(
    device_features: Dict,
    cve_file_path: str,
    similarity_threshold: float = 0.35,
    top_k_cves: int = 5,
    high_risk_threshold: float = 0.3,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict:
    """
    Run the complete vulnerability assessment pipeline.
    
    Args:
        device_features: Device feature dictionary
        cve_file_path: Path to CVE JSON file
        similarity_threshold: Minimum similarity for CVE relevance
        top_k_cves: Maximum CVEs per feature
        high_risk_threshold: Threshold for high-risk classification
        model_name: Embedding model name
    
    Returns:
        Vulnerability assessment report
    """
    print("=" * 60)
    print("VULNERABILITY ASSESSMENT PIPELINE")
    print("=" * 60)
    
    # Step 1: Parse CVE dataset
    print("\n[Step 1] Parsing CVE dataset...")
    cve_records = parse_cve_dataset(cve_file_path)
    
    if not cve_records:
        raise ValueError("No valid CVE records found")
    
    # Step 2: Construct feature text
    print("\n[Step 2] Constructing feature text...")
    device = load_device_features(device_features)
    feature_texts, feature_categories = construct_feature_text(device)
    
    print(f"  Device: {device.vendor} {device.model}")
    print(f"  Total features: {len(feature_texts)}")
    for cat in ["hardware", "software", "protocol", "functional"]:
        count = feature_categories.count(cat)
        print(f"    - {cat}: {count}")
    
    if not feature_texts:
        raise ValueError("No device features to analyze")
    
    # Step 3: Generate embeddings
    print("\n[Step 3] Generating embeddings...")
    model = load_embedding_model(model_name)
    
    feature_embeddings = generate_embeddings(model, feature_texts)
    print(f"  Feature embeddings shape: {feature_embeddings.shape}")
    
    cve_descriptions = [cve.description for cve in cve_records]
    cve_embeddings = generate_embeddings(model, cve_descriptions)
    print(f"  CVE embeddings shape: {cve_embeddings.shape}")
    
    # Step 4: Compute similarity
    print("\n[Step 4] Computing similarity matrix...")
    similarity_matrix = compute_similarity(feature_embeddings, cve_embeddings)
    print(f"  Similarity matrix shape: {similarity_matrix.shape}")
    
    # Step 5: Filter relevant CVEs
    print("\n[Step 5] Filtering relevant CVEs...")
    relevant_cves = filter_relevant_cves(
        similarity_matrix, cve_records,
        threshold=similarity_threshold,
        top_k=top_k_cves
    )
    total_matches = sum(len(cves) for cves in relevant_cves)
    print(f"  Total CVE matches: {total_matches}")
    
    # Step 6: Compute feature weights
    print("\n[Step 6] Computing feature vulnerability weights...")
    feature_weights = compute_feature_weights(
        similarity_matrix, cve_records,
        threshold=similarity_threshold
    )
    print(f"  Weight range: [{feature_weights.min():.4f}, {feature_weights.max():.4f}]")
    
    # Step 7: Compute device risk score
    print("\n[Step 7] Computing device risk score...")
    risk_scores = compute_device_risk(feature_weights, feature_categories)
    print(f"  Device Risk: {risk_scores['device_risk']:.4f}")
    print(f"  Hardware Risk: {risk_scores['hardware_risk']:.4f}")
    print(f"  Software Risk: {risk_scores['software_risk']:.4f}")
    print(f"  Protocol Risk: {risk_scores['protocol_risk']:.4f}")
    print(f"  Functional Risk: {risk_scores['functional_risk']:.4f}")
    
    # Step 8: Generate vulnerability report
    print("\n[Step 8] Generating vulnerability report...")
    report = generate_vulnerability_report(
        device=device,
        feature_texts=feature_texts,
        feature_categories=feature_categories,
        feature_weights=feature_weights,
        relevant_cves=relevant_cves,
        cve_records=cve_records,
        risk_scores=risk_scores,
        high_risk_threshold=high_risk_threshold
    )
    
    print(f"  High-risk features: {len(report['high_risk_features'])}")
    
    print("\n" + "=" * 60)
    print("ASSESSMENT COMPLETE")
    print("=" * 60)
    
    return report


def run_multi_device_assessment(
    devices: List[Dict],
    cve_file_path: str,
    **kwargs
) -> Tuple[List[Dict], np.ndarray]:
    """
    Run vulnerability assessment for multiple devices and compute similarity.
    
    Args:
        devices: List of device feature dictionaries
        cve_file_path: Path to CVE JSON file
        **kwargs: Additional arguments for run_vulnerability_assessment
    
    Returns:
        Tuple of (list of reports, device similarity matrix)
    """
    reports = []
    feature_weights_list = []
    
    for i, device_features in enumerate(devices):
        print(f"\n{'#' * 60}")
        print(f"# DEVICE {i + 1}: {device_features.get('vendor', 'Unknown')} {device_features.get('model', 'Unknown')}")
        print(f"{'#' * 60}")
        
        report = run_vulnerability_assessment(device_features, cve_file_path, **kwargs)
        reports.append(report)
        
        # Build feature weight dictionary for this device
        weights_dict = {}
        device = load_device_features(device_features)
        feature_texts, _ = construct_feature_text(device)
        
        # Re-run to get weights (or extract from internal state)
        # For simplicity, we extract from high-risk features
        for hrf in report["high_risk_features"]:
            weights_dict[hrf["feature"]] = hrf["risk_weight"]
        
        feature_weights_list.append(weights_dict)
    
    # Step 9: Compute device similarity
    print(f"\n{'=' * 60}")
    print("COMPUTING DEVICE SIMILARITY MATRIX")
    print(f"{'=' * 60}")
    
    device_objs = [load_device_features(d) for d in devices]
    similarity_matrix = compute_device_similarity(device_objs, feature_weights_list)
    
    print("\nDevice Similarity Matrix:")
    print(similarity_matrix)
    
    return reports, similarity_matrix


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example device features
    example_device = {
        "vendor": "Intel",
        "model": "Atom C3000",
        "hardware_components": ["ARM Cortex CPU", "Branch Predictor"],
        "software_components": ["Embedded Linux", "Firmware v2.1"],
        "communication_protocols": ["IEC 61850", "Modbus TCP"],
        "functional_capabilities": ["Protection", "Monitoring"]
    }
    
    # Example: Creating mock CVE data for testing
    example_cves = [
        {
            "cveMetadata": {"cveId": "CVE-2023-0001"},
            "containers": {
                "cna": {
                    "descriptions": [{"lang": "en", "value": "Buffer overflow in ARM processor allows remote code execution"}],
                    "affected": [{"product": "ARM Cortex"}],
                    "problemTypes": [{"descriptions": [{"description": "CWE-120: Buffer Overflow"}]}],
                    "metrics": [{"cvssV3_1": {"baseScore": 9.8}}]
                }
            }
        },
        {
            "cveMetadata": {"cveId": "CVE-2023-0002"},
            "containers": {
                "cna": {
                    "descriptions": [{"lang": "en", "value": "Modbus TCP protocol implementation vulnerability allows unauthorized access"}],
                    "affected": [{"product": "Industrial Controller"}],
                    "problemTypes": [{"descriptions": [{"description": "CWE-287: Improper Authentication"}]}],
                    "metrics": [{"cvssV3_1": {"baseScore": 7.5}}]
                }
            }
        },
        {
            "cveMetadata": {"cveId": "CVE-2023-0003"},
            "containers": {
                "cna": {
                    "descriptions": [{"lang": "en", "value": "Linux kernel vulnerability in embedded systems allows privilege escalation"}],
                    "affected": [{"product": "Embedded Linux"}],
                    "problemTypes": [{"descriptions": [{"description": "CWE-269: Improper Privilege Management"}]}],
                    "metrics": [{"cvssV3_1": {"baseScore": 8.1}}]
                }
            }
        }
    ]
    
    # Save example CVE data to file for testing
    with open("test_cves.json", "w") as f:
        json.dump(example_cves, f, indent=2)
    
    # Run assessment
    report = run_vulnerability_assessment(example_device, "test_cves.json")
    print(json.dumps(report, indent=2))
    
    print("\nVulnerability Assessment Module loaded successfully!")
    print("Use run_vulnerability_assessment() to analyze a device.")
    print("Use run_multi_device_assessment() for multiple devices with similarity computation.")
