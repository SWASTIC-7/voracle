"""
Device Feature Extraction from PDF Datasheets
==============================================
Uses an LLM to extract structured device features (vendor, model,
hardware/software components, protocols, capabilities) from PDF documents.
"""

import json
import os
import re
import gc
from typing import Dict, List, Optional

import torch
import pdfplumber
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# =============================================================================
# PDF Text Extraction
# =============================================================================

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract full text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Concatenated text of all pages.
    """
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    print(f"Extracted {len(full_text)} characters from {file_path}")
    return full_text


# =============================================================================
# LLM Loading
# =============================================================================

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


# =============================================================================
# Chunking
# =============================================================================

def chunk_text_by_tokens(
    text: str, tokenizer, max_tokens: int = 1200
) -> List[str]:
    """Split *text* into token-limited chunks."""
    token_ids = tokenizer(text)["input_ids"]
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids))
    return chunks


# =============================================================================
# Per-chunk LLM Extraction
# =============================================================================

_EXTRACTION_PROMPT = """\
Extract device information.

Respond with ONLY a valid JSON object.
If information is not present in this document chunk, return null or empty list.
Do not infer or assume missing information.

Do NOT repeat the instructions.
Do NOT include explanations.
Do NOT include text outside JSON.

JSON format:
{{
  "vendor": string or null,
  "model": string or null,
  "hardware_components": list,
  "software_components": list,
  "communication_protocols": list,
  "functional_capabilities": list
}}

Document:
{chunk}
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


def _run_extraction_on_chunk(
    chunk: str, model, tokenizer
) -> Optional[Dict]:
    """Run the LLM on a single chunk and return parsed JSON."""
    prompt = _EXTRACTION_PROMPT.format(chunk=chunk)

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return _extract_valid_json(result)


# =============================================================================
# Merge Chunk Results
# =============================================================================

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

    # Convert sets to sorted lists, None to empty string
    for key in merged:
        if isinstance(merged[key], set):
            merged[key] = sorted(merged[key])
    merged["vendor"] = merged["vendor"] or ""
    merged["model"] = merged["model"] or ""

    return merged


# =============================================================================
# Public API
# =============================================================================

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


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "relay.pdf"
    features = extract_device_features(pdf)
    print(json.dumps(features, indent=2))
