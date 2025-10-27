#!/usr/bin/env python3
"""
Classify whether entries in names.csv are real personal names or misparsed metadata
using ChatGPT 5 Nano via LangChain.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI


DEFAULT_MODEL_NAME = "gpt-5-nano"
DEFAULT_INPUT_PATH = Path("names.csv")
DEFAULT_OUTPUT_PATH = Path("name_classification_results.csv")


def load_names(csv_path: Path) -> List[Dict[str, str]]:
    """Read first/last names from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"first", "last"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Expected columns {sorted(required_columns)}, got {reader.fieldnames}"
            )
        rows: List[Dict[str, str]] = []
        for row in reader:
            first = (row.get("first") or "").strip()
            last = (row.get("last") or "").strip()
            full_name = " ".join(part for part in (first, last) if part)
            if not full_name:
                # Skip empty rows but keep a placeholder for reference.
                continue
            rows.append({"first": first, "last": last, "full_name": full_name})
    return rows


def build_chain(model_name: str) -> Tuple[ChatOpenAI, ChatPromptTemplate]:
    """Create the LLM client and prompt template."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is required to call the OpenAI API."
        )

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an analytic assistant. Decide if a string is a real personal "
                "name belonging to an individual human. Only consider whether the text "
                "looks like a genuine given-name plus family-name (or similar) and not "
                "something that is clearly metadata such as an institutional unit, "
                "country, keyword list, or an artifact of parsing.",
            ),
            (
                "human",
                "Evaluate the following string:\n"
                "\"{full_name}\"\n\n"
                "Answer strictly in valid JSON with keys:\n"
                "- \"label\": one of \"REAL_NAME\" or \"NOT_A_NAME\".\n"
                "- \"confidence\": number between 0 and 1 summarising certainty.\n"
                "- \"rationale\": short explanation (max 2 sentences).\n",
            ),
        ]
    )
    return llm, prompt


def classify_name(
    llm: ChatOpenAI,
    prompt: ChatPromptTemplate,
    full_name: str,
) -> Dict[str, str]:
    """Send a single name through the classification chain and parse the response."""
    response = (prompt | llm).invoke({"full_name": full_name})
    content = getattr(response, "content", None)
    if not content:
        raise RuntimeError("Received empty response from the model.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON content: {content}") from exc

    label = data.get("label")
    if label not in {"REAL_NAME", "NOT_A_NAME"}:
        raise ValueError(f"Unexpected label in response: {data}")

    confidence = data.get("confidence")
    rationale = data.get("rationale", "")

    return {
        "label": label,
        "confidence": str(confidence) if confidence is not None else "",
        "rationale": rationale,
        "raw_response": content,
    }


def write_results(
    output_path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str]
) -> None:
    """Write classification results to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify candidate names using ChatGPT 5 Nano via LangChain."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to input CSV with 'first' and 'last' columns (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for the output CSV (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL_NAME),
        help="Override the OpenAI model name (default: gpt-5-nano or $OPENAI_MODEL).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker threads to use (default: number of CPU cores).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = load_names(args.input)
    if not names:
        raise ValueError("No valid names found to classify.")

    llm, prompt = build_chain(args.model)

    results: List[Dict[str, str]] = []

    max_workers = args.workers

    def process_record(index: int, record: Dict[str, str]) -> Tuple[int, Dict[str, str]]:
        """Process a single record and return its index with results."""
        full_name = record["full_name"]
        classification = classify_name(llm, prompt, full_name)
        result_row = {
            "index": index,
            "first": record["first"],
            "last": record["last"],
            "full_name": full_name,
            "label": classification["label"],
            "confidence": classification["confidence"],
            "rationale": classification["rationale"],
            "raw_response": classification["raw_response"],
        }
        print(f"[{index}/{len(names)}] {full_name} -> {classification['label']}")
        return index - 1, result_row

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_record, idx, record): idx 
            for idx, record in enumerate(names, start=1)
        }
        
        for future in as_completed(futures):
            try:
                list_index, result_row = future.result()
                results[list_index] = result_row
            except Exception as exc:
                original_idx = futures[future]
                print(f"Error processing record {original_idx}: {exc}")
                raise

    output_fields = [
        "index",
        "first",
        "last",
        "full_name",
        "label",
        "confidence",
        "rationale",
        "raw_response",
    ]
    write_results(args.output, results, output_fields)
    print(f"Classification complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
