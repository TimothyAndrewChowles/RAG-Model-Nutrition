#!/usr/bin/env python3
"""
Simple retrieval-augmented generation (RAG) helper for campus dining menus.

The script reads NetNutrition Excel exports, embeds each item with a Hugging Face
SentenceTransformer model, and answers natural-language questions by retrieving
the most relevant menu entries.  It includes a tiny rule-based handler for the
sample question “What is a high protein food?” that surfaces the top-protein
options taken directly from the dataset.

Requirements
------------
* pandas
* numpy
* sentence-transformers  (``pip install sentence-transformers``)

Example
-------
python Code/model.py \\
    --menu-dir "Dining Food Info" \\
    --question "What is a high protein food?"
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - surfaces clearer guidance.
    raise ImportError(
        "sentence-transformers is required. Install with `pip install sentence-transformers`."
    ) from exc


NUMERIC_KEYS = [
    "KCAL_Value",
    "TotalFat_Gram",
    "TotalCarb_Gram",
    "Protein_Gram",
    "FiberTotalDietary_Gram",
    "SugarTotal_Gram",
    "Sodium_Milligram",
]


@dataclass
class Document:
    text: str
    metadata: Dict[str, str]
    nutrients: Dict[str, float]


def discover_menu_files(explicit_files: Sequence[Path], menu_dir: Optional[Path]) -> List[Path]:
    files = list(explicit_files)

    if menu_dir:
        files.extend(sorted(menu_dir.glob("*.xlsx")))

    # Remove duplicates while keeping order.
    seen = set()
    unique_files: List[Path] = []
    for file_path in files:
        fp = file_path.resolve()
        if fp not in seen:
            seen.add(fp)
            unique_files.append(fp)

    if not unique_files:
        raise FileNotFoundError("No Excel files found. Provide --menu-file or --menu-dir.")

    missing = [str(path) for path in unique_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Menu files not found: {missing}")

    return unique_files


def load_menus(menu_files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in menu_files:
        df = pd.read_excel(path)
        df = df.rename(columns=lambda c: c.strip().replace(" ", "_"))
        df["LabelDate"] = pd.to_datetime(df["LabelDate"]).dt.date
        df["SourceFile"] = path.name
        frames.append(df)

    if not frames:
        raise ValueError("No menu data loaded.")

    combined = pd.concat(frames, ignore_index=True)

    combined["Allergens"] = combined["Allergens"].fillna("").replace({"nan": "", "NaN": ""})
    combined["Allergens"] = combined["Allergens"].apply(lambda x: x if x else "none listed")

    for key in NUMERIC_KEYS + ["ServingGramWgt"]:
        if key in combined.columns:
            combined[key] = pd.to_numeric(combined[key], errors="coerce").fillna(0.0)

    return combined


def make_documents(menu_df: pd.DataFrame) -> List[Document]:
    docs: List[Document] = []

    for _, row in menu_df.iterrows():
        formal_name = row.get("FormalName")
        if not isinstance(formal_name, str) or not formal_name.strip():
            continue

        kcal = float(row.get("KCAL_Value", 0.0) or 0.0)
        if kcal <= 0:
            # Skip records without nutritional data (e.g., condiment placeholders).
            continue

        date = row.get("LabelDate")
        meal = row.get("Meal", "Unknown meal")
        course = row.get("ServiceCourse", "Uncategorized")
        station = row.get("SourceFile", "")
        protein = row.get("Protein_Gram", 0.0) or 0.0
        carbs = row.get("TotalCarb_Gram", 0.0) or 0.0
        fat = row.get("TotalFat_Gram", 0.0) or 0.0
        ingredients = str(row.get("Ingredients", "") or "").strip()
        allergens = str(row.get("Allergens", "none listed"))

        text = (
            f"{formal_name} (meal: {meal}, course: {course}, file: {station}) "
            f"served on {date} provides {protein:.1f} g protein, {carbs:.1f} g carbs, "
            f"{fat:.1f} g fat, {kcal:.0f} kcal per {row.get('ServingGramWgt', 0.0):.0f} g serving. "
            f"Ingredients: {ingredients if ingredients else 'not listed'}. "
            f"Allergens: {allergens}."
        )

        metadata = {
            "name": formal_name,
            "date": str(date),
            "meal": str(meal),
            "course": str(course),
            "source": str(station),
        }
        nutrients = {key: float(row.get(key, 0.0) or 0.0) for key in NUMERIC_KEYS}

        docs.append(Document(text=text, metadata=metadata, nutrients=nutrients))

    if not docs:
        raise ValueError("No valid menu entries found to embed.")

    return docs


class MenuRAG:
    """Simple RAG helper that wraps embedding, retrieval, and question answering."""

    def __init__(
        self,
        menu_df: pd.DataFrame,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
    ) -> None:
        self.menu_df = menu_df
        self.documents = make_documents(menu_df)
        self.model = SentenceTransformer(model_name)

        self.normalize = normalize
        self.embeddings = self.model.encode(
            [doc.text for doc in self.documents],
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=self.normalize, show_progress_bar=False
        )[0]
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[idx], float(scores[idx])) for idx in top_indices]

    def answer_question(self, question: str, top_k: int = 5) -> str:
        special = self._handle_special_cases(question)
        if special:
            return special

        hits = self.search(question, top_k=top_k)
        if not hits:
            return "I could not find any matching items in the dataset."

        lines = ["Here are some menu items that match your question:"]
        for doc, score in hits:
            name = doc.metadata["name"]
            meal = doc.metadata["meal"]
            date = doc.metadata["date"]
            protein = doc.nutrients["Protein_Gram"]
            carb = doc.nutrients["TotalCarb_Gram"]
            fat = doc.nutrients["TotalFat_Gram"]
            kcal = doc.nutrients["KCAL_Value"]
            lines.append(
                f"- {name} (served {date} at {meal}):\n"
                f"  Protein: {protein:.1f} g\n"
                f"  Carbs: {carb:.1f} g\n"
                f"  Fat: {fat:.1f} g\n"
                f"  Calories: {kcal:.0f} kcal\n"
                f"  Allergens: {doc.metadata.get('allergens', 'see ingredients')}.\n"
            )
        return "\n".join(lines)

    def _handle_special_cases(self, question: str) -> Optional[str]:
        lowered = question.lower()
        if "high protein" in lowered:
            return self._high_protein_response(top_n=5)
        if "high-protein" in lowered:
            return self._high_protein_response(top_n=5)
        return None

    def _high_protein_response(self, top_n: int = 5) -> str:
        df = self.menu_df[self.menu_df["Protein_Gram"] > 0].copy()
        if df.empty:
            return "Protein information is not available in the dataset."

        df = df.sort_values("Protein_Gram", ascending=False)
        top_rows = df.head(top_n)

        lines = ["Highest protein items in the available menus:"]
        for _, row in top_rows.iterrows():
            lines.append(
                f"- {row['FormalName']} ({row['Meal']} on {row['LabelDate']}): "
                f"{row['Protein_Gram']:.1f} g protein, {row['KCAL_Value']:.0f} kcal per serving."
            )
        return "\n".join(lines)


def build_default_menu_dir() -> Optional[Path]:
    default_dir = Path("Dining Food Info")
    return default_dir if default_dir.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query dining menus with a simple Hugging Face RAG helper.")
    parser.add_argument(
        "--menu-file",
        dest="menu_files",
        action="append",
        type=Path,
        help="Explicit path to an Excel export (can be repeated).",
    )
    parser.add_argument(
        "--menu-dir",
        type=Path,
        default=build_default_menu_dir(),
        help="Directory containing Excel exports (defaults to 'Dining Food Info' if present).",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask the assistant (e.g., 'What is a high protein food?').",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval hits to surface for general questions (default: 5).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face sentence-transformers model to use for embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    explicit_files = [path for path in (args.menu_files or [])]
    menu_files = discover_menu_files(explicit_files, args.menu_dir)
    menu_df = load_menus(menu_files)

    rag = MenuRAG(menu_df=menu_df, model_name=args.model)

    if args.question:
        print(rag.answer_question(args.question, top_k=args.top_k))
    else:
        print(f"Loaded {len(menu_files)} menu files containing {len(rag.documents)} items.")
        print("Pass --question 'your question' to query the data.")


if __name__ == "__main__":
    main()
