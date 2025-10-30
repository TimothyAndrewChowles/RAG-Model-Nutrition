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
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import math

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

DEFAULT_MEAL_SPLIT = {
    "Breakfast": 0.3,
    "Lunch": 0.35,
    "Dinner": 0.35,
}


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
            "allergens": allergens,
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
        self._doc_lookup = {self._canonical_key(doc.metadata): doc for doc in self.documents}

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
        candidate_limit = min(len(self.documents), max(top_k * 4, top_k))
        candidate_indices = list(np.argsort(scores)[::-1][:candidate_limit])

        selected_indices: List[int] = []
        selected_keys = set()

        while candidate_indices and len(selected_indices) < top_k:
            best_idx = None
            best_score = -math.inf

            for idx in list(candidate_indices):
                doc = self.documents[idx]
                key = self._canonical_key(doc.metadata)
                if key in selected_keys:
                    candidate_indices.remove(idx)
                    continue

                relevance = float(scores[idx])
                if not selected_indices:
                    mmr_score = relevance
                else:
                    diversity_penalty = max(
                        self._pair_similarity(idx, picked_idx) for picked_idx in selected_indices
                    )
                    # Encourage diversity while preserving high relevance.
                    mmr_score = 0.7 * relevance - 0.3 * diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is None:
                break

            candidate_indices.remove(best_idx)
            selected_indices.append(best_idx)
            selected_keys.add(self._canonical_key(self.documents[best_idx].metadata))

        return [(self.documents[idx], float(scores[idx])) for idx in selected_indices]

    def answer_question(self, question: str, top_k: int = 5) -> str:
        special = self._handle_special_cases(question, top_k=top_k)
        if special:
            return special

        hits = self.search(question, top_k=top_k)
        if not hits:
            return "I could not find any matching items in the dataset."

        intent = self._determine_intent(question)

        if intent == "serving":
            return self._serving_response(question, hits)
        if intent == "location":
            return self._location_response(question, hits)

        headers, rows, totals, meals, stations = self._prepare_item_table(
            hits, score_label="Match confidence"
        )
        table = self._format_table(headers, rows)
        summary = self._summaries_text(len(hits), totals, meals, stations)

        lines = [f"Top matches for \"{question}\"", table]
        if summary:
            lines.append(summary)

        return "\n".join(lines)

    def _handle_special_cases(self, question: str, top_k: int) -> Optional[str]:
        lowered = question.lower()
        calorie_target = self._extract_calorie_target(lowered)
        if "generate" in lowered and "meal" in lowered and calorie_target is not None:
            return self._daily_meal_plan_response(calorie_target)
        if "plan" in lowered and "meal" in lowered and calorie_target is not None:
            return self._daily_meal_plan_response(calorie_target)
        if "high protein" in lowered or "high-protein" in lowered:
            return self._high_protein_response(top_n=top_k)
        return None

    @staticmethod
    def _determine_intent(question: str) -> str:
        lowered = question.lower()
        serving_keywords = [
            "what's served",
            "what is served",
            "served on",
            "serve on",
            "serving",
            "menu for",
            "what do they serve",
            "what is for",
            "what are they serving",
        ]
        location_keywords = [
            "what dining hall",
            "which dining hall",
            "dining hall is this",
            "what station",
            "which station",
            "station is this",
            "where is this",
            "where can i find",
        ]

        if any(keyword in lowered for keyword in serving_keywords) or (
            "serve" in lowered and "?" in lowered
        ):
            return "serving"
        if any(keyword in lowered for keyword in location_keywords):
            return "location"
        return "default"

    def _serving_response(self, question: str, hits: Sequence[Tuple[Document, Optional[float]]]) -> str:
        primary_doc = hits[0][0]
        date_str = primary_doc.metadata.get("date", "the selected date") or "the selected date"
        station = primary_doc.metadata.get("source", "").strip() or "the listed station"

        meals_to_items: Dict[str, List[str]] = {}
        stations = set()
        dates = set()

        for doc, _ in hits:
            metadata = doc.metadata
            meal_name = metadata.get("meal", "Meal").strip() or "Meal"
            item_name = metadata.get("name", "Unknown item").strip() or "Unknown item"
            kcal = doc.nutrients.get("KCAL_Value", 0.0)
            protein = doc.nutrients.get("Protein_Gram", 0.0)
            stations.add(metadata.get("source", "").strip())
            dates.add(metadata.get("date", ""))

            detail = f"{item_name} ({kcal:.0f} kcal, {protein:.1f} g protein)"
            meals_to_items.setdefault(meal_name, []).append(detail)

        lines = []
        if len(dates) == 1:
            date_phrase = next(iter(dates)) or date_str
        else:
            date_phrase = date_str

        station_list = sorted({s for s in stations if s})
        station_phrase = ", ".join(station_list) if station_list else station

        lines.append(f"Here's what is being served around {date_phrase} at {station_phrase}:")
        for meal_name in sorted(meals_to_items.keys()):
            items = ", ".join(meals_to_items[meal_name])
            lines.append(f"- {meal_name}: {items}")

        headers, rows, totals, meal_names, station_names = self._prepare_item_table(
            hits, score_label="Match confidence"
        )
        table = self._format_table(headers, rows)
        summary = self._summaries_text(len(hits), totals, meal_names, station_names)

        lines.append("")
        lines.append("Details:")
        lines.append(table)
        if summary:
            lines.append(summary)

        return "\n".join(lines)

    def _location_response(self, question: str, hits: Sequence[Tuple[Document, Optional[float]]]) -> str:
        primary_doc = hits[0][0]
        station = primary_doc.metadata.get("source", "").strip() or "an unspecified station"
        meal = primary_doc.metadata.get("meal", "Unknown meal")
        date_str = primary_doc.metadata.get("date", "the selected date") or "the selected date"

        station_set = sorted({doc.metadata.get("source", "").strip() for doc, _ in hits if doc.metadata.get("source")})
        meal_set = sorted({doc.metadata.get("meal", "").strip() for doc, _ in hits if doc.metadata.get("meal")})
        date_set = sorted({doc.metadata.get("date", "").strip() for doc, _ in hits if doc.metadata.get("date")})

        lines = [
            f"The results point to {station}.",
            f"Most examples come from the {meal} service on {date_str}." if meal else "",
        ]

        extra_bits = []
        if station_set:
            extra_bits.append(f"Stations seen: {', '.join(station_set)}")
        if meal_set:
            extra_bits.append(f"Meals covered: {', '.join(meal_set)}")
        if date_set:
            extra_bits.append(f"Dates present: {', '.join(date_set)}")

        if extra_bits:
            lines.append(" • ".join(extra_bits))

        headers, rows, totals, meals, stations = self._prepare_item_table(
            hits, score_label="Match confidence"
        )
        table = self._format_table(headers, rows)
        summary = self._summaries_text(len(hits), totals, meals, stations)

        lines.append("")
        lines.append("Details:")
        lines.append(table)
        if summary:
            lines.append(summary)

        return "\n".join(line for line in lines if line)


    @staticmethod
    def _extract_calorie_target(text: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)(\s*[kK])?\s*(?:calories|cals|kcal|kcals)?", text)
        if not match:
            return None
        number = float(match.group(1).replace(",", ""))
        suffix = match.group(2)
        if suffix and "k" in suffix.lower():
            number *= 1000
        return number

    def _daily_meal_plan_response(self, target_calories: float) -> str:
        if target_calories <= 0:
            return "Calorie target must be greater than zero."

        if "LabelDate" not in self.menu_df.columns:
            return "Menu data is missing dates, so I cannot build a daily plan."

        available_dates = sorted({d for d in self.menu_df["LabelDate"].dropna()})
        if not available_dates:
            return "No dated menu entries are available to generate a plan."

        today = date.today()
        plan_date = today if today in available_dates else min(
            available_dates, key=lambda d: abs((d - today).days)
        )
        day_df = self.menu_df[self.menu_df["LabelDate"] == plan_date]
        if day_df.empty:
            return f"I could not find menu items for {plan_date}."

        meal_targets = {meal: frac * target_calories for meal, frac in DEFAULT_MEAL_SPLIT.items()}
        meal_candidates: Dict[str, List[Tuple[Tuple[float, float, float], pd.Series]]] = {}

        for meal, meal_target in meal_targets.items():
            meal_df = day_df[day_df["Meal"] == meal]
            if meal_df.empty:
                continue

            candidates: List[Tuple[Tuple[float, float, float], pd.Series]] = []
            for _, row in meal_df.iterrows():
                kcal = float(row.get("KCAL_Value", 0.0) or 0.0)
                protein = float(row.get("Protein_Gram", 0.0) or 0.0)
                course_penalty = 0.0 if str(row.get("ServiceCourse", "")).lower() == "entrees" else 1.0
                score = (
                    abs(kcal - meal_target),
                    course_penalty,
                    -protein,
                )
                candidates.append((score, row))

            candidates.sort(key=lambda tup: tup[0])
            if candidates:
                meal_candidates[meal] = candidates

        if not meal_candidates:
            return f"No qualifying meals were found for {plan_date}."

        selection_indices: Dict[str, int] = {}
        selected_rows: Dict[str, pd.Series] = {}

        for meal, candidates in meal_candidates.items():
            selection_indices[meal] = 0
            selected_rows[meal] = candidates[0][1]

        def total_kcal() -> float:
            return sum(float(row.get("KCAL_Value", 0.0) or 0.0) for row in selected_rows.values())

        max_iterations = 20
        iterations = 0
        while total_kcal() > target_calories * 1.05 and iterations < max_iterations:
            iterations += 1
            meal_to_adjust = max(
                selected_rows.keys(),
                key=lambda m: float(selected_rows[m].get("KCAL_Value", 0.0) or 0.0),
            )
            candidates = meal_candidates.get(meal_to_adjust, [])
            current_idx = selection_indices[meal_to_adjust]

            swapped = False
            for next_idx in range(current_idx + 1, len(candidates)):
                candidate_row = candidates[next_idx][1]
                new_total = (
                    total_kcal()
                    - float(selected_rows[meal_to_adjust].get("KCAL_Value", 0.0) or 0.0)
                    + float(candidate_row.get("KCAL_Value", 0.0) or 0.0)
                )
                if new_total <= target_calories * 1.05 or new_total < total_kcal():
                    selected_rows[meal_to_adjust] = candidate_row
                    selection_indices[meal_to_adjust] = next_idx
                    swapped = True
                    break

            if not swapped:
                break

        plan_hits: List[Tuple[Document, Optional[float]]] = []
        meals: List[str] = []
        totals = {key: 0.0 for key in NUMERIC_KEYS}

        for meal in DEFAULT_MEAL_SPLIT.keys():
            row = selected_rows.get(meal)
            if row is None:
                continue

            metadata = {
                "name": str(row.get("FormalName", "")),
                "date": str(plan_date),
                "meal": meal,
                "course": str(row.get("ServiceCourse", "")),
                "source": str(row.get("SourceFile", "")),
                "allergens": str(row.get("Allergens", "see ingredients")),
            }
            key = self._canonical_key(metadata)
            doc = self._doc_lookup.get(key)
            if not doc:
                nutrients = {nut: float(row.get(nut, 0.0) or 0.0) for nut in NUMERIC_KEYS}
                doc = Document(text="", metadata=metadata, nutrients=nutrients)

            calorie_gap = meal_targets[meal] - float(doc.nutrients.get("KCAL_Value", 0.0))
            plan_hits.append((doc, calorie_gap))
            meals.append(meal)

            for nut_key in NUMERIC_KEYS:
                totals[nut_key] += float(doc.nutrients.get(nut_key, 0.0))

        if not plan_hits:
            return f"I could not build a plan for {plan_date}."

        def gap_formatter(gap: Optional[float]) -> str:
            if gap is None:
                return "—"
            if abs(gap) < 1:
                return "On target"
            if gap > 0:
                return f"{gap:.0f} kcal under"
            return f"{abs(gap):.0f} kcal over"

        headers, rows, totals, meal_names, stations = self._prepare_item_table(
            plan_hits,
            score_label="Calorie gap",
            score_formatter=gap_formatter,
        )
        table = self._format_table(headers, rows)
        summary = self._summaries_text(len(plan_hits), totals, meal_names, stations)
        total_calories = totals.get("KCAL_Value", 0.0)
        daily_gap = target_calories - total_calories
        daily_gap_text = gap_formatter(daily_gap)

        intro = f"Daily meal plan for {plan_date} targeting {target_calories:.0f} kcal"
        if plan_date != today:
            intro += f" (closest available date to today, {today})"

        lines = [
            intro,
            table,
            f"Total calories • {total_calories:.0f} kcal ({daily_gap_text})",
        ]

        if summary:
            lines.append(summary)

        return "\n".join(lines)

    def _high_protein_response(self, top_n: int = 5) -> str:
        df = self.menu_df[self.menu_df["Protein_Gram"] > 0].copy()
        if df.empty:
            return "Protein information is not available in the dataset."

        df = df.sort_values("Protein_Gram", ascending=False)
        df = df.drop_duplicates(subset=["FormalName", "LabelDate", "Meal"])
        top_rows = df.head(top_n)

        hits: List[Tuple[Document, Optional[float]]] = []

        for _, row in top_rows.iterrows():
            key = self._canonical_key(
                {
                    "name": str(row.get("FormalName", "")),
                    "date": str(row.get("LabelDate", "")),
                    "meal": str(row.get("Meal", "")),
                }
            )
            doc = self._doc_lookup.get(key)
            if not doc:
                # Fallback: build a lightweight document snapshot.
                metadata = {
                    "name": str(row.get("FormalName", "")),
                    "date": str(row.get("LabelDate", "")),
                    "meal": str(row.get("Meal", "")),
                    "course": str(row.get("ServiceCourse", "")),
                    "source": str(row.get("SourceFile", "")),
                    "allergens": str(row.get("Allergens", "see ingredients")),
                }
                nutrients = {key: float(row.get(key, 0.0) or 0.0) for key in NUMERIC_KEYS}
                doc = Document(text="", metadata=metadata, nutrients=nutrients)
            hits.append((doc, None))

        headers, rows, totals, meals, stations = self._prepare_item_table(
            hits, score_label="Protein focus", score_formatter=lambda _: "Top pick"
        )
        table = self._format_table(headers, rows)
        summary = self._summaries_text(len(hits), totals, meals, stations)

        lines = ["Highest protein items in the available menus:", table]
        if summary:
            lines.append(summary)
        return "\n".join(lines)

    @staticmethod
    def _canonical_key(metadata: Dict[str, str]) -> Tuple[str, str, str]:
        name = (metadata.get("name") or "").strip().lower()
        date = (metadata.get("date") or "").strip()
        meal = (metadata.get("meal") or "").strip().lower()
        return name, date, meal

    def _pair_similarity(self, idx_a: int, idx_b: int) -> float:
        vec_a = self.embeddings[idx_a]
        vec_b = self.embeddings[idx_b]
        if self.normalize:
            return float(np.dot(vec_a, vec_b))
        norm_a = np.linalg.norm(vec_a) or 1.0
        norm_b = np.linalg.norm(vec_b) or 1.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def _prepare_item_table(
        self,
        hits: Sequence[Tuple[Document, Optional[float]]],
        score_label: str,
        score_formatter: Optional[Callable[[Optional[float]], str]] = None,
    ) -> Tuple[List[str], List[List[str]], Dict[str, float], Set[str], Set[str]]:
        totals = {key: 0.0 for key in NUMERIC_KEYS}
        meals = set()
        stations = set()

        headers = [
            "#",
            "Item",
            "Meal",
            "Date",
            "Calories",
            "Protein (g)",
            "Carbs (g)",
            "Fat (g)",
            "Station",
            "Allergens",
        ]

        if score_label:
            headers.append(score_label)

        rows: List[List[str]] = []

        for idx, (doc, score) in enumerate(hits, start=1):
            metadata = doc.metadata
            nutrients = doc.nutrients

            name = metadata.get("name", "Unknown item").strip() or "Unknown item"
            meal = metadata.get("meal", "Unknown meal") or "Unknown meal"
            course = metadata.get("course", "Uncategorized") or "Uncategorized"
            date = metadata.get("date", "—") or "—"
            station = metadata.get("source", "—") or "—"
            allergens = metadata.get("allergens", "see ingredients") or "see ingredients"

            row_cells = [
                f"{idx}",
                self._clip(name, 40),
                f"{meal} ({course})",
                date,
                f"{nutrients.get('KCAL_Value', 0.0):.0f}",
                f"{nutrients.get('Protein_Gram', 0.0):.1f}",
                f"{nutrients.get('TotalCarb_Gram', 0.0):.1f}",
                f"{nutrients.get('TotalFat_Gram', 0.0):.1f}",
                self._clip(station, 20),
                self._clip(allergens, 32),
            ]

            if score_label:
                if score_formatter:
                    display_score = score_formatter(score)
                elif score is None:
                    display_score = "—"
                else:
                    display_score = f"{score * 100:.1f}%"
                row_cells.append(display_score)

            rows.append(row_cells)

            for key in NUMERIC_KEYS:
                totals[key] += float(nutrients.get(key, 0.0))
            meals.add(str(metadata.get("meal", "Unknown")))
            stations.add(str(metadata.get("source", "")))

        return headers, rows, totals, meals, stations

    @staticmethod
    def _clip(value: str, limit: int = 32) -> str:
        value = (value or "").strip()
        return value if len(value) <= limit else value[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
        if not rows:
            return "No items found."

        col_widths = [len(header) for header in headers]
        for row in rows:
            for col_idx, cell in enumerate(row):
                col_widths[col_idx] = max(col_widths[col_idx], len(cell))

        def border(char: str) -> str:
            return "+" + "+".join(char * (width + 2) for width in col_widths) + "+"

        def render_row(cells: Sequence[str]) -> str:
            return "|" + "|".join(f" {cell.ljust(col_widths[idx])} " for idx, cell in enumerate(cells)) + "|"

        top_border = border("-")
        header_border = border("=")
        bottom_border = border("-")
        header_row = render_row(headers)
        data_rows = [render_row(row) for row in rows]

        return "\n".join([top_border, header_row, header_border] + data_rows + [bottom_border])

    @staticmethod
    def _summaries_text(
        count: int,
        totals: Dict[str, float],
        meals: Iterable[str],
        stations: Iterable[str],
    ) -> str:
        if count <= 0:
            return ""

        protein_avg = totals.get("Protein_Gram", 0.0) / count if "Protein_Gram" in totals else 0.0
        carb_avg = totals.get("TotalCarb_Gram", 0.0) / count if "TotalCarb_Gram" in totals else 0.0
        fat_avg = totals.get("TotalFat_Gram", 0.0) / count if "TotalFat_Gram" in totals else 0.0
        kcal_avg = totals.get("KCAL_Value", 0.0) / count if "KCAL_Value" in totals else 0.0

        meal_list = sorted({m for m in meals if m})
        station_list = sorted({s for s in stations if s})

        summary_lines = [
            (
                "Average macros • "
                f"Protein {protein_avg:.1f} g | "
                f"Carbs {carb_avg:.1f} g | "
                f"Fat {fat_avg:.1f} g | "
                f"Calories {kcal_avg:.0f} kcal"
            )
        ]

        if meal_list or station_list:
            meal_str = ", ".join(meal_list) if meal_list else "—"
            station_str = ", ".join(station_list) if station_list else "—"
            summary_lines.append(f"Coverage • Meals: {meal_str} | Stations: {station_str}")

        return "\n".join(summary_lines)


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
