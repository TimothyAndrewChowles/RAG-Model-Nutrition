#!/usr/bin/env python3
"""
Station meal-planning assistant for NetNutrition exports.

This script builds a simple AI-style planning model that selects combinations of
menu items for each meal in a date range so that the resulting calories and
macros land close to configurable targets.  It relies exclusively on the
provided NetNutrition XLSX files – no hard-coded foods.

Example
-------
python code.py \
    --menu-file "../Dining Food Info/Station 9 10.20.25-10.26.25.xlsx" \
    --start-date 2025-10-27 \
    --end-date 2025-11-02 \
    --daily-calories 1600 \
    --macro-split 50 20 30 \
    --station-name "Station 9"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# Nutrient fields we care about for planning/summary
NUTRIENT_KEYS = [
    "KCAL_Value",
    "TotalFat_Gram",
    "TotalCarb_Gram",
    "Protein_Gram",
    "FiberTotalDietary_Gram",
    "SugarTotal_Gram",
    "Sodium_Milligram",
]

# Default meal split across the day (breakfast, lunch, dinner).
DEFAULT_MEAL_SPLIT = {
    "Breakfast": 0.3,
    "Lunch": 0.35,
    "Dinner": 0.35,
}


@dataclass(frozen=True)
class MenuItem:
    """Structured view of a single NetNutrition menu item."""

    name: str
    service_course: str
    serving_g: float
    nutrients: Dict[str, float]
    ingredients: Optional[str]
    allergens: Optional[str]

    def scaled_nutrients(self, servings: int) -> Dict[str, float]:
        return {k: v * servings for k, v in self.nutrients.items()}


def load_menu(path: Path) -> pd.DataFrame:
    """Load a NetNutrition export into a normalized DataFrame."""
    df = pd.read_excel(path)
    df = df.rename(columns=lambda c: c.strip().replace(" ", "_"))
    df["LabelDate"] = pd.to_datetime(df["LabelDate"]).dt.date

    # Normalize allergens to basic strings.
    df["Allergens"] = df["Allergens"].fillna("").replace({"nan": "", "NaN": ""})
    df["Allergens"] = df["Allergens"].apply(lambda val: val if val else "none listed")

    # Fill NaNs in key numeric fields with zeros to simplify math.
    for key in NUTRIENT_KEYS:
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce").fillna(0.0)

    df["ServingGramWgt"] = pd.to_numeric(df.get("ServingGramWgt"), errors="coerce").fillna(0.0)
    return df


def compute_macro_targets(meal_targets: Dict[str, float], macro_split: Tuple[int, int, int]) -> Dict[str, Dict[str, float]]:
    """
    Convert daily calorie split and macro percentages into gram targets per meal.

    Parameters
    ----------
    meal_targets : dict
        Calories allocated to each meal (e.g., {"Breakfast": 480, ...}).
    macro_split : tuple
        Percent distribution (carb_pct, protein_pct, fat_pct) that sums to 100.
    """
    carb_pct, protein_pct, fat_pct = (pct / 100.0 for pct in macro_split)
    results: Dict[str, Dict[str, float]] = {}

    for meal, meal_kcal in meal_targets.items():
        results[meal] = {
            "KCAL_Value": meal_kcal,
            "TotalCarb_Gram": meal_kcal * carb_pct / 4.0,
            "Protein_Gram": meal_kcal * protein_pct / 4.0,
            "TotalFat_Gram": meal_kcal * fat_pct / 9.0,
        }
    return results


class MealPlannerModel:
    """
    Lightweight heuristic planner that selects a combination of menu items
    close to target calories/macros using greedy search with minor backtracking.
    """

    def __init__(
        self,
        menu: pd.DataFrame,
        daily_calories: int,
        meal_split: Dict[str, float],
        macro_split: Tuple[int, int, int],
        max_servings: int = 2,
        max_items: int = 6,
    ) -> None:
        self.menu = menu
        self.daily_calories = daily_calories
        self.meal_split = meal_split
        self.macro_split = macro_split
        self.max_servings = max_servings
        self.max_items = max_items

        self.meal_calorie_targets = {
            meal: daily_calories * frac for meal, frac in meal_split.items()
        }
        self.macro_targets = compute_macro_targets(self.meal_calorie_targets, macro_split)

    def generate_plan(self, start_date, end_date) -> Dict:
        days = pd.date_range(start=start_date, end=end_date, freq="D").date
        plan = {}

        for day in days:
            day_plan = {}
            day_totals = {key: 0.0 for key in NUTRIENT_KEYS}

            day_df = self.menu[self.menu["LabelDate"] == day]
            if day_df.empty:
                plan[str(day)] = {"meals": day_plan, "daily_totals": day_totals}
                continue

            for meal_name in sorted(day_df["Meal"].unique()):
                if meal_name not in self.meal_split:
                    continue
                meal_df = day_df[day_df["Meal"] == meal_name]
                result = self._plan_single_meal(meal_df, meal_name)
                day_plan[meal_name] = result

                # Aggregate for daily totals.
                for key in NUTRIENT_KEYS:
                    day_totals[key] += result["totals"].get(key, 0.0)

            plan[str(day)] = {"meals": day_plan, "daily_totals": day_totals}
        return plan

    def _plan_single_meal(self, meal_df: pd.DataFrame, meal_name: str) -> Dict:
        items = self._extract_items(meal_df)
        if not items:
            return {"items": [], "totals": {key: 0.0 for key in NUTRIENT_KEYS}}

        target = self.macro_targets.get(
            meal_name,
            {"KCAL_Value": self.daily_calories * (1 / 3), "TotalCarb_Gram": 0, "Protein_Gram": 0, "TotalFat_Gram": 0},
        )

        # Start from each entree (or every item if there are no entrees) and greedily improve.
        entree_candidates = [item for item in items if item.service_course == "Entrees"]
        base_candidates = entree_candidates or items

        best_state = None
        best_score = float("inf")

        for seed in base_candidates:
            state = {seed.name: 1}
            totals = seed.scaled_nutrients(1)
            score = self._score(totals, target)

            improved = True
            while improved and sum(state.values()) < self.max_items:
                improved = False
                best_local = score
                best_choice = None

                for item in items:
                    if sum(state.values()) >= self.max_items and state.get(item.name, 0) == 0:
                        continue

                    if state.get(item.name, 0) >= self.max_servings:
                        continue

                    new_totals = {k: totals.get(k, 0.0) + item.nutrients.get(k, 0.0) for k in totals.keys()}
                    new_score = self._score(new_totals, target)

                    if new_score + 1e-6 < best_local:
                        best_local = new_score
                        best_choice = item
                        best_totals = new_totals

                if best_choice:
                    state[best_choice.name] = state.get(best_choice.name, 0) + 1
                    totals = best_totals  # type: ignore[name-defined]
                    score = best_local
                    improved = True

            # Try pruning a single item if it helps.
            state, totals, score = self._prune_once(state, items, totals, target, score)

            if score < best_score:
                best_score = score
                best_state = (state, totals)

        if not best_state:
            best_state = ({base_candidates[0].name: 1}, base_candidates[0].scaled_nutrients(1))

        selection, totals = best_state

        return {
            "items": [
                {
                    "name": name,
                    "servings": count,
                    "nutrients": self._scaled_nutrients_lookup(items, name, count),
                    "ingredients": self._ingredients_lookup(items, name),
                    "allergens": self._allergens_lookup(items, name),
                }
                for name, count in sorted(selection.items())
            ],
            "totals": {key: totals.get(key, 0.0) for key in NUTRIENT_KEYS},
            "target": target,
            "score": best_score,
        }

    @staticmethod
    def _ingredients_lookup(items: List[MenuItem], name: str) -> Optional[str]:
        for item in items:
            if item.name == name:
                return item.ingredients
        return None

    @staticmethod
    def _allergens_lookup(items: List[MenuItem], name: str) -> Optional[str]:
        for item in items:
            if item.name == name:
                return item.allergens
        return None

    @staticmethod
    def _scaled_nutrients_lookup(items: List[MenuItem], name: str, count: int) -> Dict[str, float]:
        for item in items:
            if item.name == name:
                return item.scaled_nutrients(count)
        return {key: 0.0 for key in NUTRIENT_KEYS}

    def _prune_once(
        self,
        state: Dict[str, int],
        items: List[MenuItem],
        totals: Dict[str, float],
        target: Dict[str, float],
        current_score: float,
    ):
        best_state = state
        best_totals = totals
        best_score = current_score

        for name, count in list(state.items()):
            if count <= 1:
                continue
            item = next((itm for itm in items if itm.name == name), None)
            if not item:
                continue

            new_totals = {k: totals[k] - item.nutrients.get(k, 0.0) for k in totals.keys()}
            new_state = state.copy()
            new_state[name] = count - 1
            if new_state[name] == 0:
                del new_state[name]

            new_score = self._score(new_totals, target)
            if new_score + 1e-6 < best_score:
                best_score = new_score
                best_state = new_state
                best_totals = new_totals

        return best_state, best_totals, best_score

    @staticmethod
    def _score(totals: Dict[str, float], target: Dict[str, float]) -> float:
        # Weighted relative error on calories + macros.
        weights = {
            "KCAL_Value": 1.0,
            "TotalCarb_Gram": 0.7,
            "Protein_Gram": 0.9,
            "TotalFat_Gram": 0.6,
        }
        error = 0.0
        for key, weight in weights.items():
            tgt = target.get(key, 0.0)
            val = totals.get(key, 0.0)
            if tgt <= 0:
                continue
            rel_err = abs(val - tgt) / tgt
            error += weight * rel_err
        return error

    def _extract_items(self, meal_df: pd.DataFrame) -> List[MenuItem]:
        items: List[MenuItem] = []

        for _, row in meal_df.iterrows():
            kcal = float(row.get("KCAL_Value", 0.0))
            # Skip entries with essentially no nutritional contribution.
            if kcal < 1:
                continue

            nutrients = {key: float(row.get(key, 0.0)) for key in NUTRIENT_KEYS}
            item = MenuItem(
                name=row["FormalName"],
                service_course=row.get("ServiceCourse", "Other"),
                serving_g=float(row.get("ServingGramWgt", 0.0)),
                nutrients=nutrients,
                ingredients=row.get("Ingredients"),
                allergens=row.get("Allergens"),
            )
            items.append(item)

        return items


def pretty_print_plan(plan: Dict, station_name: Optional[str]) -> str:
    lines: List[str] = []
    if station_name:
        lines.append(f"Station: {station_name}")

    for day, payload in plan.items():
        lines.append(f"\n{day}")
        day_totals = payload["daily_totals"]
        lines.append(
            f"  Daily totals: {round(day_totals['KCAL_Value'])} kcal | "
            f"{round(day_totals['TotalFat_Gram'])}g fat | "
            f"{round(day_totals['TotalCarb_Gram'])}g carb | "
            f"{round(day_totals['Protein_Gram'])}g protein | "
            f"{round(day_totals['FiberTotalDietary_Gram'])}g fiber | "
            f"{round(day_totals['SugarTotal_Gram'])}g sugar | "
            f"{round(day_totals['Sodium_Milligram'])}mg sodium"
        )

        for meal, details in payload["meals"].items():
            totals = details["totals"]
            lines.append(
                f"  {meal}: {round(totals['KCAL_Value'])} kcal | "
                f"{round(totals['TotalFat_Gram'])}g fat | "
                f"{round(totals['TotalCarb_Gram'])}g carb | "
                f"{round(totals['Protein_Gram'])}g protein | "
                f"{round(totals['FiberTotalDietary_Gram'])}g fiber | "
                f"{round(totals['SugarTotal_Gram'])}g sugar | "
                f"{round(totals['Sodium_Milligram'])}mg sodium "
                f"(score={details['score']:.3f})"
            )
            for item in details["items"]:
                lines.append(
                    f"    - {item['servings']}× {item['name']} "
                    f"({round(item['nutrients']['KCAL_Value'])} kcal, "
                    f"{round(item['nutrients']['TotalCarb_Gram'])}g carb, "
                    f"{round(item['nutrients']['Protein_Gram'])}g protein, "
                    f"{round(item['nutrients']['TotalFat_Gram'])}g fat)"
                )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-style meal planner for NetNutrition station exports.")
    parser.add_argument(
        "--menu-file",
        type=Path,
        required=True,
        help="Path to the station XLSX export (e.g., 'Dining Food Info/Station 9 10.20.25-10.26.25.xlsx').",
    )
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--daily-calories",
        type=int,
        default=1600,
        help="Target calories per day used to set meal goals (default: 1600).",
    )
    parser.add_argument(
        "--macro-split",
        nargs=3,
        type=int,
        default=(50, 20, 30),
        metavar=("CARB_PCT", "PROTEIN_PCT", "FAT_PCT"),
        help="Macro percentage split for carbs/protein/fat (must sum to 100).",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format for the final plan (default: text).",
    )
    parser.add_argument(
        "--station-name",
        default=None,
        help="Optional descriptive station name for pretty output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    macro_total = sum(args.macro_split)
    if macro_total != 100:
        raise ValueError("Macro split must sum to 100 (received %s)." % (args.macro_split,))

    menu_df = load_menu(args.menu_file)
    start_date = pd.to_datetime(args.start_date).date()
    end_date = pd.to_datetime(args.end_date).date()

    planner = MealPlannerModel(
        menu=menu_df,
        daily_calories=args.daily_calories,
        meal_split=DEFAULT_MEAL_SPLIT,
        macro_split=tuple(args.macro_split),  # type: ignore[arg-type]
    )
    plan = planner.generate_plan(start_date=start_date, end_date=end_date)

    if args.output == "json":
        print(json.dumps(plan, indent=2))
    else:
        print(pretty_print_plan(plan, args.station_name))


if __name__ == "__main__":
    main()
