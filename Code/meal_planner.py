#!/usr/bin/env python3
"""
Station meal-planning assistant for NetNutrition exports.

This script builds a smart planner that balances calories, macros, allergens,
and personalization settings for every meal in a date range.  It relies on
NetNutrition XLSX exports and can cache normalized data for faster re-runs.

Example
-------
python Code/meal_planner.py \
    --menu-file "../Dining Food Info/Station 9 10.20.25-10.26.25.xlsx" \
    --start-date 2025-10-27 \
    --end-date 2025-11-02 \
    --daily-calories 2000 \
    --macro-split 45 30 25 \
    --meal-split Breakfast=30 Lunch=40 Dinner=30 \
    --exclude-allergens milk peanuts \
    --prefer-keyword veggie \
    --max-repeat-per-week 2 \
    --alternates 2 \
    --output json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import pulp  # type: ignore
except ImportError as exc:  # pragma: no cover - surfaces clearer guidance.
    raise ImportError(
        "The meal planner requires 'pulp' (https://pypi.org/project/PuLP/). "
        "Install with `pip install pulp` and retry."
    ) from exc


# Nutrient fields used for planning/summary.
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

# Columns that may contain station names depending on the export template.
STATION_COLUMNS = ("Station", "StationName", "Station_Name", "Concept", "Restaurant")

# Cache folder for normalized menus.
CACHE_ROOT = Path(__file__).resolve().parent / ".cache"


def _cache_path(src: Path) -> Path:
    CACHE_ROOT.mkdir(exist_ok=True)
    hash_key = hashlib.sha1(f"{src.resolve()}::{src.stat().st_mtime_ns}".encode("utf-8")).hexdigest()
    return CACHE_ROOT / f"{hash_key}.parquet"


def load_menu(path: Path, *, use_cache: bool = True) -> pd.DataFrame:
    """
    Load a NetNutrition export into a normalized DataFrame.

    The first call for a given file is cached as Parquet to skip repeated XLSX parsing.
    """
    if use_cache:
        cache_path = _cache_path(path)
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    df = pd.read_excel(path)
    df = df.rename(columns=lambda c: c.strip().replace(" ", "_"))
    if "LabelDate" not in df.columns:
        raise ValueError("Expected a 'LabelDate' column in the NetNutrition export.")
    df["LabelDate"] = pd.to_datetime(df["LabelDate"]).dt.date

    df["Allergens"] = df.get("Allergens", "").fillna("").replace({"nan": "", "NaN": ""})
    df["Allergens"] = df["Allergens"].apply(lambda val: val.strip() if val else "none listed")

    for key in NUTRIENT_KEYS:
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce").fillna(0.0)
        else:
            df[key] = 0.0

    df["ServingGramWgt"] = pd.to_numeric(df.get("ServingGramWgt"), errors="coerce").fillna(0.0)
    if use_cache:
        cache_path = _cache_path(path)
        df.to_parquet(cache_path, index=False)
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


def normalize_meal_split(raw_split: Optional[Sequence[str]]) -> Dict[str, float]:
    """
    Convert CLI strings like ["Breakfast=30", "Lunch=40", "Dinner=30"] into
    fractions that sum to 1.0.  Defaults to DEFAULT_MEAL_SPLIT.
    """
    if not raw_split:
        return DEFAULT_MEAL_SPLIT.copy()

    meal_split: Dict[str, float] = {}
    for chunk in raw_split:
        if "=" not in chunk:
            raise ValueError(f"Meal split '{chunk}' must look like Breakfast=35.")
        meal, val = chunk.split("=", 1)
        try:
            pct = float(val)
        except ValueError as exc:
            raise ValueError(f"Meal split '{chunk}' is not numeric.") from exc
        meal_split[meal.strip()] = pct

    total = sum(meal_split.values())
    if total <= 0:
        raise ValueError("Meal split percentages must sum to a positive number.")

    return {meal: pct / total for meal, pct in meal_split.items()}


def _tokenize_allergens(value: str) -> Tuple[str, ...]:
    tokens = []
    for piece in value.replace("/", ",").replace(";", ",").split(","):
        piece = piece.strip().lower()
        if piece:
            tokens.append(piece)
    return tuple(sorted(set(tokens)))


@dataclass(frozen=True)
class MenuItem:
    """Structured view of a single NetNutrition menu item."""

    name: str
    meal: str
    service_course: str
    station: str
    serving_g: float
    nutrients: Dict[str, float]
    ingredients: Optional[str]
    allergens: Optional[str]
    allergen_tokens: Tuple[str, ...] = field(default_factory=tuple)
    search_blob: str = ""

    def scaled_nutrients(self, servings: int) -> Dict[str, float]:
        return {k: v * servings for k, v in self.nutrients.items()}


@dataclass
class PlannerConstraints:
    """
    Personalization and compliance switches for the meal planner.
    """

    exclude_allergens: Tuple[str, ...] = ()
    include_keywords: Tuple[str, ...] = ()
    exclude_keywords: Tuple[str, ...] = ()
    station_filter: Optional[str] = None
    meal_filter: Optional[Tuple[str, ...]] = None
    max_repeat_per_week: Optional[int] = None
    max_servings_per_item: int = 2
    max_items_per_meal: int = 6
    alternates: int = 1


class MealPlannerModel:
    """
    Optimization-backed planner that chooses servings close to macro targets.
    """

    def __init__(
        self,
        menu: pd.DataFrame,
        daily_calories: int,
        meal_split: Dict[str, float],
        macro_split: Tuple[int, int, int],
        constraints: Optional[PlannerConstraints] = None,
    ) -> None:
        self.menu = menu
        self.daily_calories = daily_calories
        self.meal_split = meal_split
        self.macro_split = macro_split
        self.constraints = constraints or PlannerConstraints()

        self.meal_calorie_targets = {
            meal: daily_calories * frac for meal, frac in meal_split.items()
        }
        self.macro_targets = compute_macro_targets(self.meal_calorie_targets, macro_split)
        self._repeat_tracker: Dict[Tuple[int, int, str], int] = {}

    def generate_plan(self, start_date: date, end_date: date) -> Dict[str, Dict]:
        days = pd.date_range(start=start_date, end=end_date, freq="D").date
        plan: Dict[str, Dict] = {}

        for day in days:
            day_df = self.menu[self.menu["LabelDate"] == day]
            if day_df.empty:
                plan[str(day)] = {"meals": {}, "daily_totals": {key: 0.0 for key in NUTRIENT_KEYS}}
                continue

            if self.constraints.station_filter:
                station_df = None
                for column in STATION_COLUMNS:
                    if column in day_df.columns:
                        station_df = day_df[day_df[column].fillna("").str.contains(self.constraints.station_filter, case=False, na=False)]
                        break
                day_df = station_df if station_df is not None else day_df

            day_plan = {}
            day_totals = {key: 0.0 for key in NUTRIENT_KEYS}
            week_key = day.isocalendar()

            meals = day_df["Meal"].dropna().unique()
            if self.constraints.meal_filter:
                meals = [m for m in meals if m in self.constraints.meal_filter]

            for meal_name in sorted(meals):
                if meal_name not in self.meal_split:
                    continue

                meal_df = day_df[day_df["Meal"] == meal_name]
                options = self._plan_with_alternates(meal_df, meal_name, week_key)
                if not options:
                    continue

                day_plan[meal_name] = {
                    "options": options,
                    "target": self.macro_targets.get(meal_name, {}),
                }
                primary = options[0]
                for key in NUTRIENT_KEYS:
                    day_totals[key] += primary["totals"].get(key, 0.0)
                self._register_repeat(week_key, primary["items"])

            plan[str(day)] = {"meals": day_plan, "daily_totals": day_totals}

        return plan

    def _plan_with_alternates(self, meal_df: pd.DataFrame, meal_name: str, week_key) -> List[Dict]:
        items = self._extract_items(meal_df, meal_name)
        if not items:
            return []

        target = self.macro_targets.get(meal_name, {"KCAL_Value": self.daily_calories / 3})
        blocked: set[str] = set()
        options: List[Dict] = []
        total_options = max(1, self.constraints.alternates + 1)

        for _ in range(total_options):
            option = self._solve_ilp(items, target, week_key, blocked)
            if not option:
                break
            options.append(option)

            # Remove the calorie-heaviest item for the next alternate to ensure diversity.
            heaviest = max(option["items"], key=lambda item: item["nutrients"]["KCAL_Value"], default=None)
            if heaviest:
                blocked.add(heaviest["name"])

        return options

    def _solve_ilp(
        self,
        items: List[MenuItem],
        target: Dict[str, float],
        week_key,
        blocked: set[str],
    ) -> Optional[Dict]:
        allowed: Dict[str, MenuItem] = {}
        for item in items:
            if item.name in blocked:
                continue
            allowance = self._remaining_allowance(week_key, item.name)
            if allowance <= 0:
                continue
            allowed[item.name] = item

        if not allowed:
            return None

        problem = pulp.LpProblem("MealPlanner", pulp.LpMinimize)
        serving_vars: Dict[str, pulp.LpVariable] = {}

        for name, item in allowed.items():
            cap = min(self.constraints.max_servings_per_item, self._remaining_allowance(week_key, name))
            if cap <= 0:
                continue
            serving_vars[name] = pulp.LpVariable(f"serv_{hashlib.md5(name.encode()).hexdigest()[:6]}", lowBound=0, upBound=cap, cat="Integer")

        if not serving_vars:
            return None

        total_servings = pulp.lpSum(serving_vars.values())
        problem += total_servings >= 1
        problem += total_servings <= max(1, self.constraints.max_items_per_meal)

        objective_terms = []
        weight_map = {
            "KCAL_Value": 1.0,
            "TotalCarb_Gram": 0.6,
            "Protein_Gram": 0.9,
            "TotalFat_Gram": 0.5,
        }
        totals_lookup: Dict[str, pulp.LpAffineExpression] = {}

        for nutrient, weight in weight_map.items():
            expr = pulp.lpSum(var * allowed[name].nutrients.get(nutrient, 0.0) for name, var in serving_vars.items())
            target_value = target.get(nutrient, 0.0)
            pos = pulp.LpVariable(f"{nutrient}_pos", lowBound=0)
            neg = pulp.LpVariable(f"{nutrient}_neg", lowBound=0)
            problem += expr - target_value == pos - neg
            objective_terms.append(weight * (pos + neg))
            totals_lookup[nutrient] = expr

        # Gentle preference for keyword hits so toppings/veggies are not ignored.
        keyword_weight = 0.05
        if self.constraints.include_keywords:
            lowered = tuple(kw.lower() for kw in self.constraints.include_keywords)
        else:
            lowered = ()

        for name, var in serving_vars.items():
            item = allowed[name]
            prefer_hits = sum(1 for kw in lowered if kw in item.search_blob)
            avoid_hits = sum(1 for kw in self.constraints.exclude_keywords if kw in item.search_blob)
            if prefer_hits:
                objective_terms.append(-keyword_weight * prefer_hits * var)
            if avoid_hits:
                objective_terms.append(keyword_weight * avoid_hits * var)

        # Mild penalty on the number of servings to discourage over-selection.
        objective_terms.append(0.01 * total_servings)

        problem += pulp.lpSum(objective_terms)
        solver = pulp.PULP_CBC_CMD(msg=False)
        status = problem.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            return None

        selection: Dict[str, int] = {}
        for name, var in serving_vars.items():
            qty = int(round(var.value() or 0))
            if qty > 0:
                selection[name] = qty

        if not selection:
            return None

        totals = {key: 0.0 for key in NUTRIENT_KEYS}
        payload_items = []
        for name, count in selection.items():
            item = allowed[name]
            scaled = item.scaled_nutrients(count)
            for key in totals.keys():
                totals[key] += scaled.get(key, 0.0)
            payload_items.append(
                {
                    "name": item.name,
                    "servings": count,
                    "nutrients": scaled,
                    "ingredients": item.ingredients,
                    "allergens": item.allergens,
                    "service_course": item.service_course,
                    "station": item.station,
                }
            )

        return {
            "items": sorted(payload_items, key=lambda itm: itm["name"]),
            "totals": totals,
            "score": self._score(totals, target),
        }

    def _register_repeat(self, week_key, items: List[Dict]) -> None:
        if not self.constraints.max_repeat_per_week:
            return
        week = (week_key.year, week_key.week)
        for item in items:
            key = (week[0], week[1], item["name"])
            self._repeat_tracker[key] = self._repeat_tracker.get(key, 0) + item["servings"]

    def _remaining_allowance(self, week_key, item_name: str) -> int:
        if not self.constraints.max_repeat_per_week:
            return self.constraints.max_servings_per_item
        week = (week_key.year, week_key.week)
        key = (week[0], week[1], item_name)
        used = self._repeat_tracker.get(key, 0)
        return max(0, self.constraints.max_repeat_per_week - used)

    def _score(self, totals: Dict[str, float], target: Dict[str, float]) -> float:
        weights = {
            "KCAL_Value": 1.0,
            "TotalCarb_Gram": 0.6,
            "Protein_Gram": 0.9,
            "TotalFat_Gram": 0.5,
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

    def _extract_items(self, meal_df: pd.DataFrame, meal_name: str) -> List[MenuItem]:
        items: List[MenuItem] = []
        banned_allergens = tuple(allo.lower() for allo in self.constraints.exclude_allergens)
        prefer_keywords = tuple(kw.lower() for kw in self.constraints.include_keywords)
        avoid_keywords = tuple(kw.lower() for kw in self.constraints.exclude_keywords)

        name_counts: Dict[str, int] = {}

        for _, row in meal_df.iterrows():
            nutrients = {key: float(row.get(key, 0.0)) for key in NUTRIENT_KEYS}
            name = row.get("FormalName") or row.get("Recipe", "Unknown Item")
            count = name_counts.get(name, 0) + 1
            name_counts[name] = count
            if count > 1:
                name = f"{name} #{count}"
            station = ""
            for col in STATION_COLUMNS:
                if col in row and pd.notna(row[col]):
                    station = str(row[col])
                    break
            ingredients = row.get("Ingredients")
            if isinstance(ingredients, float) and pd.isna(ingredients):
                ingredients = None
            allergens = row.get("Allergens", "none listed")
            if isinstance(allergens, float) and pd.isna(allergens):
                allergens = "none listed"
            allergen_tokens = _tokenize_allergens(allergens if isinstance(allergens, str) else str(allergens))

            if banned_allergens and any(allo in allergen_tokens for allo in banned_allergens):
                continue

            text_blob = " ".join(
                str(val).lower()
                for val in (name, ingredients or "", station, row.get("ServiceCourse", ""), row.get("Recipe", ""))
                if val
            )
            if avoid_keywords and any(kw in text_blob for kw in avoid_keywords):
                continue
            if prefer_keywords and not any(kw in text_blob for kw in prefer_keywords):
                # Still keep it available but mark blob for objective preferences.
                pass

            item = MenuItem(
                name=name,
                meal=meal_name,
                service_course=row.get("ServiceCourse", "Other"),
                station=station,
                serving_g=float(row.get("ServingGramWgt", 0.0)),
                nutrients=nutrients,
                ingredients=ingredients,
                allergens=allergens,
                allergen_tokens=allergen_tokens,
                search_blob=text_blob,
            )
            items.append(item)

        return items


def pretty_print_plan(plan: Dict[str, Dict], station_name: Optional[str]) -> str:
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
            options = details.get("options", [])
            if not options:
                lines.append(f"  {meal}: no viable plan.")
                continue
            primary = options[0]
            totals = primary["totals"]
            lines.append(
                f"  {meal}: {round(totals['KCAL_Value'])} kcal | "
                f"{round(totals['TotalFat_Gram'])}g fat | "
                f"{round(totals['TotalCarb_Gram'])}g carb | "
                f"{round(totals['Protein_Gram'])}g protein "
                f"(score={primary['score']:.3f})"
            )
            for item in primary["items"]:
                lines.append(
                    f"    - {item['servings']}× {item['name']} "
                    f"({round(item['nutrients']['KCAL_Value'])} kcal, "
                    f"{round(item['nutrients']['TotalCarb_Gram'])}g carb, "
                    f"{round(item['nutrients']['Protein_Gram'])}g protein, "
                    f"{round(item['nutrients']['TotalFat_Gram'])}g fat)"
                )
            if len(options) > 1:
                lines.append(f"    → {len(options) - 1} alternate option(s) available.")

    return "\n".join(lines)


def plan_from_file(
    menu_file: Path,
    *,
    start_date: date,
    end_date: date,
    daily_calories: int,
    macro_split: Tuple[int, int, int],
    meal_split: Dict[str, float],
    constraints: PlannerConstraints,
    use_cache: bool = True,
) -> Dict[str, Dict]:
    menu_df = load_menu(menu_file, use_cache=use_cache)
    planner = MealPlannerModel(
        menu=menu_df,
        daily_calories=daily_calories,
        meal_split=meal_split,
        macro_split=macro_split,
        constraints=constraints,
    )
    return planner.generate_plan(start_date=start_date, end_date=end_date)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimization-based meal planner for NetNutrition station exports.")
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
        default=1800,
        help="Target calories per day used to set meal goals (default: 1800).",
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
        "--meal-split",
        nargs="+",
        metavar="MEAL=PCT",
        help="Override the default meal split (e.g., Breakfast=25 Lunch=40 Dinner=35).",
    )
    parser.add_argument(
        "--exclude-allergens",
        nargs="*",
        default=(),
        help="Allergens to avoid entirely (space-separated list, e.g., milk peanuts).",
    )
    parser.add_argument(
        "--prefer-keyword",
        nargs="*",
        default=(),
        help="Ingredient/menu keywords to prioritize (case-insensitive).",
    )
    parser.add_argument(
        "--avoid-keyword",
        nargs="*",
        default=(),
        help="Keywords that should be filtered out from consideration.",
    )
    parser.add_argument(
        "--station-filter",
        default=None,
        help="Restrict planning to stations matching this string.",
    )
    parser.add_argument(
        "--meal-filter",
        nargs="*",
        default=None,
        help="Restrict planning to these meals (e.g., Breakfast Lunch Dinner).",
    )
    parser.add_argument(
        "--max-repeat-per-week",
        type=int,
        default=None,
        help="Maximum number of times a menu item may appear per ISO week.",
    )
    parser.add_argument(
        "--max-servings-per-item",
        type=int,
        default=2,
        help="Upper bound for servings of a single item within one meal.",
    )
    parser.add_argument(
        "--max-items-per-meal",
        type=int,
        default=6,
        help="Upper bound on the number of servings picked for a meal.",
    )
    parser.add_argument(
        "--alternates",
        type=int,
        default=1,
        help="How many alternate options (per meal) to compute in addition to the primary plan.",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format for the final plan (default: text).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Parquet menu caching for this run.",
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

    meal_split = normalize_meal_split(args.meal_split)
    constraints = PlannerConstraints(
        exclude_allergens=tuple(args.exclude_allergens),
        include_keywords=tuple(args.prefer_keyword),
        exclude_keywords=tuple(args.avoid_keyword),
        station_filter=args.station_filter,
        meal_filter=tuple(args.meal_filter) if args.meal_filter else None,
        max_repeat_per_week=args.max_repeat_per_week,
        max_servings_per_item=args.max_servings_per_item,
        max_items_per_meal=args.max_items_per_meal,
        alternates=max(0, args.alternates),
    )

    start_date = pd.to_datetime(args.start_date).date()
    end_date = pd.to_datetime(args.end_date).date()

    plan = plan_from_file(
        args.menu_file,
        start_date=start_date,
        end_date=end_date,
        daily_calories=args.daily_calories,
        macro_split=tuple(args.macro_split),  # type: ignore[arg-type]
        meal_split=meal_split,
        constraints=constraints,
        use_cache=not args.no_cache,
    )

    if args.output == "json":
        print(json.dumps(plan, indent=2))
    else:
        print(pretty_print_plan(plan, args.station_name))


if __name__ == "__main__":
    main()
