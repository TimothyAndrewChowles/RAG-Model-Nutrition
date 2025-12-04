# 🍎 ISU Nutrition Model

An intelligent meal-planning assistant for Illinois State University dining data.  
Query menus, get personalized meal plans, and analyze nutrition with serving sizes in grams.

---

## 🚀 Highlights
- Smart search over ISU dining items (by name, ingredients, allergens)
- Optimization-backed meal planner honoring calorie targets, macro splits, allergens, keyword preferences, and repeat limits
- Serving sizes in grams with per-item nutrition totals
- RAG pipeline: local embeddings + vector DB + LLM for answers
- Desktop GUI with conversational search, analytics dashboard, and a full Meal Planner workspace
- CLI utilities for quick Q&A and station meal plans, with cached parsing for faster re-runs

---

## 🧠 How it Works
1. **Ingest:** load dining CSV/JSON → clean → embed → store in ChromaDB  
2. **Retrieve:** find relevant items by text or filters  
3. **Generate:** craft a plan that hits target kcal/macros with gram-level portions  
4. **Explain:** return items, portions, totals, and notes

## ⚙️ Setup
Use Python 3.10+ and a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Code/requirements.txt
```

Key dependencies: `pandas`, `numpy`, `openpyxl`, `sentence-transformers`, `pulp` (ILP solver), and `pyarrow`. Torch installs alongside `sentence-transformers`; install `torch` explicitly if your environment needs it.  
Put your NetNutrition `.xlsx` exports in `Dining Food Info/` (default lookup) or point the tools at another folder. Menu parsing caches normalized Parquet files under `Code/.cache/`; delete that folder to force a fresh parse.

## 🖥️ Desktop GUI
Launch the desktop explorer:

```bash
python Code/gui.py
```

Pick the folder that holds your NetNutrition Excel exports.  
- **Ask Questions:** conversational search with a “High protein picks” shortcut.  
- **Nutrition Dashboard:** filter by date range, meal, station, or allergen text to see macro averages/totals, top protein and sodium items, station variety, and export filtered rows to CSV.  
- **Meal Planner:** set calorie and macro targets, per-meal splits, allergen toggles, keyword prefer/avoid lists, station and meal filters, max repeat limits, servings/items caps, and alternates. Plans render in a tree with per-meal totals, ingredient/allergen details, and CSV/JSON export buttons. Long runs are threaded to keep the UI responsive.

## 💬 CLI Q&A
Ask questions over the menus (defaults to the `Dining Food Info/` folder):

```bash
python Code/model.py --menu-dir "Dining Food Info" --question "What is a high protein food?"
```

## 🍽️ Meal Planner CLI
Automation-friendly planner with the same switches as the GUI:

```bash
python Code/meal_planner.py \
  --menu-file "Dining Food Info/Station 9 10.20.25-10.26.25.xlsx" \
  --start-date 2025-10-27 --end-date 2025-11-02 \
  --daily-calories 2000 \
  --macro-split 45 30 25 \
  --meal-split Breakfast=30 Lunch=40 Dinner=30 \
  --exclude-allergens milk peanuts soy \
  --prefer-keyword veggie --avoid-keyword fried \
  --station-filter "Marketplace" --meal-filter Lunch Dinner \
  --max-repeat-per-week 2 --alternates 2 --output json
```

Helpful flags:
- `--meal-split` accepts arbitrary `MEAL=PCT` pairs that are normalized at runtime.
- `--prefer-keyword` / `--avoid-keyword` bias the optimizer toward desirable/undesirable ingredients.
- `--max-repeat-per-week` caps how often an item can appear in ISO week windows.
- `--max-servings-per-item` and `--max-items-per-meal` bound serving counts per meal.
- `--no-cache` forces a fresh XLSX parse (otherwise Parquet cache is reused).
