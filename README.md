# 🍎 ISU Nutrition Model

An intelligent meal-planning assistant for Illinois State University dining data.  
Query menus, get personalized meal plans, and analyze nutrition with serving sizes in grams.

---

## 🚀 Highlights
- Smart search over ISU dining items (by name, ingredients, allergens)
- Optimization-backed meal planner that honors calorie targets, macro splits, allergens, keyword preferences, and repeat limits
- Serving sizes in grams with per-item nutrition totals
- RAG pipeline: local embeddings + vector DB + LLM for answers
<<<<<<< HEAD
- CLI utilities for quick Q&A and station meal plans
- Desktop GUI for manual exploration of menu questions and dashboards
=======
- Desktop GUI with conversational search, analytics dashboard, and a full Meal Planner workspace
- FastAPI/CLI hooks for automation, now with NetNutrition caching for faster re-runs
>>>>>>> 6eae690ccc46dc599a0eda88c49fbdf3702bfa4e

---

## 🧠 How it Works
1. **Ingest:** load dining CSV/JSON → clean → embed → store in ChromaDB  
2. **Retrieve:** find relevant items by text or filters  
3. **Generate:** craft a plan that hits target kcal/macros with gram-level portions  
4. **Explain:** return items, portions, totals, and notes

## ⚙️ Setup
<<<<<<< HEAD
1. Use Python 3.10+ (recommended: a fresh virtualenv):  
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
2. Install dependencies (more than the minimal `Code/requirements.txt`):  
   ```bash
   pip install pandas numpy sentence-transformers torch tk
   ```
3. Put your NetNutrition `.xlsx` exports in `Dining Food Info/` (default lookup) or another folder you will pass to the commands below.

## 🖥️ Desktop GUI
Launch the desktop explorer:
=======

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Code/requirements.txt
```

Key dependencies:
- `pandas`, `numpy`, `openpyxl` for NetNutrition parsing
- `pulp` + CBC solver for the ILP planner
- `pyarrow` for cached Parquet normalization
- `sentence-transformers` for MenuRAG embeddings

Menu parsing automatically caches normalized Parquet files under `Code/.cache/` keyed by file path + timestamp. Delete that folder to force a fresh parse.

## 🖥️ Desktop GUI
With dependencies installed, launch:
>>>>>>> 6eae690ccc46dc599a0eda88c49fbdf3702bfa4e

```bash
python Code/gui.py
```

<<<<<<< HEAD
Pick the folder that holds your NetNutrition Excel exports. Tab 1 lets you ask natural-language questions (with a quick “High protein picks” shortcut). Tab 2 hosts a nutrition dashboard: filter by date range, meal, station, or allergen text to see macro averages/totals, top protein and sodium items, station variety, and export the filtered rows to CSV for deeper analysis.

## 💬 CLI Q&A
Ask questions over the menus (defaults to the `Dining Food Info/` folder):

```bash
python Code/model.py --menu-dir "Dining Food Info" --question "What is a high protein food?"
```

## 🍽️ Meal Planner
Build a simple station plan over a date range:
=======
- **Ask Questions:** conversational MenuRAG search with quick shortcuts.
- **Nutrition Dashboard:** filter by date, meal, station, or allergen snippets to summarize macros and export filtered rows.
- **Meal Planner (new):** configure calorie + macro targets, per-meal splits, allergen toggles, keyword preferences/avoid lists, station + meal filters, max repeats, and number of alternates. Plans render in a Treeview with per-meal totals, ingredient/allergen details, and CSV/JSON export buttons. Long-running planning happens on a worker thread so the UI stays responsive.

## 🧾 Meal Planner CLI

The CLI mirrors the GUI controls for automation/scripting:
>>>>>>> 6eae690ccc46dc599a0eda88c49fbdf3702bfa4e

```bash
python Code/meal_planner.py \
  --menu-file "Dining Food Info/Station 9 10.20.25-10.26.25.xlsx" \
<<<<<<< HEAD
  --start-date 2025-10-20 \
  --end-date 2025-10-26 \
  --daily-calories 1600 \
  --output text
```
=======
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
- `--prefer-keyword` / `--avoid-keyword` bias the ILP objective toward desirable ingredients.
- `--max-repeat-per-week` caps how often an item can appear in ISO week windows.
- `--no-cache` forces a fresh XLSX parse (otherwise Parquet cache is reused).

`plan_from_file(...)` exposes the same functionality for scripts or future FastAPI endpoints.

## ✅ Testing

Synthetic NetNutrition fixtures live under `Code/tests/fixtures/`. Run pytest after installing requirements (CBC from PuLP is required):

```bash
pytest Code/tests
```

Tests cover menu normalization/caching, macro math, and constraint handling for the planner.
>>>>>>> 6eae690ccc46dc599a0eda88c49fbdf3702bfa4e
