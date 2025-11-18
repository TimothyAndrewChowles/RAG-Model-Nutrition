# 🍎 ISU Nutrition Model

An intelligent meal-planning assistant for Illinois State University dining data.  
Query menus, get personalized meal plans, and analyze nutrition with serving sizes in grams.

---

## 🚀 Highlights
- Smart search over ISU dining items (by name, ingredients, allergens)
- Optimization-backed meal planner that honors calorie targets, macro splits, allergens, keyword preferences, and repeat limits
- Serving sizes in grams with per-item nutrition totals
- RAG pipeline: local embeddings + vector DB + LLM for answers
- Desktop GUI with conversational search, analytics dashboard, and a full Meal Planner workspace
- FastAPI/CLI hooks for automation, now with NetNutrition caching for faster re-runs

---

## 🧠 How it Works
1. **Ingest:** load dining CSV/JSON → clean → embed → store in ChromaDB  
2. **Retrieve:** find relevant items by text or filters  
3. **Generate:** craft a plan that hits target kcal/macros with gram-level portions  
4. **Explain:** return items, portions, totals, and notes

## ⚙️ Setup

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

```bash
python Code/gui.py
```

- **Ask Questions:** conversational MenuRAG search with quick shortcuts.
- **Nutrition Dashboard:** filter by date, meal, station, or allergen snippets to summarize macros and export filtered rows.
- **Meal Planner (new):** configure calorie + macro targets, per-meal splits, allergen toggles, keyword preferences/avoid lists, station + meal filters, max repeats, and number of alternates. Plans render in a Treeview with per-meal totals, ingredient/allergen details, and CSV/JSON export buttons. Long-running planning happens on a worker thread so the UI stays responsive.

## 🧾 Meal Planner CLI

The CLI mirrors the GUI controls for automation/scripting:

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
