# 🍎 ISU Nutrition Model

An intelligent meal-planning assistant for Illinois State University dining data.  
Query menus, get personalized meal plans, and analyze nutrition with serving sizes in grams.

---

## 🚀 Highlights
- Smart search over ISU dining items (by name, ingredients, allergens)
- AI meal plans for calorie and macro goals (breakfast, lunch, dinner)
- Serving sizes in grams with per-item nutrition totals
- RAG pipeline: local embeddings + vector DB + LLM for answers
- CLI utilities for quick Q&A and station meal plans
- Desktop GUI for manual exploration of menu questions and dashboards

---

## 🧠 How it Works
1. **Ingest:** load dining CSV/JSON → clean → embed → store in ChromaDB  
2. **Retrieve:** find relevant items by text or filters  
3. **Generate:** craft a plan that hits target kcal/macros with gram-level portions  
4. **Explain:** return items, portions, totals, and notes

## ⚙️ Setup
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

```bash
python Code/gui.py
```

Pick the folder that holds your NetNutrition Excel exports. Tab 1 lets you ask natural-language questions (with a quick “High protein picks” shortcut). Tab 2 hosts a nutrition dashboard: filter by date range, meal, station, or allergen text to see macro averages/totals, top protein and sodium items, station variety, and export the filtered rows to CSV for deeper analysis.

## 💬 CLI Q&A
Ask questions over the menus (defaults to the `Dining Food Info/` folder):

```bash
python Code/model.py --menu-dir "Dining Food Info" --question "What is a high protein food?"
```

## 🍽️ Meal Planner
Build a simple station plan over a date range:

```bash
python Code/meal_planner.py \
  --menu-file "Dining Food Info/Station 9 10.20.25-10.26.25.xlsx" \
  --start-date 2025-10-20 \
  --end-date 2025-10-26 \
  --daily-calories 1600 \
  --output text
```
