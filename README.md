# 🍎 ISU Nutrition Model

An intelligent meal-planning assistant for Illinois State University dining data.  
Query menus, get personalized meal plans, and analyze nutrition with serving sizes in grams.

---

## 🚀 Highlights
- Smart search over ISU dining items (by name, ingredients, allergens)
- AI meal plans for calorie and macro goals (breakfast, lunch, dinner)
- Serving sizes in grams with per-item nutrition totals
- RAG pipeline: local embeddings + vector DB + LLM for answers
- FastAPI endpoint + simple CLI for quick testing
- Desktop GUI for manual exploration of menu questions and dashboards

---

## 🧠 How it Works
1. **Ingest:** load dining CSV/JSON → clean → embed → store in ChromaDB  
2. **Retrieve:** find relevant items by text or filters  
3. **Generate:** craft a plan that hits target kcal/macros with gram-level portions  
4. **Explain:** return items, portions, totals, and notes

## 🖥️ Desktop GUI
Install the dependencies listed in `Code/requirements.txt`, then launch:

```bash
python Code/gui.py
```

Pick the folder that holds your NetNutrition Excel exports. Tab 1 lets you ask natural-language questions (with a quick “High protein picks” shortcut). Tab 2 hosts a nutrition dashboard: filter by date range, meal, station, or allergen text to see macro averages/totals, top protein and sodium items, station variety, and export the filtered rows to CSV for deeper analysis.
