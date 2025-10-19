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

---

## 🧠 How it Works
1. **Ingest:** load dining CSV/JSON → clean → embed → store in ChromaDB  
2. **Retrieve:** find relevant items by text or filters  
3. **Generate:** craft a plan that hits target kcal/macros with gram-level portions  
4. **Explain:** return items, portions, totals, and notes


