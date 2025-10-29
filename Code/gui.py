#!/usr/bin/env python3
"""
Tkinter desktop entry point for exploring NetNutrition exports.

Tab 1 repeats the conversational search experience backed by MenuRAG.
Tab 2 adds an analytics dashboard that summarizes calories/macros by date,
meal, station, and allergen filters, with CSV export for downstream work.
"""

from __future__ import annotations

import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import pandas as pd

from model import MenuRAG, NUMERIC_KEYS, build_default_menu_dir, discover_menu_files, load_menus


class MenuExplorerApp(tk.Tk):
    """Tkinter front-end for querying dining menus and viewing nutrition summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.title("NetNutrition Menu Explorer")
        self.geometry("820x600")
        self.minsize(720, 520)

        default_dir = build_default_menu_dir()
        self.menu_dir_var = tk.StringVar(value=str(default_dir) if default_dir else "")
        self.status_var = tk.StringVar(value="Select a menu directory and press Load Menus.")
        self.topk_var = tk.IntVar(value=5)

        self._rag: Optional[MenuRAG] = None
        self._menu_df: Optional[pd.DataFrame] = None
        self._dashboard_last_df: Optional[pd.DataFrame] = None
        self._loading = False
        self._interactive_widgets: list[tk.Widget] = []

        self._build_widgets()

    def _build_widgets(self) -> None:
        root_frame = ttk.Frame(self, padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(root_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # Directory selector lives above the tabs.
        dir_frame = ttk.Frame(root_frame)
        dir_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(dir_frame, text="Menu directory:").pack(side=tk.LEFT)
        dir_entry = ttk.Entry(dir_frame, textvariable=self.menu_dir_var)
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self._interactive_widgets.append(dir_entry)

        browse_button = ttk.Button(dir_frame, text="Browse…", command=self._choose_directory)
        browse_button.pack(side=tk.LEFT)
        self._interactive_widgets.append(browse_button)

        load_button = ttk.Button(dir_frame, text="Load Menus", command=self._load_menus_clicked)
        load_button.pack(side=tk.LEFT, padx=(8, 0))
        self._interactive_widgets.append(load_button)

        # Ask tab (MenuRAG search).
        ask_tab = ttk.Frame(notebook, padding=10)
        notebook.add(ask_tab, text="Ask Questions")

        self.question_entry = ttk.Entry(ask_tab)
        self.question_entry.insert(0, "What is a high protein food?")
        self.question_entry.pack(fill=tk.X, padx=4, pady=(4, 0))
        self._interactive_widgets.append(self.question_entry)

        controls = ttk.Frame(ask_tab)
        controls.pack(fill=tk.X, padx=4, pady=6)

        ttk.Label(controls, text="Top results:").pack(side=tk.LEFT)
        self.topk_spinbox = ttk.Spinbox(controls, from_=1, to=15, width=4, textvariable=self.topk_var)
        self.topk_spinbox.pack(side=tk.LEFT, padx=(4, 16))
        self._interactive_widgets.append(self.topk_spinbox)

        ask_button = ttk.Button(controls, text="Ask", command=self._ask_clicked)
        ask_button.pack(side=tk.LEFT)
        self._interactive_widgets.append(ask_button)

        high_protein_button = ttk.Button(controls, text="High protein picks", command=self._high_protein_clicked)
        high_protein_button.pack(side=tk.LEFT, padx=(8, 0))
        self._interactive_widgets.append(high_protein_button)

        output_frame = ttk.Frame(ask_tab)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        self.output_text = tk.Text(output_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.configure(yscrollcommand=scrollbar.set)

        # Analytics tab.
        analytics_tab = ttk.Frame(notebook, padding=10)
        notebook.add(analytics_tab, text="Nutrition Dashboard")

        filters_frame = ttk.LabelFrame(analytics_tab, text="Filters")
        filters_frame.pack(fill=tk.X, expand=False, pady=(0, 10))

        row1 = ttk.Frame(filters_frame)
        row1.pack(fill=tk.X, padx=8, pady=(8, 4))

        ttk.Label(row1, text="Start date:").pack(side=tk.LEFT)
        self.start_date_combo = ttk.Combobox(row1, state="readonly", width=12)
        self.start_date_combo.pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(row1, text="End date:").pack(side=tk.LEFT)
        self.end_date_combo = ttk.Combobox(row1, state="readonly", width=12)
        self.end_date_combo.pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(row1, text="Meal:").pack(side=tk.LEFT)
        self.meal_combo = ttk.Combobox(row1, state="readonly", width=16, values=["All"])
        self.meal_combo.current(0)
        self.meal_combo.pack(side=tk.LEFT, padx=(4, 16))

        row2 = ttk.Frame(filters_frame)
        row2.pack(fill=tk.X, padx=8, pady=(0, 8))

        ttk.Label(row2, text="Station:").pack(side=tk.LEFT)
        self.station_combo = ttk.Combobox(row2, state="readonly", width=22, values=["All"])
        self.station_combo.current(0)
        self.station_combo.pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(row2, text="Exclude allergen text:").pack(side=tk.LEFT)
        self.allergen_entry = ttk.Entry(row2, width=24)
        self.allergen_entry.pack(side=tk.LEFT, padx=(4, 16))

        update_button = ttk.Button(row2, text="Update summary", command=self._update_dashboard_summary)
        update_button.pack(side=tk.LEFT)
        export_button = ttk.Button(row2, text="Export CSV", command=self._export_dashboard_csv)
        export_button.pack(side=tk.LEFT, padx=(8, 0))

        self._interactive_widgets.extend(
            [
                self.start_date_combo,
                self.end_date_combo,
                self.meal_combo,
                self.station_combo,
                self.allergen_entry,
                update_button,
                export_button,
            ]
        )

        summary_frame = ttk.LabelFrame(analytics_tab, text="Summary")
        summary_frame.pack(fill=tk.BOTH, expand=True)

        self.dashboard_text = tk.Text(summary_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.dashboard_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Status bar.
        status_bar = ttk.Label(root_frame, textvariable=self.status_var, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(6, 0))

    def _choose_directory(self) -> None:
        selection = filedialog.askdirectory(title="Select menu directory", initialdir=self.menu_dir_var.get())
        if selection:
            self.menu_dir_var.set(selection)

    def _load_menus_clicked(self) -> None:
        if self._loading:
            return

        raw_dir = self.menu_dir_var.get().strip()
        if not raw_dir:
            messagebox.showinfo("Select directory", "Choose a folder that contains NetNutrition Excel files.")
            return

        menu_dir = Path(raw_dir).expanduser()
        if not menu_dir.exists():
            messagebox.showerror("Directory not found", f"Path '{menu_dir}' does not exist.")
            return

        self._set_loading(True, message="Loading menus…")
        thread = threading.Thread(target=self._load_menus_worker, args=(menu_dir,), daemon=True)
        thread.start()

    def _load_menus_worker(self, menu_dir: Path) -> None:
        try:
            files = discover_menu_files([], menu_dir)
            menu_df = load_menus(files)
            rag = MenuRAG(menu_df=menu_df)
            self.after(
                0,
                lambda: self._on_load_success(
                    rag=rag,
                    menu_count=len(files),
                    item_count=len(rag.documents),
                ),
            )
        except Exception as exc:
            self.after(0, lambda: self._on_load_error(exc))

    def _on_load_success(self, rag: MenuRAG, menu_count: int, item_count: int) -> None:
        self._rag = rag
        self._menu_df = rag.menu_df.copy()
        self._dashboard_last_df = None
        self._set_loading(False, message=f"Loaded {menu_count} file(s) covering {item_count} menu items.")
        self._refresh_dashboard_filters()
        self._update_dashboard_summary()

    def _on_load_error(self, exc: Exception) -> None:
        self._rag = None
        self._menu_df = None
        self._dashboard_last_df = None
        self._set_loading(False, message="Ready.")
        messagebox.showerror("Failed to load menus", str(exc))

    def _ask_clicked(self) -> None:
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showinfo("Ask a question", "Enter a question first.")
            return
        self._submit_query(question, top_k=self.topk_var.get())

    def _high_protein_clicked(self) -> None:
        self._submit_query("What is a high protein food?", top_k=self.topk_var.get())

    def _submit_query(self, question: str, top_k: int) -> None:
        if not self._rag:
            messagebox.showinfo("Load menus", "Load menus before asking questions.")
            return

        self._set_loading(True, message="Running query…")
        thread = threading.Thread(target=self._query_worker, args=(question, top_k), daemon=True)
        thread.start()

    def _query_worker(self, question: str, top_k: int) -> None:
        try:
            response = self._rag.answer_question(question, top_k=top_k)
            self.after(0, lambda: self._on_query_success(question, response))
        except Exception as exc:
            self.after(0, lambda: self._on_query_error(exc))

    def _on_query_success(self, question: str, response: str) -> None:
        self._append_output(f"> {question}\n{response}\n\n")
        self._set_loading(False, message="Ready.")

    def _on_query_error(self, exc: Exception) -> None:
        self._set_loading(False, message="Ready.")
        messagebox.showerror("Query failed", str(exc))

    def _append_output(self, text: str) -> None:
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

    def _refresh_dashboard_filters(self) -> None:
        if self._menu_df is None or self._menu_df.empty:
            for widget in (self.start_date_combo, self.end_date_combo, self.meal_combo, self.station_combo):
                widget.configure(values=["All"])
                widget.set("All")
            self.allergen_entry.delete(0, tk.END)
            self._set_dashboard_text("Load menus to see summary data.")
            return

        dates = sorted(set(self._menu_df["LabelDate"].dropna()))
        date_values = [str(date) for date in dates] or ["All"]

        self.start_date_combo.configure(values=date_values)
        self.end_date_combo.configure(values=date_values)
        self.start_date_combo.set(date_values[0])
        self.end_date_combo.set(date_values[-1])

        meals = ["All"] + sorted(val for val in self._menu_df["Meal"].dropna().unique())
        self.meal_combo.configure(values=meals)
        self.meal_combo.set("All")

        stations = ["All"] + sorted(val for val in self._menu_df["SourceFile"].dropna().unique())
        self.station_combo.configure(values=stations)
        self.station_combo.set("All")

        self.allergen_entry.delete(0, tk.END)

    def _update_dashboard_summary(self) -> None:
        if self._menu_df is None or self._menu_df.empty:
            self._set_dashboard_text("Load menus to see summary data.")
            return

        filtered = self._menu_df.copy()

        start_str = self.start_date_combo.get()
        end_str = self.end_date_combo.get()
        meal = self.meal_combo.get()
        station = self.station_combo.get()
        allergen_text = self.allergen_entry.get().strip().lower()

        if start_str:
            try:
                start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
                filtered = filtered[filtered["LabelDate"] >= start_date]
            except ValueError:
                pass

        if end_str:
            try:
                end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
                filtered = filtered[filtered["LabelDate"] <= end_date]
            except ValueError:
                pass

        if meal and meal != "All":
            filtered = filtered[filtered["Meal"] == meal]

        if station and station != "All":
            filtered = filtered[filtered["SourceFile"] == station]

        if allergen_text:
            filtered = filtered[~filtered["Allergens"].str.lower().str.contains(allergen_text, na=False)]

        self._dashboard_last_df = filtered.copy()

        if filtered.empty:
            self._set_dashboard_text("No items match the current filters.")
            return

        total_items = len(filtered)
        unique_items = filtered["FormalName"].nunique()
        date_span = f"{filtered['LabelDate'].min()} → {filtered['LabelDate'].max()}"
        meal_info = meal if meal and meal != "All" else "all meals"
        station_info = station if station and station != "All" else "all stations"

        available_numeric = [col for col in NUMERIC_KEYS if col in filtered.columns]
        totals = filtered[available_numeric].sum(numeric_only=True)
        averages = filtered[available_numeric].mean(numeric_only=True)

        lines = [
            f"Items: {total_items} total ({unique_items} unique) across {date_span}",
            f"Meals: {meal_info} | Stations: {station_info}",
            "",
            "Average per item:",
            f"  Calories: {averages.get('KCAL_Value', 0.0):.0f} kcal",
            f"  Protein: {averages.get('Protein_Gram', 0.0):.1f} g",
            f"  Carbs: {averages.get('TotalCarb_Gram', 0.0):.1f} g",
            f"  Fat: {averages.get('TotalFat_Gram', 0.0):.1f} g",
            "",
            "Totals across selection:",
            f"  Calories: {totals.get('KCAL_Value', 0.0):.0f} kcal",
            f"  Protein: {totals.get('Protein_Gram', 0.0):.1f} g",
            f"  Carbs: {totals.get('TotalCarb_Gram', 0.0):.1f} g",
            f"  Fat: {totals.get('TotalFat_Gram', 0.0):.1f} g",
            "",
        ]

        top_protein = filtered.sort_values("Protein_Gram", ascending=False).head(3)
        if not top_protein.empty:
            lines.append("Top protein items:")
            for _, row in top_protein.iterrows():
                lines.append(
                    f"  - {row['FormalName']} ({row['Meal']} on {row['LabelDate']}): "
                    f"{row['Protein_Gram']:.1f} g protein, {row['KCAL_Value']:.0f} kcal"
                )
            lines.append("")

        if "Sodium_Milligram" in filtered.columns:
            top_sodium = filtered.sort_values("Sodium_Milligram", ascending=False).head(3)
            if not top_sodium.empty:
                lines.append("Highest sodium items:")
                for _, row in top_sodium.iterrows():
                    lines.append(
                        f"  - {row['FormalName']} ({row['Meal']} on {row['LabelDate']}): "
                        f"{row['Sodium_Milligram']:.0f} mg sodium"
                    )
                lines.append("")

        station_counts = (
            filtered.groupby("SourceFile")["FormalName"].nunique().sort_values(ascending=False).head(5)
            if "SourceFile" in filtered.columns
            else pd.Series(dtype=int)
        )
        if not station_counts.empty:
            lines.append("Menu variety by station (unique items):")
            for station_name, count in station_counts.items():
                lines.append(f"  - {station_name}: {count}")
            lines.append("")

        self._set_dashboard_text("\n".join(lines))

    def _set_dashboard_text(self, message: str) -> None:
        self.dashboard_text.configure(state=tk.NORMAL)
        self.dashboard_text.delete("1.0", tk.END)
        self.dashboard_text.insert(tk.END, message)
        self.dashboard_text.configure(state=tk.DISABLED)

    def _export_dashboard_csv(self) -> None:
        if self._dashboard_last_df is None or self._dashboard_last_df.empty:
            messagebox.showinfo("Nothing to export", "Run the summary first to populate results.")
            return

        path = filedialog.asksaveasfilename(
            title="Export filtered items",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self._dashboard_last_df.to_csv(path, index=False)
            messagebox.showinfo("Export complete", f"Saved {len(self._dashboard_last_df)} items to {path}.")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def _set_loading(self, loading: bool, message: Optional[str] = None) -> None:
        self._loading = loading
        if message:
            self.status_var.set(message)

        for widget in self._interactive_widgets:
            try:
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state="disabled" if loading else "readonly")
                elif isinstance(widget, ttk.Spinbox):
                    widget.configure(state="disabled" if loading else "normal")
                else:
                    widget.configure(state=tk.DISABLED if loading else tk.NORMAL)
            except tk.TclError:
                widget.configure(state="disabled" if loading else "normal")

    def run(self) -> None:
        self.mainloop()


def main() -> None:
    app = MenuExplorerApp()
    app.run()


if __name__ == "__main__":
    main()
