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
        self.geometry("960x660")
        self.minsize(820, 560)

        self._configure_styles()

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

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_bg = "#f5f5f7"
        surface_bg = "#ffffff"
        accent = "#0a84ff"
        text_primary = "#1c1c1e"
        text_secondary = "#6e6e73"
        subtle_border = "#d2d2d7"

        self._theme = {
            "base_bg": base_bg,
            "surface_bg": surface_bg,
            "accent": accent,
            "text_primary": text_primary,
            "text_secondary": text_secondary,
            "subtle_border": subtle_border,
        }

        self.configure(background=base_bg)
        style.configure(".", background=base_bg, foreground=text_primary)
        style.configure("App.TFrame", background=base_bg)
        style.configure("Hero.TFrame", background=base_bg)
        style.configure("Card.TFrame", background=surface_bg, relief="flat", borderwidth=1, bordercolor=subtle_border)
        style.configure("Card.TLabelframe", background=surface_bg, bordercolor=subtle_border, relief="flat")
        style.configure(
            "Card.TLabelframe.Label",
            background=surface_bg,
            foreground=text_primary,
            font=(".AppleSystemUIFont", 12, "bold"),
        )
        style.configure(
            "Title.TLabel",
            font=(".AppleSystemUIFont", 22, "bold"),
            foreground=text_primary,
            background=base_bg,
        )
        style.configure(
            "Subtitle.TLabel",
            font=(".AppleSystemUIFont", 12),
            foreground=text_secondary,
            background=base_bg,
        )
        style.configure(
            "Section.TLabel",
            font=(".AppleSystemUIFont", 14, "semibold"),
            foreground=text_primary,
            background=surface_bg,
        )
        style.configure("Status.TLabel", font=(".AppleSystemUIFont", 11), foreground=text_secondary, background=base_bg)

        style.configure(
            "Accent.TButton",
            background=accent,
            foreground="#ffffff",
            font=(".AppleSystemUIFont", 12, "semibold"),
            borderwidth=0,
            focusthickness=3,
            focuscolor=accent,
            padding=(14, 8),
        )
        style.map(
            "Accent.TButton",
            background=[("disabled", "#b7d7ff"), ("pressed", "#0060df"), ("active", "#358bfd")],
            foreground=[("disabled", "#f5faff")],
        )

        style.configure(
            "Secondary.TButton",
            background="#e4ebf5",
            foreground=accent,
            font=(".AppleSystemUIFont", 12),
            borderwidth=0,
            padding=(14, 8),
        )
        style.map(
            "Secondary.TButton",
            background=[("pressed", "#d4deee"), ("active", "#dde6f4")],
            foreground=[("disabled", "#a0b3c8")],
        )

        style.configure(
            "TEntry",
            fieldbackground=surface_bg,
            foreground=text_primary,
            bordercolor=subtle_border,
            lightcolor=subtle_border,
            darkcolor=subtle_border,
            insertcolor=text_primary,
            padding=6,
            relief="flat",
        )
        style.configure(
            "TCombobox",
            fieldbackground=surface_bg,
            background=surface_bg,
            bordercolor=subtle_border,
            foreground=text_primary,
            arrowsize=16,
        )
        style.configure(
            "TSpinbox",
            fieldbackground=surface_bg,
            bordercolor=subtle_border,
            foreground=text_primary,
            arrowsize=14,
        )

        style.configure(
            "Card.Vertical.TScrollbar",
            troughcolor=surface_bg,
            bordercolor=surface_bg,
            background="#dadde6",
            arrowsize=12,
        )
        style.map(
            "Card.Vertical.TScrollbar",
            background=[("pressed", "#b6bdcb"), ("active", "#c8cdd8"), ("!active", "#dadde6")],
            arrowcolor=[("active", accent), ("!active", "#86868b")],
        )

        style.configure("TNotebook", background=base_bg, tabposition="n", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            font=(".AppleSystemUIFont", 12),
            padding=(18, 10),
            background=base_bg,
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", surface_bg), ("active", surface_bg)],
            foreground=[("selected", text_primary), ("!selected", text_secondary)],
        )

        self.option_add("*TCombobox*Listbox.font", ".AppleSystemUIFont 12")
        self.option_add("*Font", ".AppleSystemUIFont 12")
        self.option_add("*TButton.padding", 6)
        self.option_add("*TEntry.padding", 6)
        self.option_add("*TEntry.font", ".AppleSystemUIFont 12")
        self.option_add("*TEntry.borderWidth", 1)
        self.option_add("*TEntry.relief", "flat")

    def _build_widgets(self) -> None:
        theme = getattr(self, "_theme", {})
        surface_bg = theme.get("surface_bg", "#ffffff")
        text_primary = theme.get("text_primary", "#1c1c1e")
        text_secondary = theme.get("text_secondary", "#6e6e73")

        root_frame = ttk.Frame(self, padding=22, style="App.TFrame")
        root_frame.pack(fill=tk.BOTH, expand=True)

        hero = ttk.Frame(root_frame, style="Hero.TFrame")
        hero.pack(fill=tk.X, pady=(0, 18))

        ttk.Label(hero, text="NetNutrition Navigator", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            hero,
            text="Explore dining menus, surface nutrition insights, and build personalized meal plans.",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(4, 0))

        control_card = ttk.Frame(root_frame, style="Card.TFrame", padding=18)
        control_card.pack(fill=tk.X, pady=(0, 16))

        controls_grid = ttk.Frame(control_card, style="Card.TFrame")
        controls_grid.pack(fill=tk.X)

        ttk.Label(controls_grid, text="Menu directory", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        dir_entry = ttk.Entry(controls_grid, textvariable=self.menu_dir_var)
        dir_entry.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self._interactive_widgets.append(dir_entry)

        browse_button = ttk.Button(controls_grid, text="Browse…", style="Secondary.TButton", command=self._choose_directory)
        browse_button.grid(row=1, column=3, padx=(12, 0))
        self._interactive_widgets.append(browse_button)

        load_button = ttk.Button(
            controls_grid,
            text="Load Menus",
            style="Accent.TButton",
            command=self._load_menus_clicked,
        )
        load_button.grid(row=1, column=4, padx=(12, 0))
        self._interactive_widgets.append(load_button)

        controls_grid.columnconfigure(0, weight=1)

        notebook_card = ttk.Frame(root_frame, style="Card.TFrame", padding=18)
        notebook_card.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(notebook_card)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Ask tab (MenuRAG search).
        ask_tab = ttk.Frame(notebook, padding=2, style="Card.TFrame")
        notebook.add(ask_tab, text="Ask Questions")

        ask_header = ttk.Frame(ask_tab, style="Card.TFrame")
        ask_header.pack(fill=tk.X, padx=4, pady=(4, 8))

        ttk.Label(ask_header, text="Ask the assistant", style="Section.TLabel").pack(anchor=tk.W)
        ttk.Label(
            ask_header,
            text="Type a nutrition question, station lookup, or meal planning prompt.",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        question_row = ttk.Frame(ask_tab, style="Card.TFrame")
        question_row.pack(fill=tk.X, padx=4, pady=(0, 8))

        self.question_entry = ttk.Entry(question_row)
        self.question_entry.insert(0, "What is a high protein food?")
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._interactive_widgets.append(self.question_entry)

        ask_button = ttk.Button(
            question_row,
            text="Ask",
            command=self._ask_clicked,
            style="Accent.TButton",
        )
        ask_button.pack(side=tk.LEFT, padx=(12, 0))
        self._interactive_widgets.append(ask_button)

        controls = ttk.Frame(ask_tab, style="Card.TFrame")
        controls.pack(fill=tk.X, padx=4, pady=(0, 10))

        ttk.Label(controls, text="Top results:", style="Subtitle.TLabel").pack(side=tk.LEFT)
        self.topk_spinbox = ttk.Spinbox(controls, from_=1, to=15, width=4, textvariable=self.topk_var)
        self.topk_spinbox.pack(side=tk.LEFT, padx=(4, 16))
        self._interactive_widgets.append(self.topk_spinbox)

        high_protein_button = ttk.Button(
            controls, text="High protein picks", style="Secondary.TButton", command=self._high_protein_clicked
        )
        high_protein_button.pack(side=tk.LEFT, padx=(8, 0))
        self._interactive_widgets.append(high_protein_button)

        output_frame = ttk.Frame(ask_tab, style="Card.TFrame")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        self.output_text = tk.Text(
            output_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=(".AppleSystemUIFont", 12),
            background=surface_bg,
            foreground=text_primary,
            insertbackground=text_primary,
            highlightthickness=1,
            highlightcolor=theme.get("subtle_border", "#d2d2d7"),
            highlightbackground=theme.get("subtle_border", "#d2d2d7"),
            bd=0,
            relief=tk.FLAT,
            padx=14,
            pady=14,
        )
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.configure(style="Card.Vertical.TScrollbar")

        # Analytics tab.
        analytics_tab = ttk.Frame(notebook, padding=2, style="Card.TFrame")
        notebook.add(analytics_tab, text="Nutrition Dashboard")

        filters_frame = ttk.LabelFrame(analytics_tab, text="Filters", style="Card.TLabelframe", padding=16)
        filters_frame.pack(fill=tk.X, expand=False, pady=(0, 12), padx=4)

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

        summary_frame = ttk.LabelFrame(analytics_tab, text="Summary", style="Card.TLabelframe", padding=16)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=4)

        self.dashboard_text = tk.Text(
            summary_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=(".AppleSystemUIFont", 12),
            background=surface_bg,
            foreground=text_primary,
            insertbackground=text_primary,
            highlightthickness=1,
            highlightcolor=theme.get("subtle_border", "#d2d2d7"),
            highlightbackground=theme.get("subtle_border", "#d2d2d7"),
            bd=0,
            relief=tk.FLAT,
            padx=14,
            pady=14,
        )
        self.dashboard_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Status bar.
        status_bar = ttk.Label(root_frame, textvariable=self.status_var, anchor=tk.W, style="Status.TLabel")
        status_bar.pack(fill=tk.X, pady=(12, 0))

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
