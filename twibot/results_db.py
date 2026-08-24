"""SQLite-based results storage with checkpointing for experiment runs."""

import json
import sqlite3
import hashlib
import datetime
from typing import List, Dict, Any, Optional


class AblationResultsDB:
    """SQLite-based storage for experiment results with checkpointing."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE,
                    dataset TEXT,
                    model_name TEXT,
                    model_type TEXT,
                    selection_mode TEXT,
                    max_tweets INTEGER,
                    status TEXT DEFAULT 'pending',
                    accuracy REAL,
                    f1_score REAL,
                    total_samples INTEGER,
                    correct_samples INTEGER,
                    error_count INTEGER DEFAULT 0,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    UNIQUE(dataset, model_name, selection_mode, max_tweets)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    user_id TEXT,
                    username TEXT,
                    gold_label TEXT,
                    pred_label TEXT,
                    correct INTEGER,
                    error TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_experiment
                ON predictions(experiment_id)
            """)
            conn.commit()

    def get_experiment_id(
        self,
        dataset: str,
        model_name: str,
        selection_mode: str,
        max_tweets: int
    ) -> str:
        """Generate a unique experiment ID."""
        key = f"{dataset}_{model_name}_{selection_mode}_{max_tweets}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def experiment_exists(self, experiment_id: str) -> bool:
        """Check if an experiment has been completed."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT status FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchone()
            return result is not None and result[0] == 'completed'

    def start_experiment(
        self,
        experiment_id: str,
        dataset: str,
        model_name: str,
        model_type: str,
        selection_mode: str,
        max_tweets: int
    ):
        """Mark an experiment as started."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments
                (experiment_id, dataset, model_name, model_type, selection_mode,
                 max_tweets, status, started_at)
                VALUES (?, ?, ?, ?, ?, ?, 'running', ?)
            """, (experiment_id, dataset, model_name, model_type, selection_mode,
                  max_tweets, datetime.datetime.now().isoformat()))
            conn.commit()

    def add_prediction(
        self,
        experiment_id: str,
        user_id: str,
        username: str,
        gold_label: str,
        pred_label: Optional[str],
        error: Optional[str] = None
    ):
        """Add a single prediction result."""
        correct = 1 if pred_label and gold_label == pred_label else 0
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO predictions
                (experiment_id, user_id, username, gold_label, pred_label, correct, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, user_id, username, gold_label, pred_label, correct, error))
            conn.commit()

    def complete_experiment(
        self,
        experiment_id: str,
        accuracy: float,
        f1_score: float,
        total_samples: int,
        correct_samples: int,
        error_count: int
    ):
        """Mark an experiment as completed with final metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE experiments
                SET status = 'completed', accuracy = ?, f1_score = ?,
                    total_samples = ?, correct_samples = ?, error_count = ?,
                    completed_at = ?
                WHERE experiment_id = ?
            """, (accuracy, f1_score, total_samples, correct_samples, error_count,
                  datetime.datetime.now().isoformat(), experiment_id))
            conn.commit()

    def fail_experiment(self, experiment_id: str, error_message: str):
        """Mark an experiment as failed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE experiments
                SET status = 'failed', error_message = ?, completed_at = ?
                WHERE experiment_id = ?
            """, (error_message, datetime.datetime.now().isoformat(), experiment_id))
            conn.commit()

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all completed experiment results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            results = conn.execute("""
                SELECT * FROM experiments
                WHERE status = 'completed'
                ORDER BY dataset, model_type, model_name, selection_mode, max_tweets
            """).fetchall()
            return [dict(r) for r in results]

    def export_to_json(self, output_path: str):
        """Export all results to a JSON file."""
        results = self.get_all_results()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    def export_to_csv(self, output_path: str):
        """Export all results to a CSV file."""
        import csv
        results = self.get_all_results()
        if not results:
            return

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
