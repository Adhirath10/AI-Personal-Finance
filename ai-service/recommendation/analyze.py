import argparse
import json
import logging
import sqlite3
import sys

from engine import RecommendationEngine


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze one user's financial position for recommendations."
    )
    parser.add_argument("--user-id", required=True, help="User ID to analyze.")
    parser.add_argument(
        "--database-path",
        default=None,
        help="Existing SQLite database path; defaults to the project database.",
    )
    parser.add_argument(
        "--analysis-month",
        help="Month to analyze in YYYY-MM format; defaults to the latest transaction month.",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = RecommendationEngine(database_path=args.database_path).analyze_user(
            args.user_id, analysis_month=args.analysis_month
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (FileNotFoundError, ValueError, sqlite3.Error, OSError) as exc:
        logging.getLogger(__name__).error("Financial analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
