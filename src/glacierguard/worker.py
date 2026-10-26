"""
GlacierGuard-AI Background Surveillance Worker
==============================================
Standalone, decoupled background worker process that periodically discovers
new Sentinel-1 and Sentinel-2 satellite products, runs the processing pipeline,
and updates SQLite persistence.

Can be run in two modes:
1. One-off batch execution:
   ``python -m glacierguard.worker --once``
2. Continuous scheduled surveillance:
   ``python -m glacierguard.worker --interval-hours 6.0``
"""

import argparse
import datetime
import logging
import os
import signal
import sys
import time
from typing import List, Optional

from glacierguard.config import Settings, default_settings
from glacierguard.db import init_db, seed_from_qa_csv
from glacierguard.models import PipelineRun
from glacierguard.pipeline import ProcessingPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("glacierguard.worker")


class MonitoringWorker:
    """Surveillance scheduler running independently of the Streamlit UI."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or default_settings
        self.pipeline = ProcessingPipeline(self.settings)
        self._running = False

    def setup(self) -> None:
        """Initialize database tables and seed baseline if database is empty."""
        logger.info("Initializing GlacierGuard database at: %s", self.settings.db_path)
        init_db(self.settings.db_path)

        # Seed from baseline CSV if empty
        seeded = seed_from_qa_csv(
            csv_path=self.settings.qa_csv_path,
            db_path=self.settings.db_path,
        )
        if seeded > 0:
            logger.info("Successfully seeded database with %d baseline lake records.", seeded)
        else:
            logger.info("Database already contains data or baseline CSV not found.")

    def run_once(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[PipelineRun]:
        """Execute one complete surveillance cycle across all configured lakes."""
        self.setup()
        logger.info("Starting surveillance cycle for %d lakes...", len(self.settings.list_lake_ids()))
        runs = self.pipeline.run_all_lakes(start_date=start_date, end_date=end_date)
        successes = sum(1 for r in runs if r.status == "SUCCESS")
        logger.info(
            "Cycle completed: %d/%d lakes processed successfully.",
            successes,
            len(runs),
        )
        return runs

    def start(self, interval_hours: Optional[float] = None) -> None:
        """Run continuous surveillance loop with graceful shutdown handling."""
        hours = interval_hours or self.settings.poll_interval_hours
        sleep_sec = max(60.0, hours * 3600.0)
        self._running = True

        def _handle_shutdown(signum, frame):
            logger.info("Received shutdown signal (%s). Stopping worker...", signum)
            self._running = False

        # Register signal handlers for clean exit
        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)

        logger.info(
            "GlacierGuard worker started. Polling every %.1f hours (%.0f seconds).",
            hours,
            sleep_sec,
        )

        while self._running:
            try:
                cycle_start = time.time()
                self.run_once()
                elapsed = time.time() - cycle_start

                remaining_sleep = max(0.0, sleep_sec - elapsed)
                logger.info(
                    "Cycle took %.1f s. Sleeping for %.1f minutes until next poll.",
                    elapsed,
                    remaining_sleep / 60.0,
                )

                # Sleep in short increments to respond quickly to SIGINT/SIGTERM
                sleep_chunks = int(remaining_sleep / 5.0)
                for _ in range(sleep_chunks):
                    if not self._running:
                        break
                    time.sleep(5.0)

            except Exception as exc:
                logger.error("Error during surveillance cycle: %s", exc, exc_info=True)
                time.sleep(60.0)  # Short pause before retry on fatal exception

        logger.info("GlacierGuard worker cleanly stopped.")


def main():
    """Command line entrypoint for the surveillance worker."""
    parser = argparse.ArgumentParser(
        description="GlacierGuard-AI Autonomous Surveillance Worker"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single surveillance discovery cycle and exit.",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=None,
        help="Polling interval in hours (overrides settings).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Custom SQLite database file path.",
    )
    args = parser.parse_args()

    settings = Settings()
    if args.db_path:
        from pathlib import Path
        settings.db_path = Path(args.db_path)

    worker = MonitoringWorker(settings=settings)

    if args.once:
        worker.run_once()
    else:
        worker.start(interval_hours=args.interval_hours)


if __name__ == "__main__":
    main()
