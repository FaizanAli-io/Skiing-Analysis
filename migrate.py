"""Apply database migrations. Run once per deploy, before restarting services:

    python migrate.py
"""

import logging

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from services.bootstrap import run_migrations  # noqa: E402

if __name__ == "__main__":
    version = run_migrations()
    print(f"Migrations complete. Schema version {version}.")
