"""Entry point.

    python main.py --invoice_path=data/invoices/invoice_1001.txt

The brief names this file and this flag, so this file exists and this flag is spelled that
way. It is a shim on purpose -- the logic lives in `galatiq.cli`, where it can be imported
and tested, rather than in a script at the repository root.
"""

from galatiq.cli import app

if __name__ == "__main__":
    app()
