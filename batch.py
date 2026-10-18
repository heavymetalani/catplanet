#!/usr/bin/env python3
"""
Process every video in data/inbox/:  python batch.py [options]

This is a shortcut; the code lives in catplanet/batch.py.
Run `python batch.py --help` for every option.
"""
from catplanet.batch import main

if __name__ == "__main__":
    raise SystemExit(main())
