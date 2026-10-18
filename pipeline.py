#!/usr/bin/env python3
"""
Run one video through the pipeline:  python pipeline.py <video> [options]

This is a shortcut; the code lives in catplanet/pipeline.py.
Run `python pipeline.py --help` for every option.
"""
from catplanet.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
