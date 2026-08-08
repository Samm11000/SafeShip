"""python -m prototypes.structural [base] [head]"""
import json
import sys

from .extract import extract

base = sys.argv[1] if len(sys.argv) > 1 else "HEAD~1"
head = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
print(json.dumps(extract(base, head), indent=2))
