#!/usr/bin/env python
"""Run tests/test_imports.py from the correct project root directory."""
import os
import sys

# Change to the correct project root directory
project_root = '/home/darthmaverus/projects/ambiguity2/'
os.chdir(project_root)

# Add current directory to path if needed
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Execute the test file
exec(open('tests/test_imports.py').read())
