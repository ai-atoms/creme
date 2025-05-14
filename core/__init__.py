# core/__init__.py

import os, sys

# until package ready
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# initialize package
print(f'[creme] initializing core package at {root_dir}...')

