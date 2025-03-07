#!/usr/bin/env python3
"""
Helper script to run the database synchronization utility.
"""

import os
import sys
import subprocess

def main():
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the path to sync_db.py
    sync_script = os.path.join(current_dir, "sync_db.py")
    
    print("Running database synchronization...")
    
    # Make sure sync_db.py is executable
    if not os.access(sync_script, os.X_OK):
        os.chmod(sync_script, 0o755)
    
    # Run the sync script
    try:
        result = subprocess.run([sys.executable, sync_script], check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running sync: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
