"""
Progress reporting utilities
"""
import sys

def show_progress(operation, percent=0, status="", final=False):
    """
    Display a progress indicator in the console.
    
    Args:
        operation: String describing the current operation
        percent: Progress percentage (0-100)
        status: Additional status message
        final: Whether this is the final update for this operation
    """
    bar_length = 20
    filled_length = int(bar_length * percent / 100)
    bar = '■' * filled_length + '□' * (bar_length - filled_length)
    
    if final:
        sys.stdout.write(f"\r{operation}: [{bar}] {percent:.1f}% - {status} ✓ Completed\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r{operation}: [{bar}] {percent:.1f}% - {status}")
        sys.stdout.flush()
