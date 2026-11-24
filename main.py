#!/usr/bin/env python3
"""Entry point for the Fly-in drone simulation."""

import os
import sys

from fly_in2.cli import main

if __name__ == "__main__":
    try:
        code = main()
        sys.stdout.flush()   # surface a closed pipe here, not at exit
    except KeyboardInterrupt:
        # Ctrl+C: exit quietly with the shell's SIGINT status (128 + 2).
        sys.stderr.write("\n")
        raise SystemExit(130)
    except BrokenPipeError:
        # The reader closed the pipe (e.g. `| head`). Python flushes stdout
        # again on exit, so point it at devnull to avoid a second error
        # (Python docs, signal module, "Note on SIGPIPE").
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(1)
    raise SystemExit(code)
