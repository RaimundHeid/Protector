#!/bin/sh

pkill -f "spsa_.*\.py" && echo "Python SPSA process killed." || echo "No Python SPSA process found."
pkill -f "cutechess-cli"  && echo "cutechess-cli killed."      || echo "No cutechess-cli found."
