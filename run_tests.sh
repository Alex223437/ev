#!/usr/bin/env bash
# Spustí testy všech úloh, každou v samostatném procesu.
#
# Úlohy jsou záměrně samostatné a používají stejné názvy modulů (ga, problems,
# experiments), takže je nelze sesbírat jedním během pytestu.

set -u

python_bin="${PYTHON:-}"
if [ -z "$python_bin" ]; then
    if [ -x ".venv/bin/python" ]; then
        python_bin=".venv/bin/python"
    else
        python_bin="python3"
    fi
fi

status=0
for tests in */tests; do
    [ -d "$tests" ] || continue
    echo "== ${tests%/tests} =="
    "$python_bin" -m pytest "$tests" -q || status=1
done

exit $status
