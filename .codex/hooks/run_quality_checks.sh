#!/usr/bin/env bash
set -u

run_check() {
  local label="$1"
  shift

  if ! output=$("$@" 2>&1); then
    printf '{"continue":false,"stopReason":"%s failed","systemMessage":%s}\n' "$label" "$(printf '%s' "$output" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
    return 1
  fi

  return 0
}

run_check "Ruff lint" ruff check . || exit 0
run_check "Ruff format check" ruff format --check . || exit 0
run_check "pytest" pytest || exit 0

printf '{"continue":true,"systemMessage":"All quality checks passed"}\n'
