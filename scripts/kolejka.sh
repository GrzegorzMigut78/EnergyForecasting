#!/usr/bin/env bash
# Kolejka zadań EnergyForecasting (WSL/Linux).
#
# Wykonuje po kolei pliki *.job z katalogu kolejki (domyślnie C:\EnergyForecasting\kolejka).
# Każdy plik zawiera jedno polecenie w pierwszej linii. Dozwolone są WYŁĄCZNIE:
#   update                         git pull + instalacja zależności z uv.lock
#   test                           testy projektu
#   prognozowanie <podpolecenie> configs/... [opcje]
#       podpolecenia: check-data, identify, run, all, combine, evaluate
# Inne polecenia, znaki spoza [A-Za-z0-9_./:=,+-], ścieżki bezwzględne i ".." są odrzucane.
# Nic nie jest przekazywane do powłoki (bez eval): argumenty trafiają wprost do programu.
#
# Uruchomienie (okno zostaje otwarte):  bash ~/EnergyForecasting/scripts/kolejka.sh
# Zatrzymanie: Ctrl+C (bieżące zadanie zostanie przerwane i trafi do failed/).
set -uo pipefail

REPO="${REPO:-$HOME/EnergyForecasting}"
Q="${KOLEJKA:-/mnt/c/EnergyForecasting/kolejka}"
SAFE='^[A-Za-z0-9_./:=,+-]+$'
ALLOWED_SUB=" check-data identify run all combine evaluate "
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$Q/done" "$Q/failed" "$Q/logs"

validate() {
  local t
  for t in "$@"; do
    [[ "$t" =~ $SAFE ]] || return 1
    [[ "$t" == *..* || "$t" == /* ]] && return 1
  done
  return 0
}

execute() { # wykonuje zweryfikowane polecenie w katalogu projektu
  cd "$REPO" || return 1
  case "${1:-}" in
    update)
      [ $# -eq 1 ] || return 2
      git pull --ff-only && uv sync --frozen --extra gefcom
      ;;
    test)
      [ $# -eq 1 ] || return 2
      uv run pytest -q -p no:warnings
      ;;
    prognozowanie)
      [[ "$ALLOWED_SUB" == *" ${2:-} "* ]] || { echo "ODRZUCONE: podpolecenie '${2:-}'"; return 2; }
      [[ "${3:-}" == configs/* ]] || { echo "ODRZUCONE: konfiguracja spoza configs/"; return 2; }
      uv run prognozowanie "${@:2}"
      ;;
    *)
      echo "ODRZUCONE: nieznane polecenie '${1:-}'"
      return 2
      ;;
  esac
}

run_job() {
  local f="$1" name line log rc pid
  name=$(basename "$f" .job)
  line=$(head -n 1 "$f" | tr -d '\r')
  log="$Q/logs/$name.log"
  local -a args
  read -ra args <<< "$line"
  echo "=== start $(date -Is): $line" >> "$log"
  if ! validate "${args[@]}"; then
    echo "ODRZUCONE: niedozwolone znaki lub ścieżki" >> "$log"
    mv "$f" "$Q/failed/"
    return
  fi
  echo "$name" > "$Q/current"
  execute "${args[@]}" >> "$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do # sygnał życia także podczas długiego zadania
    date -Is > "$Q/heartbeat"
    sleep 30
  done
  wait "$pid"
  rc=$?
  echo "=== koniec $(date -Is): kod $rc" >> "$log"
  if [ "$rc" -eq 0 ]; then mv "$f" "$Q/done/"; else mv "$f" "$Q/failed/"; fi
  rm -f "$Q/current"
}

echo "Kolejka działa: $Q (projekt: $REPO). Zostaw to okno otwarte; Ctrl+C kończy."
while true; do
  date -Is > "$Q/heartbeat"
  shopt -s nullglob
  jobs_found=("$Q"/*.job)
  shopt -u nullglob
  if [ ${#jobs_found[@]} -gt 0 ]; then
    IFS=$'\n' sorted=($(printf '%s\n' "${jobs_found[@]}" | sort))
    unset IFS
    for f in "${sorted[@]}"; do
      echo "$(date +%T) zadanie: $(basename "$f")"
      run_job "$f"
      echo "$(date +%T) zakończone: $(basename "$f")"
    done
  fi
  sleep 20
done
