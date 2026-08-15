#!/bin/bash
# Autonomous GPU-vs-CPU llama-bench driver for the NP2 (wireless adb).
# Waits for each model file to land on the phone, then benches GPU (-ngl 99)
# and CPU (-ngl 0) on the SAME file, with thermal + battery gating and TG pings.
export PATH="$PATH:$HOME/Library/Android/sdk/platform-tools"
A(){ adb -s $PHONE "$@"; }
D=/data/local/tmp
LOG=~/np2-aibench2/results-gpu.txt
: > "$LOG"
log(){ echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
RUNV="cd $D && LD_LIBRARY_PATH=/vendor/lib64"

battery(){ A shell 'dumpsys battery 2>/dev/null' | tr -d '\r' | awk '/level:/{print $2; exit}'; }
thermal(){ A shell 'dumpsys thermalservice 2>/dev/null' | tr -d '\r' | awk '/Thermal Status:/{print $3; exit}'; }

cooldown(){  # wait until thermal status <=1 (max 5 min) + fixed floor
  python3 -c "import time;time.sleep(45)"
  for i in $(seq 1 10); do
    t=$(thermal); [ -z "$t" ] && break
    [ "$t" -le 1 ] 2>/dev/null && break
    log "  (thermal status=$t, cooling...)"; python3 -c "import time;time.sleep(30)"
  done
}

wait_model(){  # $1=fname $2=mac_path ; waits until phone size == CURRENT mac size (stable)
  for i in $(seq 1 720); do   # up to ~3h
    local want got want2
    want=$(stat -f%z "$2" 2>/dev/null)
    got=$(A shell "stat -c%s $D/$1 2>/dev/null" | tr -d '\r')
    if [ -n "$want" ] && [ "$got" = "$want" ]; then
      python3 -c "import time;time.sleep(10)"
      want2=$(stat -f%z "$2" 2>/dev/null)
      [ "$want" = "$want2" ] && return 0   # mac file stable + phone matches = done
    fi
    python3 -c "import time;time.sleep(15)"
  done
  return 1
}

bench_one(){  # $1=fname $2=label $3=ngl $4=extra
  local out
  out=$(A shell "$RUNV ./llama-bench -m $D/$1 -p 512 -n 128 -ngl $3 $4 -r 3 2>/dev/null" </dev/null | tr -d '\r' | grep -E "\| *pp512|\| *tg128")
  if [ -z "$out" ]; then
    # rerun capturing stderr tail for diagnosis
    err=$(A shell "$RUNV ./llama-bench -m $D/$1 -p 512 -n 128 -ngl $3 $4 -r 1 2>&1" </dev/null | tr -d '\r' | tail -4)
    log "  FAILED [$2 ngl=$3]: $err"
    echo "FAIL"
    return
  fi
  echo "$out" | tee -a "$LOG" >/dev/null
  local pp tg
  pp=$(echo "$out" | grep pp512 | awk -F'|' '{print $(NF-1)}' | xargs)
  tg=$(echo "$out" | grep tg128 | awk -F'|' '{print $(NF-1)}' | xargs)
  echo "pp=$pp tg=$tg"
}

run_model(){  # $1=fname $2=label $3=mac_path $4=cpu_threads
  log ""; log "########## $2 ##########"
  if ! wait_model "$1" "$3"; then log "  model never landed; skipping"; tg "⚠️ $2: model never landed on phone"; return; fi
  b=$(battery); log "  battery=$b% thermal=$(thermal)"
  if [ -n "$b" ] && [ "$b" -lt 15 ] 2>/dev/null; then tg "🪫 NP2 battery ${b}% — pausing bench until charged"; while [ "$(battery)" -lt 30 ] 2>/dev/null; do python3 -c "import time;time.sleep(120)"; done; fi
  A shell 'input keyevent 223' >/dev/null 2>&1   # screen off
  log "--- GPU (ngl=99) ---"
  g=$(bench_one "$1" "$2" 99 "")
  log "  GPU: $g"
  cooldown
  log "--- CPU (ngl=0, t=$4) ---"
  c=$(bench_one "$1" "$2" 0 "-t $4")
  log "  CPU: $c"
  cooldown
  tg "📊 $2
GPU(ngl99): $g
CPU(t=$4): $c"
}

log "### NP2 GPU-vs-CPU BENCH (OpenCL Adreno 730) ###"
tg "🚀 NP2 GPU benchmark round starting (OpenCL on Adreno 730 confirmed working). Models bench as they land."

run_model qwen35-2b-q40.gguf   "Qwen3.5-2B Q4_0 (Adreno-optimal)" ~/np2-aibench2/qwen35-2b-q40.gguf 4
run_model qwen35-2b-q4km.gguf  "Qwen3.5-2B Q4_K_M"                ~/np2-aibench2/qwen35-2b-q4km.gguf 4
run_model qwen35-9b-q40.gguf   "Qwen3.5-9B Q4_0"                  ~/np2-aibench2/qwen35-9b-q40.gguf 4
run_model llama31-8b-q4km.gguf "Llama-3.1-8B Q4_K_M"              ~/np2-aibench2/llama31-8b-q4km.gguf 4

log ""; log "### TEXT BENCH COMPLETE ###"
SUM=$(grep -E "##########|GPU:|CPU:" "$LOG" | sed 's/^[0-9:]* *//')
tg "🏁 NP2 GPU-vs-CPU text benchmarks done:
$SUM"
