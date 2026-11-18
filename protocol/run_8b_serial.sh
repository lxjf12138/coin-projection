#!/usr/bin/env bash
# Experiments for the ARR revision, strictly serial on ONE GPU (SCALE=8b: the rented card; SCALE=smoke: the
# local structural fixture with small models, whose outputs are never paper numbers).
#
# Protocol for every comparison stage: fixed-length stegotexts (--stop fixed).  Each text has T tokens (plus at
# most 8 to reach a UTF-8 boundary) and embeds a 64*T-bit random stream with the rotation range coder; prompts,
# keys and streams are shared by all methods, so the length never depends on the message.
#
# Fairness: one job at a time (flock), refuse to start while any other process holds the GPU, a fixed CPU thread
# count (from the cgroup CPU quota), environment recorded at every stage, a validator after every chunk (the run
# stops at the first result no analysis could repair).  Every job is resumable (--skip-done / per-file outputs).
#
# Usage:   run_8b_serial.sh STAGE [STAGE ...]
#          R0 checks + smoke   R9 equivalence   R1 e0 atlas   F figure data (Fig 1/3, Llama-3.1-8B)
#          R5 gof-states   R2 main table (7 methods)   R4 LAS   T timing (lean, interleaved, one machine)
#          TB per-step CPU microbenchmark on the R5 states   R9b on-trajectory decomposition (needs R2)
#          R6 long texts   R8 temperature / nucleus   R8c long prompts
#          R8s Stepwise (k=256) added to the R8 and R8c files   R10 coders: Discop and arithmetic coding in
#             place of range coding (Native, SyncPool, coin; the first N_ROB texts of R2)
#          R11 ReTokSync on Discop (the first N_RTS texts of R2's prompts, keys and streams)
#          P  CPU only, after R2 R4 R5: GoF draws, Native fallback, non-canonical counts, Stepwise diagnosis,
#             tokenizer origins
#          R1x e0 atlas of each model on the other two languages (forward only)
#          R2r receivers of SyncPool and coin projection on the texts of R2 that RECV_FIRST left undecoded
#             (same prompts, keys and streams; own file, compared with the R2 records afterwards)
#          plan order: R0 R9 R1 F R5 R2 R4 T TB R9b R6 R8 R8c P, then R8s R10 R11, then R1x R2r
# Env:     SCALE (8b | smoke), M (models root), D (C4 prompts dir), OUT (results), STEPWISE_SRC, LAS_RELEASE, PY,
#          THREADS (default: the cgroup quota, at most 8), CPUSET (optional taskset list), N (texts per cell, 100),
#          N_RTS (ReTokSync texts per cell, 50),
#          T (128), MWIS (exact | official), RECV_FIRST (20: receivers on the first 20 texts of each cell, Native
#          on all; -1 = all texts), N_TIME (texts per cell in T), P_LONG_W / P_LONG_C (R8c prompt sizes)
set -u
P="$(cd "$(dirname "$0")" && pwd)"
SCALE=${SCALE:-8b}
if [ "$SCALE" = smoke ]; then  # structural run: every stage, a few texts each
  : "${N:=2}" "${CHUNK:=2}" "${T:=16}" "${T_LONG:=48}" "${N_LONG:=2}" "${N_ROB:=2}" "${N_TIME:=1}" "${N_RTS:=2}"
  : "${RECV_FIRST:=1}" "${E0_N:=2}" "${E0_T:=8}" "${GOF_CTX:=2}" "${GOF_DRAWS:=200}" "${EQUIV_N:=4}" "${E2B_MAX:=2}"
  : "${KS:=16 256 0}" "${E2B_KS:=256 0}" "${TB_STATES:=2}" "${TB_REPS:=3}" "${CHECK_LAWS:=2}"
fi
M=${M:-$PWD/models}          # one directory per model, named as in SPECS (e.g. Meta-Llama-3.1-8B)
D=${D:-$PWD/data/c4}         # c4_{en,zh,ja}_1000.jsonl from data/fetch_c4.py
OUT=${OUT:-$P/runs_$SCALE}
N=${N:-100}
CHUNK=${CHUNK:-25}
T=${T:-128}
T_LONG=${T_LONG:-512}
N_LONG=${N_LONG:-30}
N_ROB=${N_ROB:-50}
N_RTS=${N_RTS:-50}
N_TIME=${N_TIME:-10}
E0_N=${E0_N:-50}
E0_T=${E0_T:-64}
GOF_CTX=${GOF_CTX:-20}
GOF_DRAWS=${GOF_DRAWS:-5000}
GOF_KS=${GOF_KS:-16 256 0}
EQUIV_N=${EQUIV_N:-24}
E2B_MAX=${E2B_MAX:-15}
E2B_KS=${E2B_KS:-64 1024 4096 0}
KS=${KS:-16 64 256 1024 4096 0}
TB_STATES=${TB_STATES:-10}
TB_REPS=${TB_REPS:-20}
CHECK_LAWS=${CHECK_LAWS:-12}
P_LONG_W=${P_LONG_W:-100}      # R8c: prompt of 100 words (EN) / 200 characters (ZH, JA)
P_LONG_C=${P_LONG_C:-200}
MWIS=${MWIS:-exact}
RECV_FIRST=${RECV_FIRST:-20}   # receiver on the first RECV_FIRST texts of each cell; Native on every text
quota_threads() {  # whole cores granted by the cgroup quota (cpu.max "quota period"), at most 8, at least 1
  local q p
  { read -r q p < /sys/fs/cgroup/cpu.max; } 2>/dev/null || { nproc; return; }
  if [ "$q" = max ]; then nproc; else echo $(( q / p > 0 ? q / p : 1 )); fi
}
if [ -z "${THREADS:-}" ]; then THREADS=$(quota_threads); [ "$THREADS" -gt 8 ] && THREADS=8; fi
CPUSET=${CPUSET:-}
PY=${PY:-python}
export STEPWISE_SRC=${STEPWISE_SRC:-}     # <Consistency clone>/src   (required, see README)
export LAS_RELEASE=${LAS_RELEASE:-}      # <LAS release clone>       (only stage R4 / methods including las)
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS TOKENIZERS_PARALLELISM=false
case "$SCALE" in
  8b) SPECS=("llama31_8b Meta-Llama-3.1-8B en words" "qwen3_8b_base Qwen3-8B-Base zh chars"
             "llmjp3_7p2b llm-jp-3-7.2b ja chars") ;;
  smoke) SPECS=("smoke_llama Llama-3.2-3B en words" "smoke_qwen Qwen3-4B-Base zh chars"
                "smoke_llmjp llm-jp-3-3.7b ja chars") ;;
  *) echo "SCALE must be 8b or smoke" >&2; exit 1 ;;
esac
FIG_MODEL=${FIG_MODEL:-${SPECS[0]}}
METHODS="native basic mwis stepwise syncpool coin hubdrop"
STREAM=(--coder rrc --stop fixed --T "$T" --recv-first "$RECV_FIRST")
F_CONTEXTS=("I'll call you this" "I'll see you this" "Let's meet again this" "We can talk about it this"
            "See you next")

mkdir -p "$OUT/logs" "$OUT/fig" || exit 1
exec 9>"$OUT/.serial.lock"
flock -n 9 || { echo "another run_8b_serial.sh holds $OUT/.serial.lock" >&2; exit 2; }
cd /tmp || exit 1

guard() {  # refuse to start a job while anything else uses the GPU
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -c .)
  if [ "$busy" -gt 0 ]; then
    echo "GPU busy, refusing to start:" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
    exit 3
  fi
}

record_env() {  # $1 = stage
  local f="$OUT/logs/env_$1_$(date +%Y%m%d_%H%M%S).txt"
  {
    date '+%F %T'; hostname; uname -r
    nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,persistence_mode --format=csv
    lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|MHz'
    echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)  nproc: $(nproc)"
    grep -E 'nr_throttled|throttled_usec' /sys/fs/cgroup/cpu.stat 2>/dev/null
    free -g | head -2
    echo "SCALE=$SCALE THREADS=$THREADS CPUSET=${CPUSET:-none} N=$N T=$T KS=$KS MWIS=$MWIS RECV_FIRST=$RECV_FIRST N_TIME=$N_TIME"
    $PY -I -c 'import torch, transformers, numpy, cryptography; print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "numpy", numpy.__version__, "cryptography", cryptography.__version__)'
    for r in "$P" "$STEPWISE_SRC" "$LAS_RELEASE"; do echo "$r $(git -C "$r" rev-parse HEAD 2>/dev/null || echo no-git)"; done
    sha256sum "$P"/*.py "$P"/*.sh "$P"/checks/*.py "$P"/../analysis/*.py "$P"/../plots/*.py
  } > "$f" 2>&1
  echo "env recorded in $f"
}

job() {  # job NAME CMD...   (guarded, logged; pinned only when CPUSET is set)
  local name="$1"; shift
  guard
  echo "=== $name start $(date '+%F %T')"
  if [ -n "$CPUSET" ]; then taskset -c "$CPUSET" "$@"; else "$@"; fi 2>&1 | grep -v '^Loading'
  local rc=${PIPESTATUS[0]}
  echo "=== $name end $(date '+%F %T') rc=$rc"
  return "$rc"
}

validate() {  # validate [--one-machine] FILE...: stop the stage at the first irreparable result
  $PY -I "$P/checks/validate_run.py" "$@"
}

each() {  # each FN: call FN tag dir lang unit for the three models
  local fn="$1" spec
  for spec in "${SPECS[@]}"; do
    # shellcheck disable=SC2086
    $fn $spec || return $?
  done
}

chunked() {  # chunked FN: rounds of CHUNK texts, all three models per round (balanced partials)
  local fn="$1" s c spec
  for ((s = 0; s < N; s += CHUNK)); do
    c=$(( N - s < CHUNK ? N - s : CHUNK ))
    for spec in "${SPECS[@]}"; do
      # shellcheck disable=SC2086
      $fn $spec "$s" "$c" || return $?
    done
  done
}

c4job() {  # c4job NAME tag dir lang unit ARGS...: c4proto.py on this model's prompts, as a job
  local name=$1 tag=$2 dir=$3 lang=$4 unit=$5; shift 5
  job "$name" $PY -I -u "$P/c4proto.py" --model-dir "$M/$dir" --tag "$tag" --prompts "$D/c4_${lang}_1000.jsonl" \
    --unit "$unit" "$@"
}

long_units() { [ "$1" = words ] && echo "$P_LONG_W" || echo "$P_LONG_C"; }

# ---------------------------------------------------------------------------
r0_one() {  # tokenizer checks (CPU), bit identity of every code path, then short stegotexts per method incl. LAS
  local tag=$1 dir=$2 lang=$3 unit=$4
  job "R0 split $tag" $PY -I -u "$P/checks/check_coin_split.py" --protocol-dir "$P" --model-dir "$M/$dir" --laws 10 || return $?
  job "R0 sparse-dense $tag" $PY -I -u "$P/checks/check_sparse_dense.py" --model-dir "$M/$dir" \
    --laws "$CHECK_LAWS" --real || return $?
  local prompt
  case "$lang" in en) prompt="The history of the" ;; zh) prompt="中国古代的" ;; *) prompt="日本の伝統的な" ;; esac
  job "R0 stream-cpu $tag" $PY -I -u "$P/checks/check_stream_cpu.py" --model-dir "$M/$dir" --prompt "$prompt" \
    --T 16 --samples 1 --k 16 256 0 --coders rrc --mwis-solver "$MWIS" || return $?
  c4job "R0 smoke $tag" "$@" --methods $METHODS las --k 16 0 --samples 2 --coder rrc --stop fixed --T 32 \
    --recv-first -1 --mwis-solver "$MWIS" --proj-also --out "$OUT/smoke_$tag.jsonl" \
    --steps-out "$OUT/smoke_steps_$tag.jsonl" || return $?
  validate "$OUT/smoke_$tag.jsonl"
}
r1_one() {
  job "R1 e0 $1" $PY -I -u "$P/e0_atlas.py" --model-dir "$M/$2" --tag "$1" --prompts "$D/c4_${3}_1000.jsonl" \
    --unit "$4" --k $KS --n-prompts "$E0_N" --T "$E0_T" --out "$OUT/e0_$1.json"
}
f_all() {  # Fig 1 / Fig 3 data: one step of the first model at the full vocabulary, candidate contexts
  # shellcheck disable=SC2086
  set -- $FIG_MODEL
  local c
  job "F pools $1" $PY -I -u "$P/../plots/pool_for_context.py" "$M/$2" 0 14 "${F_CONTEXTS[@]}" \
    "I'll call you this@@␣week|␣weekend|␣even|␣evening|␣after|␣afternoon" > "$OUT/fig/pools_$1.txt" || return $?
  job "F forest $1" $PY -I -u "$P/../plots/fig3_dump_forest.py" "$M/$2" 0.005 "$OUT/fig/forest_$1_" \
    "${F_CONTEXTS[@]}" || return $?
  cat "$OUT/fig/pools_$1.txt"
}
r2_one() {  # $5 start $6 count: the main table, seven methods x KS
  c4job "R2 e1 $1 [$5,+$6)" "$1" "$2" "$3" "$4" --methods $METHODS --k $KS --start "$5" --samples "$6" \
    "${STREAM[@]}" --mwis-solver "$MWIS" --skip-done --out "$OUT/e1_$1.jsonl" \
    --steps-out "$OUT/steps_e1_$1.jsonl" || return $?
  validate "$OUT/e1_$1.jsonl"
}
r4_one() {  # LAS (public release planner, tau = 0) on the same prompts, keys and streams as R2
  c4job "R4 las $1 [$5,+$6)" "$1" "$2" "$3" "$4" --methods las --k $KS --start "$5" --samples "$6" \
    "${STREAM[@]}" --skip-done --out "$OUT/e1_$1.jsonl" --steps-out "$OUT/steps_e1_$1.jsonl" || return $?
  validate "$OUT/e1_$1.jsonl"
}
r5_one() {  # GPU part of the GoF test: save the fixed states; the draws run later on any CPU
  local tag=$1 dir=$2 lang=$3 unit=$4
  job "R5 gof-states $tag" $PY -I -u "$P/gof_single_step.py" --model-dir "$M/$dir" --tag "$tag" \
    --prompts "$D/c4_${lang}_1000.jsonl" --unit "$unit" --n-ctx "$GOF_CTX" --draws 0 \
    --states-out "$OUT/gof_states_$tag.npz" --out "$OUT/gof_states_$tag.meta.jsonl"
}
t_one() {  # W6 timing: lean code (no diagnostics), methods interleaved per text, warm-up, receivers timed too
  c4job "T timing $1" "$@" --methods $METHODS las --k $KS --samples "$N_TIME" "${STREAM[@]}" --recv-first -1 \
    --mwis-solver "$MWIS" --lean --interleave --warmup 1 --skip-done --out "$OUT/timing_$1.jsonl" \
    --steps-out "$OUT/steps_timing_$1.jsonl" || return $?
  validate --one-machine "$OUT/timing_$1.jsonl"
}
tb_one() {  # CPU microbenchmark of each method's per-step work on the R5 laws (same machine as T)
  job "TB bench $1" $PY -I -u "$P/timing_bench.py" --model-dir "$M/$2" --tag "$1" \
    --states-in "$OUT/gof_states_$1.npz" --n-states "$TB_STATES" --reps "$TB_REPS" --k $KS --out "$OUT/tb_$1.jsonl"
}
r6_one() {  # long texts: T_LONG tokens at Full
  c4job "R6 long $1" "$@" --methods coin native syncpool --k 0 --samples "$N_LONG" \
    --coder rrc --stop fixed --T "$T_LONG" --skip-done --out "$OUT/long_$1.jsonl" \
    --steps-out "$OUT/steps_long_$1.jsonl" || return $?
  validate "$OUT/long_$1.jsonl"
}
r8_one() {  # W5: lower temperature and nucleus sampling
  local tag=$1 dir=$2 lang=$3 unit=$4 s
  for s in "0.7 1.0" "1.0 0.95"; do
    set -- $s
    job "R8 e0 $tag T=$1 p=$2" $PY -I -u "$P/e0_atlas.py" --model-dir "$M/$dir" --tag "$tag" \
      --prompts "$D/c4_${lang}_1000.jsonl" --unit "$unit" --k 256 0 --n-prompts "$E0_N" --T "$E0_T" \
      --temperature "$1" --top-p "$2" --out "$OUT/e0_${tag}_T$1_p$2.json" || return $?
    c4job "R8 e1 $tag T=$1 p=$2" "$tag" "$dir" "$lang" "$unit" --methods native syncpool coin \
      --k 256 0 --samples "$N_ROB" --temperature "$1" --top-p "$2" "${STREAM[@]}" --skip-done \
      --out "$OUT/e1_${tag}_T$1_p$2.jsonl" || return $?
    validate "$OUT/e1_${tag}_T$1_p$2.jsonl" || return $?
  done
}
r8c_one() {  # long prompts (strict: texts shorter than the prompt size are dropped)
  local tag=$1 dir=$2 lang=$3 unit=$4 pu
  pu=$(long_units "$unit")
  job "R8c e0 $tag prompt=$pu" $PY -I -u "$P/e0_atlas.py" --model-dir "$M/$dir" --tag "$tag" \
    --prompts "$D/c4_${lang}_1000.jsonl" --unit "$unit" --prompt-units "$pu" --k 256 0 --n-prompts "$E0_N" \
    --T "$E0_T" --out "$OUT/e0_${tag}_prompt$pu.json" || return $?
  c4job "R8c e1 $tag prompt=$pu" "$tag" "$dir" "$lang" "$unit" --prompt-units "$pu" --methods native syncpool coin \
    --k 256 0 --samples "$N_ROB" "${STREAM[@]}" --skip-done --out "$OUT/e1_${tag}_prompt$pu.jsonl" || return $?
  validate "$OUT/e1_${tag}_prompt$pu.jsonl"
}
r8s_one() {  # Stepwise at k=256 on the R8 and R8c prompts, keys and streams, added to the same files
  local tag=$1 dir=$2 lang=$3 unit=$4 s pu
  for s in "0.7 1.0" "1.0 0.95"; do
    set -- $s
    c4job "R8s e1 $tag T=$1 p=$2" "$tag" "$dir" "$lang" "$unit" --methods stepwise --k 256 \
      --samples "$N_ROB" --temperature "$1" --top-p "$2" "${STREAM[@]}" --skip-done \
      --out "$OUT/e1_${tag}_T$1_p$2.jsonl" || return $?
    validate "$OUT/e1_${tag}_T$1_p$2.jsonl" || return $?
  done
  pu=$(long_units "$unit")
  c4job "R8s e1 $tag prompt=$pu" "$tag" "$dir" "$lang" "$unit" --prompt-units "$pu" --methods stepwise --k 256 \
    --samples "$N_ROB" "${STREAM[@]}" --skip-done --out "$OUT/e1_${tag}_prompt$pu.jsonl" || return $?
  validate "$OUT/e1_${tag}_prompt$pu.jsonl"
}
r10_one() {  # other exact / standard coders on the prompts, keys and streams of the first N_ROB texts of R2
  local tag=$1 dir=$2 lang=$3 unit=$4 c
  for c in discop ac; do
    c4job "R10 e1 $tag coder=$c" "$tag" "$dir" "$lang" "$unit" --methods native syncpool coin --k 256 0 \
      --samples "$N_ROB" --coder "$c" --stop fixed --T "$T" --recv-first "$RECV_FIRST" --skip-done \
      --out "$OUT/e1_${tag}_coder_$c.jsonl" --steps-out "$OUT/steps_e1_${tag}_coder_$c.jsonl" || return $?
    validate "$OUT/e1_${tag}_coder_$c.jsonl" || return $?
  done
}
r11_one() {  # ReTokSync: Discop coder, its reset after a merge, bits repaired offline; first N_RTS texts of R2
  local tag=$1 dir=$2 lang=$3 unit=$4
  job "R11 retoksync $tag" $PY -I -u "$P/retoksync_adapter.py" --model-dir "$M/$dir" --tag "$tag" \
    --prompts "$D/c4_${lang}_1000.jsonl" --unit "$unit" --k 256 0 --samples "$N_RTS" --T "$T" \
    --skip-done --out "$OUT/e1_${tag}_retoksync.jsonl" || return $?
  validate "$OUT/e1_${tag}_retoksync.jsonl"
}
r1x_one() {  # the decomposition of R1 with each model on the prompts of the other two languages
  local tag=$1 dir=$2 own=$3 l u
  for l in en zh ja; do
    [ "$l" = "$own" ] && continue
    [ "$l" = en ] && u=words || u=chars
    job "R1x e0 $tag on $l" $PY -I -u "$P/e0_atlas.py" --model-dir "$M/$dir" --tag "${tag}_x$l" \
      --prompts "$D/c4_${l}_1000.jsonl" --unit "$u" --k $KS --n-prompts "$E0_N" --T "$E0_T" \
      --out "$OUT/e0_${tag}_x$l.json" || return $?
  done
}
r2r_one() {  # SyncPool and coin projection, sender and receiver, on the texts [RECV_FIRST, N) of R2
  [ "$RECV_FIRST" -ge 0 ] && [ "$RECV_FIRST" -lt "$N" ] || { echo "R2r: nothing left undecoded"; return 0; }
  c4job "R2r e1 $1 [$RECV_FIRST,$N)" "$1" "$2" "$3" "$4" --methods syncpool coin --k $KS \
    --start "$RECV_FIRST" --samples $(( N - RECV_FIRST )) --coder rrc --stop fixed --T "$T" --recv-first -1 \
    --skip-done --out "$OUT/e1_$1_recvall.jsonl" || return $?
  validate "$OUT/e1_$1_recvall.jsonl"
}
r9_one() {
  job "R9 equiv $1" $PY -I -u "$P/check_equiv.py" "$M/$2" "$D/c4_${3}_1000.jsonl" "$4" "$EQUIV_N"
}
r9b_one() {  # needs R2 records
  job "R9b e2b $1" $PY -I -u "$P/e2b_ontraj.py" --tag "$1" --model-dir "$M/$2" \
    --in "$OUT/e1_$1.jsonl" --out "$OUT/e2b_$1.jsonl" --ks $E2B_KS --max-sample "$E2B_MAX"
}
p_gof() {  # chi-square draws on the R5 states, the shared RRC coder, the main MWIS solver
  job "P gof $1" $PY -I -u "$P/gof_single_step.py" --model-dir "$M/$2" --tag "$1" \
    --states-in "$OUT/gof_states_$1.npz" --k $GOF_KS --draws "$GOF_DRAWS" --coder rrc --mwis-solver "$MWIS" \
    --out "$OUT/gof_$1.jsonl"
}
p_all() {  # needs R2 R4 R5 of all three models
  each p_gof || return $?
  local tags=() dirs=() spec A="$P/../analysis"
  for spec in "${SPECS[@]}"; do
    # shellcheck disable=SC2086
    set -- $spec
    tags+=(--tag "$1"); dirs+=("$M/$2")
  done
  job "P native-fallback" env COIN_MODELS="$M" $PY -I -u "$P/native_fallback.py" --runs "$OUT" || return $?
  job "P noncanon" env COIN_MODELS="$M" COIN_DATA="$D" $PY -I -u "$P/noncanon_events.py" --runs "$OUT" \
    "${tags[@]}" || return $?
  job "P stepwise-diag" $PY -I -u "$A/stepwise_diag.py" --protocol-dir "$P" --runs "$OUT" --models-root "$M" \
    --out "$OUT/stepwise_diag.json" || return $?
  job "P tok-origin" $PY -I -u "$A/tok_origin.py" --protocol-dir "$P" --out "$OUT/tok_origin.json" "${dirs[@]}"
}

for stage in "$@"; do
  record_env "$stage"
  case "$stage" in
    R0) each r0_one ;;
    R1) each r1_one ;;
    F) f_all ;;
    R2) chunked r2_one ;;
    R4) chunked r4_one ;;
    R5) each r5_one ;;
    T) each t_one ;;
    TB) each tb_one ;;
    R6) each r6_one ;;
    R8) each r8_one ;;
    R8c) each r8c_one ;;
    R8s) each r8s_one ;;
    R10) each r10_one ;;
    R11) each r11_one ;;
    R1x) each r1x_one ;;
    R2r) each r2r_one ;;
    R9) each r9_one ;;
    R9b) each r9b_one ;;
    P) p_all ;;
    *) echo "unknown stage $stage" >&2; exit 1 ;;
  esac 2>&1 | tee -a "$OUT/logs/$stage.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "STAGE $stage FAILED rc=$rc $(date '+%F %T'); stopping" | tee -a "$OUT/logs/$stage.log"
    exit "$rc"
  fi
  echo "STAGE $stage DONE $(date '+%F %T')" | tee -a "$OUT/logs/$stage.log"
done
