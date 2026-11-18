"""C4 continuation protocol for the coin paper (Yan / Stepwise style).

Shared context: the first 10 words (en) or 10 characters (ja, zh) of a C4 train
document; base-model continuation at T = 1; random 128-bit message; the Stepwise
repository's ArithmeticCoder for every method (bit-identical lazy-boundary
variant, see FastAC); sender stops at the first token boundary after the message
is resolved at which the text is valid UTF-8 (it keeps encoding the same value,
which the decoder replays harmlessly).

Channel: the continuation as a Python str (UTF-8 bytes decoded with
errors="replace"); every receiver gets only this str plus the shared prompt and
key.  Receivers:
  native, stepwise         canonical retokenization of prompt + text (Stepwise repo)
  basic, mwis              unique prefix parse over the prefix-free pool
  syncpool, hubdrop        unique group-root parse, shared-randomness member draw
  coin                     unique survivor parse, shared coins

Prefix relations are byte-exact (token byte table) for every method; the
Stepwise repo's official algorithms (Basic removal rule, MWIS forest DP,
connected components, consistency filter) are called on that graph.

Run as:  cd /tmp && python -I .../c4proto.py --help
"""
from __future__ import annotations

import sys

import os as _os

# Run as a script, this file is __main__; helpers that `import c4proto` (las_adapter) must get this same module
# object, or they see a second copy with its own switches (DIAG, TIMERS) and exception classes.
if __name__ == "__main__":
    sys.modules.setdefault("c4proto", sys.modules[__name__])

# The coin code is self-contained in this directory.  The only external code is the
# official Stepwise repository (Yan & Murawaki), used for its coder, stepper and
# baselines; point STEPWISE_SRC at its src/ directory (pinned commit, see README).
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_STEPWISE_SRC = _os.environ.get("STEPWISE_SRC")
if _STEPWISE_SRC:
    sys.path.insert(0, _STEPWISE_SRC)

import argparse
import collections
import functools
import decimal
import hashlib
import hmac
import json
import math
import os
import random
import re
import socket
import struct
import time
from decimal import Decimal

import numpy as np
import torch

from coin_core import CoinIndex, coin_prf, coin_project, coin_project_sparse, last_live_sparse
from mwis_exact import prefix_mwis

try:
    from tokenization_consistency import pools as SW
    from tokenization_consistency.coding import ArithmeticCoder, CodingError
    from tokenization_consistency.models import LMStepper
except ImportError as _e:  # pragma: no cover - depends on the environment
    raise SystemExit(
        "tokenization_consistency is not importable.  The official Stepwise code is used for the coder and "
        "the Stepwise baselines: clone github.com/ryehr/Consistency at commit 962473d (v0.2.0) and set "
        "STEPWISE_SRC=<clone>/src (see README, 'External baselines')."
    ) from _e

HUB_Q = 0.05
LN2 = math.log(2.0)

# ---------------------------------------------------------------------------
# Instrumentation.  Nothing here changes a text, a bit or a decoding result:
#   TIMERS  wall seconds per component (top-k, coin PRF, projection, grouping, MWIS), summed over a text and
#           reset by the sender at the start of every text;
#   DIAG    False in timing runs: the plans skip their diagnostics (entropies of the projection), which feed
#           only the statistics, so the timed code is the algorithm alone;
#   _sync   after every forward, so the forward's GPU time is charged to the forward rather than to the next
#           host-side read of the logits (CUDA runs asynchronously).
# ---------------------------------------------------------------------------
TIMERS = collections.defaultdict(float)
DIAG = True
_CUDA = torch.cuda.is_available()


def _sync():
    if _CUDA:
        torch.cuda.synchronize()


def _throttle():
    """(throttled periods, throttled microseconds) of this cgroup, or (None, None) outside a cgroup v2."""
    try:
        with open("/sys/fs/cgroup/cpu.stat") as fh:
            kv = dict(line.split() for line in fh)
        return int(kv.get("nr_throttled", 0)), int(kv.get("throttled_usec", 0))
    except (OSError, ValueError):
        return None, None


@functools.lru_cache(maxsize=1)
def run_info():
    """Machine fields written into every record (texts from different machines stay traceable)."""
    gpu = torch.cuda.get_device_name(0) if _CUDA else "cpu"
    return dict(host=socket.gethostname(), gpu=gpu)


def machine_meta(model_dir: str) -> dict:
    """Everything about the machine and the software that a timing or a result could depend on."""
    import platform

    import transformers

    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    def read(path):
        try:
            with open(path) as fh:
                return fh.read().strip()
        except OSError:
            return None

    files, digests = [], {}
    for name in sorted(os.listdir(model_dir)):
        full = os.path.join(model_dir, name)
        if os.path.isfile(full):
            files.append([name, os.path.getsize(full)])
            if name.endswith(".json") or name.endswith(".model"):
                with open(full, "rb") as fh:
                    digests[name] = hashlib.sha256(fh.read()).hexdigest()
    return dict(run_info(), python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, transformers=transformers.__version__, numpy=np.__version__,
                tokenizers=ver("tokenizers"), cryptography=ver("cryptography"),
                torch_threads=torch.get_num_threads(), omp=os.environ.get("OMP_NUM_THREADS"),
                cpu_max=read("/sys/fs/cgroup/cpu.max"), cpus_allowed=len(os.sched_getaffinity(0)),
                model_files=files, model_digests=digests)


# ---------------------------------------------------------------------------
# Tokenizer byte table
# ---------------------------------------------------------------------------

def _bytes_to_unicode_inv():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes_table(tok, vocab_width: int):
    data = json.loads(tok.backend_tokenizer.to_str())
    bytelevel = "ByteLevel" in json.dumps(data.get("decoder"))
    added = {a["id"] for a in data.get("added_tokens", [])} | set(tok.all_special_ids)
    inv = _bytes_to_unicode_inv()
    n_tok = len(tok)                       # ~30 ms per call in this transformers version
    pieces = tok.convert_ids_to_tokens(list(range(min(n_tok, vocab_width))))
    table = []
    for i in range(vocab_width):
        t = pieces[i] if i < n_tok else None
        if t is None or i in added:
            table.append(b"")
            continue
        if bytelevel:
            try:
                table.append(bytes(inv[c] for c in t))
            except KeyError:
                table.append(b"")
        else:
            m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", t)
            table.append(bytes([int(m.group(1), 16)]) if m else t.replace("▁", " ").encode("utf-8"))
    return table, ("bytelevel" if bytelevel else "sentencepiece")


# ---------------------------------------------------------------------------
# Arithmetic coder: the Stepwise repo's coder, boundaries computed lazily
# ---------------------------------------------------------------------------

class FastAC(ArithmeticCoder):
    """Same intervals as ArithmeticCoder (tensor branch) without building all boundaries.

    Interior boundaries low + Decimal(c_i) * width are monotone because the float64
    cumsum of non-negative weights is monotone and Decimal ops round monotonically, so
    the official monotone fix-up can only touch the anchored last boundary.
    """

    @staticmethod
    def _cum(probs):
        w = probs.detach().to("cpu", torch.float64).flatten()
        return torch.cumsum(w / w.sum(), dim=0).numpy()

    def _b(self, c, i):
        n = c.size
        if i == 0:
            return self.low
        with decimal.localcontext(self._context):
            width = self.high - self.low
            if i < n:
                return self.low + Decimal(float(c[i - 1])) * width
            last = self.low + Decimal(float(c[n - 2])) * width if n >= 2 else self.low
            return max(self.high, last)

    def _set(self, lo, hi, index):
        if hi <= lo:
            raise CodingError(f"degenerate code interval at step {self.steps} (symbol {index})")
        self.low, self.high = lo, hi
        self.steps += 1

    def encode_step(self, probs, value: int) -> int:
        c = self._cum(probs)
        n = c.size
        if n == 0:
            raise CodingError("cannot code an empty candidate pool")
        target = Decimal(value)
        with decimal.localcontext(self._context):
            r = float((target - self.low) / (self.high - self.low))
        i = min(max(int(np.searchsorted(c[: n - 1], r, side="right")), 0), n - 1)
        while i > 0 and self._b(c, i) > target:
            i -= 1
        while i < n - 1 and self._b(c, i + 1) <= target:
            i += 1
        lo = self._b(c, i)
        if lo > target:
            raise CodingError("message value fell outside the code interval")
        self._set(lo, self._b(c, i + 1), i)
        return i

    def decode_step(self, probs, index: int) -> None:
        c = self._cum(probs)
        if not 0 <= index < c.size:
            raise CodingError(f"symbol {index} outside pool of {c.size}")
        self._set(self._b(c, index), self._b(c, index + 1), index)


class RRCCoder:
    """Rotation range coder (Yan & Murawaki, ACL 2026; Algorithm 3/4 + replay guard).

    Matches external_baselines/RRC/rrc_core.py using per-step shared randomness
    step_uniform(key, b"rrc", step) and an isolated local Decimal context so
    each message is self-contained and bit-reproducible in isolation.
    """

    def __init__(self, bit_length: int, key: bytes, margin: int = 1200):
        self.bit_length = bit_length
        self.key = key
        digits = int((Decimal(bit_length) * Decimal(2).log10()).to_integral_value(rounding=decimal.ROUND_CEILING))
        self._ctx = decimal.Context(prec=digits + margin)
        self.L = Decimal(0)
        self.R = Decimal(2) ** bit_length
        self.d_s = None
        self.history = []
        self.steps = 0

    @staticmethod
    def _cum(probs):
        w = probs.detach().to("cpu", torch.float64).flatten()
        n = w.numel()
        bnd = torch.empty(n + 1, dtype=torch.float64)
        bnd[0] = 0.0
        torch.cumsum(w, dim=0, out=bnd[1:])
        bnd[1:] /= bnd[n].clone()
        return bnd.numpy()

    @staticmethod
    def _mod(a: Decimal, m: Decimal) -> Decimal:
        return a - m * (a / m).to_integral_value(rounding=decimal.ROUND_FLOOR)

    def encode_step(self, probs, value: int) -> int:
        bnd = self._cum(probs)
        n = bnd.size - 1
        if n <= 0:
            raise CodingError("cannot code an empty candidate pool")
        with decimal.localcontext(self._ctx):
            if self.d_s is None:
                self.d_s = Decimal(value)
            delta = self.R - self.L
            off = Decimal(step_uniform(self.key, b"rrc", self.steps))
            self.history.append((self.L, delta, off))
            self.d_s = self.L + self._mod((self.d_s - self.L) + off * delta, delta)
            if not (self.L <= self.d_s < self.R):
                raise CodingError("RRC rotation left the interval; working precision exhausted")
            lo, hi = 0, n - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if self.L + Decimal(str(bnd[mid + 1])) * delta > self.d_s:
                    hi = mid
                else:
                    lo = mid + 1
            upper = self.L + Decimal(str(bnd[lo + 1])) * delta
            if not upper > self.d_s:
                raise CodingError("RRC sub-interval collapsed below working precision")
            lower = self.L + Decimal(str(bnd[lo])) * delta
            self.L, self.R = lower, upper
            self.steps += 1
            return lo

    def decode_step(self, probs, index: int) -> None:
        bnd = self._cum(probs)
        n = bnd.size - 1
        if not 0 <= index < n:
            raise CodingError(f"symbol {index} outside pool of {n}")
        with decimal.localcontext(self._ctx):
            delta = self.R - self.L
            off = Decimal(step_uniform(self.key, b"rrc", self.steps))
            self.history.append((self.L, delta, off))
            lower = self.L + Decimal(str(bnd[index])) * delta
            upper = self.L + Decimal(str(bnd[index + 1])) * delta
            if upper <= lower:
                raise CodingError(f"degenerate RRC interval at step {self.steps} (symbol {index})")
            self.L, self.R = lower, upper
            self.steps += 1

    def _replay_int(self) -> int:
        with decimal.localcontext(self._ctx):
            mid = (self.L + self.R) / 2
            for L_prev, delta_prev, off in reversed(self.history):
                mid = L_prev + self._mod(mid - L_prev - off * delta_prev, delta_prev)
            return int(mid.to_integral_value(rounding=decimal.ROUND_HALF_DOWN))

    def resolved(self, value: int) -> bool:
        with decimal.localcontext(self._ctx):
            diff = (self.L + self.R) / 2 - self.d_s
            if not (Decimal("-0.5") < diff <= Decimal("0.5")):
                return False
        return self._replay_int() == value

    def message_bits(self) -> str:
        if not self.history:
            raise CodingError("stegotext has no tokens beyond the context")
        d = self._replay_int()
        if not 0 <= d < (1 << self.bit_length):
            raise CodingError(f"recovered value {d} is outside [0, 2**{self.bit_length})")
        return format(d, f"0{self.bit_length}b")


class DiscopCoder:
    """Discop with the Huffman recursion (Ding et al., IEEE S&P 2023), the default setting 'Discop' of the
    official code (external_baselines/Discop/src/stega_cy.pyx: create_huffman_tree, cy_encode_step,
    cy_decode_step), ported rule for rule.

    Per step, a Huffman tree is built over the law handed to the coder: the official two-queue construction on
    the probabilities in decreasing order, with its tie rule, the smaller node becoming the left child.  At every
    internal node on the path one uniform u places two copies of the node's distribution, at u and at
    u + 1/2 (mod 1); the next stream bit selects the copy whose child is entered.  When the two copies fall into
    different children the node carries that bit, otherwise none.  Either child is entered with its conditional
    probability whatever the bit, so the token follows the law it is handed.

    The only change is the source of the uniforms: the official code draws random.random() from a seeded
    generator, one per visited node; here the node at depth d of step t uses step_uniform(key, b"discop"+t, d),
    so the receiver needs no generator state and the uniforms never depend on the stream.
    """

    def __init__(self, bit_length: int, key: bytes):
        self.bit_length = bit_length
        self.key = key
        self.value = None          # sender: the stream as an integer
        self.bits = []             # stream bits fixed so far (sender: embedded; receiver: decoded)
        self.steps = 0

    @staticmethod
    def _tree(probs):
        w = probs.detach().to("cpu", torch.float64).flatten().numpy()
        live = np.flatnonzero(w > 0)
        if live.size == 0:
            raise CodingError("cannot code an empty candidate pool")
        desc = live[np.argsort(-w[live], kind="stable")]
        leaf = desc[::-1].copy()                  # queue 1 of the official code: increasing probability
        n = leaf.size
        prob = (w[leaf] / w[live].sum()).tolist() + [0.0] * (n - 1)
        left, right = [0] * (n - 1), [0] * (n - 1)
        i1, h2 = 0, n                             # fronts of queue 1 (leaves) and queue 2 (merged nodes)
        for m in range(n, 2 * n - 1):
            if i1 < n and h2 < m and prob[i1] < prob[h2]:
                a = i1; i1 += 1
            elif i1 >= n:
                a = h2; h2 += 1
            elif h2 >= m:
                a = i1; i1 += 1
            else:
                a = h2; h2 += 1
            if i1 < n and h2 < m and prob[i1] < prob[h2]:
                b = i1; i1 += 1
            elif i1 >= n:
                b = h2; h2 += 1
            elif h2 >= m:
                b = i1; i1 += 1
            else:
                b = h2; h2 += 1
            prob[m] = prob[a] + prob[b]
            left[m - n], right[m - n] = a, b
        return prob, left, right, 2 * n - 2, n, leaf, w.size

    def _copies(self, prob, left, node, n, depth):
        """Children (False = left, True = right) entered by copy 0 and copy 1 at this node."""
        s = prob[node]
        u = step_uniform(self.key, b"discop" + struct.pack(">Q", self.steps), depth)
        p0, p1 = u * s, (u + 0.5) * s
        if p1 > s:
            p1 -= s
        part = prob[left[node - n]]
        return p0 >= part, p1 >= part

    def encode_step(self, probs, value: int) -> int:
        if self.value is None:
            self.value = value
        prob, left, right, node, n, leaf, _ = self._tree(probs)
        depth = 0
        while node >= n:
            c0, c1 = self._copies(prob, left, node, n, depth)
            pos = len(self.bits)
            bit = (self.value >> (self.bit_length - 1 - pos)) & 1 if pos < self.bit_length else 0
            go_right = c1 if bit else c0
            if c0 != c1 and pos < self.bit_length:
                self.bits.append("1" if bit else "0")
            node = right[node - n] if go_right else left[node - n]
            depth += 1
        self.steps += 1
        return int(leaf[node])

    def decode_step(self, probs, index: int) -> None:
        prob, left, right, root, n, leaf, size = self._tree(probs)
        if not 0 <= index < size:
            raise CodingError(f"symbol {index} outside pool of {size}")
        hit = np.flatnonzero(leaf == index)
        if hit.size != 1:
            raise CodingError(f"symbol {index} has zero probability")
        parent = np.empty(2 * n - 1, dtype=np.int64)
        if n > 1:
            parent[np.asarray(left)] = np.arange(n, 2 * n - 1)
            parent[np.asarray(right)] = np.arange(n, 2 * n - 1)
        path, x = [], int(hit[0])
        while x != root:
            px = int(parent[x])
            path.append(right[px - n] == x)
            x = px
        node = root
        for depth, go_right in enumerate(reversed(path)):
            c0, c1 = self._copies(prob, left, node, n, depth)
            if c0 != c1:
                if len(self.bits) < self.bit_length:
                    self.bits.append("0" if c0 == go_right else "1")
            elif c0 != go_right:
                raise CodingError(f"Discop: symbol {index} is unreachable at step {self.steps}")
            node = right[node - n] if go_right else left[node - n]
        self.steps += 1

    def prefix(self) -> str:
        return "".join(self.bits)

    def message_bits(self) -> str:
        if len(self.bits) < self.bit_length:
            raise CodingError(f"only {len(self.bits)} of {self.bit_length} message bits decoded")
        return self.prefix()


def make_coder(coder: str, bit_length: int, key: bytes):
    if coder == "ac":
        return FastAC(bit_length=bit_length)
    if coder == "rrc":
        return RRCCoder(bit_length=bit_length, key=key)
    if coder == "discop":
        return DiscopCoder(bit_length=bit_length, key=key)
    raise ValueError(f"unknown coder: {coder}")


# ---------------------------------------------------------------------------
# Shared randomness
# ---------------------------------------------------------------------------

def sample_key(tag: str, sample: int) -> bytes:
    return hashlib.sha256(f"coin-acl-c4|{tag}|{sample}".encode()).digest()


def step_key(key: bytes, label: bytes, step: int) -> bytes:
    return hmac.new(key, label + struct.pack(">Q", step), hashlib.sha256).digest()


def step_uniform(key: bytes, label: bytes, step: int) -> float:
    return (int.from_bytes(step_key(key, label, step)[:8], "big") >> 11) / float(1 << 53)


def draw(weights: np.ndarray, u: float) -> int:
    c = np.cumsum(weights)
    return int(min(np.searchsorted(c, u * c[-1], side="right"), weights.size - 1))


class RecvError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Environment: model, tokenizer, byte structures
# ---------------------------------------------------------------------------

class Env:
    def __init__(self, model_dir: str, device: str = "cuda", proj: str = "dense",
                 temperature: float = 1.0, top_p: float = 1.0):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        # decode must be byte-faithful for every method (Llama-3 ships cleanup=True)
        self.tok.clean_up_tokenization_spaces = False
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
        self.V = int(self.model.get_output_embeddings().weight.shape[0])
        self.table, self.tok_kind = token_bytes_table(self.tok, self.V)
        self.mask = torch.tensor([bool(b) for b in self.table], device=device)
        self.index = CoinIndex(self.table)
        # proj: how coin projection and SyncPool grouping find the live prefix pairs.
        # "dense" masks the whole static pair table, O(|V| + #pairs); "sparse" visits only
        # the support S and its static ancestors, O(|S| + #pairs into S); "auto" uses
        # sparse when |S| <= |V|/16.  All three give bit-identical results.
        if proj not in ("dense", "sparse", "auto"):
            raise ValueError(f"unknown proj {proj!r}")
        self.proj = proj
        # sampling law before top-k: softmax(logits / temperature), then nucleus top_p.
        # temperature=1 and top_p=1 leave the law untouched (the paper's main setting).
        if not temperature > 0 or not 0 < top_p <= 1:
            raise ValueError(f"need temperature > 0 and 0 < top_p <= 1, got {temperature}, {top_p}")
        self.temperature, self.top_p = float(temperature), float(top_p)
        self.klen = np.array([len(b) for b in self.table], dtype=np.int64)
        self.rep_of = self.index.rep_of
        self.anc, self.desc = self.index.anc, self.index.desc
        self.max_key_len = self.index.max_key_len
        self.first_member = np.zeros(self.V, dtype=np.int64)
        self.multi = np.zeros(self.V, dtype=bool)
        for rep, ms in self.index.homographs.items():
            self.first_member[rep] = ms[0]
            self.multi[rep] = len(ms) > 1
        # static pair orders: per extension, ancestors by decreasing / increasing length, so
        # that after filtering to the live support the last pair per extension is its
        # shortest (root) / longest (parent) live proper prefix -- no per-step sorting
        o = np.lexsort((-self.klen[self.anc], self.desc))
        self.anc_short, self.desc_short = self.anc[o], self.desc[o]
        o = np.lexsort((self.klen[self.anc], self.desc))
        self.anc_long, self.desc_long = self.anc[o], self.desc[o]

    def last_live(self, supp, longest: bool):
        """(extension, ancestor) for each live extension: its longest or shortest live proper prefix."""
        if self.proj != "dense":
            S = np.flatnonzero(supp)
            if self.use_sparse(S.size):
                return last_live_sparse(self.index, S, self.klen, longest)
        a, d = (self.anc_long, self.desc_long) if longest else (self.anc_short, self.desc_short)
        live = supp[a] & supp[d]
        a, d = a[live], d[live]
        if not d.size:
            return d, a
        last = np.r_[d[1:] != d[:-1], True]
        return d[last], a[last]

    def use_sparse(self, n_support: int) -> bool:
        return self.proj == "sparse" or (self.proj == "auto" and 16 * n_support <= self.V)

    def stepper(self, prompt_ids):
        return LMStepper(self.model, self.tok, prompt_ids, vocab_size=self.V, use_cache=True)

    def law(self, stepper) -> torch.Tensor:
        lg = stepper._logits[: self.V].to(torch.float64).masked_fill(~self.mask, -torch.inf)
        if self.temperature != 1.0:
            lg = lg / self.temperature
        p = torch.softmax(lg, dim=-1).cpu()
        self.p_untruncated = p          # Stepwise's last resort ranks tokens outside the nucleus by this law
        return nucleus(p, self.top_p) if self.top_p < 1.0 else p

    # -- byte-level prefix-conflict graph on a token pool (positions 0..n-1) --
    def conflict(self, ids: np.ndarray):
        """ext[i]: j != i whose bytes start with bytes_i; adj[i]: sorted comparable j."""
        n = ids.size
        reps = self.rep_of[ids]
        if (reps < 0).any():
            raise RuntimeError("pool contains a token without visible bytes")
        pos_of = {}
        for i, r in enumerate(reps.tolist()):
            pos_of.setdefault(r, []).append(i)
        S = np.unique(reps)
        if self.proj != "dense" and self.use_sparse(S.size):
            sa, sd = self.index.support_pairs(S)        # same pairs, same order as the dense mask
            pa, pd = S[sa], S[sd]
        else:
            inpool = np.zeros(self.V, dtype=bool)
            inpool[reps] = True
            live = inpool[self.anc] & inpool[self.desc]
            pa, pd = self.anc[live], self.desc[live]
        ext = [[] for _ in range(n)]
        pre = [[] for _ in range(n)]
        for a, d in zip(pa.tolist(), pd.tolist()):
            for i in pos_of[a]:
                for j in pos_of[d]:
                    ext[i].append(j)
                    pre[j].append(i)
        for group in pos_of.values():
            if len(group) > 1:
                for i in group:
                    for j in group:
                        if i != j:
                            ext[i].append(j)
                            pre[i].append(j)
        ext = [sorted(set(e)) for e in ext]
        adj = [sorted(set(e) | set(q)) for e, q in zip(ext, pre)]
        return ext, adj


def nucleus(p: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the fewest most likely tokens whose mass reaches top_p (ties: lower id first), renormalize.

    Runs on the CPU in float64 with a stable sort, so sender and receiver keep the same set.
    """
    order = torch.sort(p, descending=True, stable=True).indices
    c = torch.cumsum(p[order], 0)
    n = int(torch.searchsorted(c, torch.tensor([top_p], dtype=c.dtype))[0]) + 1
    n = max(1, min(n, int((p > 0).sum())))
    keep = order[:n]
    q = torch.zeros_like(p)
    q[keep] = p[keep]
    return q / q.sum()


def _topk_raw(p: torch.Tensor, k: int):
    nnz = int((p > 0).sum())
    kk = nnz if k == 0 else min(k, nnz)
    vals, ids = torch.topk(p, kk)
    return ids, vals, kk


_LAST_TOPK = [None]


def topk(p: torch.Tensor, k: int):
    """The candidate set every method starts from (timed as the shared top-k step).  The sender reads the
    last result back (outside the timers) for the entropy of the top-k law on its own trajectory."""
    s = time.perf_counter()
    out = _topk_raw(p, k)
    TIMERS["topk"] += time.perf_counter() - s
    _LAST_TOPK[0] = out
    return out


# ---------------------------------------------------------------------------
# Methods.  plan(...) is called identically by sender and receiver.
# ---------------------------------------------------------------------------

class Plan:
    """weights: coder input (CPU float64); token(i, ...) -> id; parse(rem) -> (i, id)."""

    def __init__(self, weights, token, parse, stats):
        self.weights, self.token, self.parse, self.stats = weights, token, parse, stats


def _kl_nats(q: np.ndarray, pk: np.ndarray) -> float:
    m = q > 0
    if (pk[m] <= 0).any():
        return float("inf")
    return float((q[m] * np.log(q[m] / pk[m])).sum())


def _pool_stats(env, p, ids_pool, probs_pool, cov_k):
    """Filter methods: coder law = renormalised pool."""
    kept = float(probs_pool.sum())
    return dict(retained=kept, kl_sec=-math.log(max(kept / cov_k, 1e-300)),
                kl_kappa=-math.log(max(kept / cov_k, 1e-300)), pool=int(ids_pool.numel()))


def plan_native(env, p, k, prefix, key, step):
    ids, vals, _ = topk(p, k)
    cov = float(vals.sum())

    def parse(rem):
        raise AssertionError("native uses the canonical receiver")

    return Plan(vals, lambda i: int(ids[i]), parse,
                dict(retained=cov, kl_sec=0.0, kl_kappa=0.0, pool=int(ids.numel()))), ids


def _prefix_parse(env, ids_np):
    lookup = {}
    for i, t in enumerate(ids_np.tolist()):
        lookup.setdefault(env.table[t], i)

    def parse(rem):
        hits = [lookup[rem[:L]] for L in range(1, min(len(rem), env.max_key_len) + 1) if rem[:L] in lookup]
        if len(hits) != 1:
            raise RecvError(f"prefix parse found {len(hits)} pool tokens")
        return hits[0], int(ids_np[hits[0]])

    return parse


def plan_basic(env, p, k, prefix, key, step):
    ids, vals, _ = topk(p, k)
    s = time.perf_counter()
    cov = float(vals.sum())
    ext, _ = env.conflict(ids.numpy())
    removed = set()
    for i in range(ids.numel()):           # official rule: drop i if a not-yet-removed j extends it
        for j in ext[i]:
            if j not in removed:
                removed.add(i)
                break
    keep = torch.tensor([i for i in range(ids.numel()) if i not in removed], dtype=torch.long)
    pid, ppr = ids[keep], vals[keep]
    TIMERS["filter"] += time.perf_counter() - s
    return Plan(ppr, lambda i: int(pid[i]), _prefix_parse(env, pid.numpy()),
                _pool_stats(env, p, pid, ppr, cov)), None


def plan_mwis(env, p, k, prefix, key, step):
    ids, vals, _ = topk(p, k)
    s = time.perf_counter()
    cov = float(vals.sum())
    if getattr(env, "mwis_solver", "official") == "exact":
        keep = torch.from_numpy(prefix_mwis(env, ids.numpy(), vals.numpy()))
    else:
        _, adj = env.conflict(ids.numpy())
        keep = torch.tensor(SW.maximum_weight_independent_set(adj, vals.tolist()), dtype=torch.long)
    pid, ppr = ids[keep], vals[keep]
    TIMERS["filter"] += time.perf_counter() - s
    return Plan(ppr, lambda i: int(pid[i]), _prefix_parse(env, pid.numpy()),
                _pool_stats(env, p, pid, ppr, cov)), None


def plan_stepwise(env, p, k, prefix, key, step):
    s = time.perf_counter()
    pool = SW._pool_consistency(env.tok, prefix, p, k)
    weights = pool.probs
    if len(pool) and not float(pool.probs.sum()) > 0:
        # The release assumes a full-support law.  Under top-p every token outside the nucleus has probability 0,
        # so when no nucleus token is consistent its widening returns only zero-probability tokens.  Apply its own
        # last resort instead: the most likely consistent token, ranked by the law before truncation (stable
        # order, so sender and receiver agree), emitted with coder weight 1 (no bits).
        full = getattr(env, "p_untruncated", p)
        ranked = torch.sort(full, descending=True, stable=True).indices.tolist()
        tok_id = SW.first_consistent_token(env.tok, prefix, ranked)
        if tok_id is None:
            raise CodingError("Stepwise found no consistent token")
        ids1 = torch.tensor([tok_id], dtype=torch.long)
        pool = SW.CandidatePool(ids1, p[ids1], pool_size=pool.pool_size)
        weights = torch.ones(1, dtype=torch.float64)
    TIMERS["filter"] += time.perf_counter() - s
    kl = 0.0
    if DIAG:   # KL to the top-k law needs the top-k pool; the method itself does not
        ids, vals, _ = topk(p, k)
        cov = float(vals.sum())
        pk = torch.zeros_like(p)
        pk[ids] = vals / cov
        if float(pool.probs.sum()) > 0:
            q = (pool.probs / pool.probs.sum()).numpy()
            kl = _kl_nats(q, pk[pool.token_ids].numpy())
        else:
            kl = float("inf")   # the last-resort token lies outside the support of the top-k law

    def parse(rem):
        raise AssertionError("stepwise uses the canonical receiver")

    return Plan(weights, lambda i: int(pool.token_ids[i]), parse,
                dict(retained=float(pool.probs.sum()), kl_sec=kl, kl_kappa=kl, pool=len(pool),
                     widened=int(pool.pool_size > min(k, p.numel())))), pool


def group_items(env, item_rep, item_w):
    """Prefix components of the items (= subtrees of the shortest live ancestor), in the
    Stepwise repo's SyncPool order: mass descending, ties by first item position.
    Returns (rank of each item's group, group weights in coder order, root rep per group)."""
    s = time.perf_counter()
    S = np.unique(item_rep)
    if env.proj != "dense" and env.use_sparse(S.size):
        d, a = last_live_sparse(env.index, S, env.klen, longest=False)
    else:
        supp = np.zeros(env.V, dtype=bool)
        supp[item_rep] = True
        d, a = env.last_live(supp, longest=False)
    # d ascending: an item's root is its shortest live proper prefix, else the item itself
    j = np.minimum(np.searchsorted(d, item_rep), max(d.size - 1, 0))
    hit = (d[j] == item_rep) if d.size else np.zeros(item_rep.size, dtype=bool)
    g_root = np.where(hit, a[j] if d.size else item_rep, item_rep)
    uniq, first, inv = np.unique(g_root, return_index=True, return_inverse=True)
    mass = np.bincount(inv, weights=item_w)
    order = np.lexsort((first, -mass))
    rank = np.empty(uniq.size, dtype=np.int64)
    rank[order] = np.arange(uniq.size)
    TIMERS["group"] += time.perf_counter() - s
    return rank[inv], mass[order], uniq[order]


def _grouped_plan(env, item_tok, item_w, item_grp, gw, root_rep, pk_item, u, retained, kl_sec):
    """Coder picks a group (weights gw); one shared uniform u draws the member ~ item_w."""
    n, G = item_w.size, gw.size
    srt = np.argsort(item_grp, kind="stable")
    c = np.cumsum(item_w[srt])
    starts = np.searchsorted(item_grp[srt], np.arange(G), side="left")
    ends = np.r_[starts[1:], n]
    base = np.where(starts > 0, c[np.maximum(starts - 1, 0)], 0.0)
    pick = np.clip(np.searchsorted(c, base + u * (c[ends - 1] - base), side="right"), starts, ends - 1)
    chosen_item = srt[pick]
    chosen_tok = item_tok[chosen_item]
    kl_kappa = _kl_nats(gw / gw.sum(), pk_item[chosen_item]) if DIAG else 0.0
    root_order = np.argsort(root_rep, kind="stable")
    roots_sorted = root_rep[root_order]

    def parse(rem):
        hits = []
        for L in range(1, min(len(rem), env.max_key_len) + 1):
            r = env.index.key_to_rep.get(rem[:L])
            if r is None:
                continue
            j = int(np.searchsorted(roots_sorted, r))
            if j < G and roots_sorted[j] == r:
                hits.append(int(root_order[j]))
        if len(hits) != 1:
            raise RecvError(f"group-root parse found {len(hits)} groups")
        t = int(chosen_tok[hits[0]])
        if not rem.startswith(env.table[t]):
            raise RecvError("shared member draw does not match the text")
        return hits[0], t

    stats = dict(retained=retained, kl_sec=kl_sec, kl_kappa=kl_kappa, pool=int(G))
    return Plan(torch.from_numpy(np.ascontiguousarray(gw)), lambda i: int(chosen_tok[i]), parse, stats)


def plan_syncpool(env, p, k, prefix, key, step):
    ids, vals, _ = topk(p, k)
    cov = float(vals.sum())
    ids_np, w = ids.numpy(), vals.numpy()
    grp, gw, roots = group_items(env, env.rep_of[ids_np], w)
    u = step_uniform(key, b"member", step)
    return _grouped_plan(env, ids_np, w, grp, gw, roots, w / cov, u, cov, 0.0), None


class _TokLaw:
    """The top-k token law (ids, vals / cov), looked up by token id; only homograph draws ask for it."""

    def __init__(self, ids, vals, cov):
        self.ids, self.vals, self.cov = ids, vals, cov
        self._srt = None

    def prepare(self):
        """Sorted index for many lookups (the hub-dropping ablation at large k)."""
        if self._srt is None:
            o = np.argsort(self.ids, kind="stable")
            self._srt = (o, self.ids[o])

    def at(self, toks):
        toks = np.asarray(toks, dtype=np.int64)
        if self._srt is None:
            out = np.zeros(toks.size)
            for n, t in enumerate(toks.tolist()):
                j = np.flatnonzero(self.ids == t)
                if j.size:
                    out[n] = self.vals[j[0]] / self.cov
            return out
        o, si = self._srt
        j = np.minimum(np.searchsorted(si, toks), si.size - 1)
        return np.where(si[j] == toks, self.vals[o[j]] / self.cov, 0.0)


def _key_law(env, p, k):
    """Key law of the top-k pool on its support: S (key ids ascending), pS (masses merged over
    byte-identical tokens, normalized), the token law, and the top-k mass.  The sparse and dense
    branches add the same terms in the same order and normalize the same array, so they agree bit
    for bit; the dense branch only avoids np.unique on a near-full vocabulary."""
    ids, vals, _ = topk(p, k)
    s = time.perf_counter()
    cov = float(vals.sum())
    ids_np, vals_np = ids.numpy(), vals.numpy()
    reps = env.rep_of[ids_np]
    if env.proj != "dense" and env.use_sparse(ids_np.size):
        S, inv = np.unique(reps, return_inverse=True)
        b = np.bincount(inv, weights=vals_np, minlength=S.size)
        if not (b > 0).all():
            S, b = S[b > 0], b[b > 0]
    else:
        full = np.bincount(reps, weights=vals_np, minlength=env.V)
        S = np.flatnonzero(full)
        b = full[S]
    pS = b / b.sum()
    TIMERS["keylaw"] += time.perf_counter() - s
    return S, pS, _TokLaw(ids_np, vals_np, cov), cov


def _rep_law(env, p, k):
    """Dense form of the key law for the analysis scripts: (pr, tokp, cov) as arrays over the vocabulary."""
    S, pS, law, cov = _key_law(env, p, k)
    pr = np.zeros(env.V)
    pr[S] = pS
    tokp = np.zeros(env.V)
    tokp[law.ids] = law.vals / cov
    return pr, tokp, cov


def _homograph(env, rep, law, u):
    """Token for a key: its only token, or a shared draw among its positive-mass byte-identical tokens."""
    if not env.multi[rep]:
        return int(env.first_member[rep])
    ms = env.index.homographs[rep]
    q = law.at(ms)
    members = [m for m, x in zip(ms, q.tolist()) if x > 0]
    if len(members) == 1:
        return members[0]
    return members[draw(q[q > 0], u)]


def _rep_tokens(env, reps, law, u):
    """Token per key: the only member, or a shared draw among positive-mass homographs."""
    toks = env.first_member[reps].copy()
    multi = np.flatnonzero(env.multi[reps])
    if multi.size > 4:
        law.prepare()
    for i in multi.tolist():
        toks[i] = _homograph(env, int(reps[i]), law, u)
    return toks


def _coin_projection(env, S, pS, coins, sparse):
    if sparse:
        return coin_project_sparse(S, pS, env.index, coins, diag=DIAG)
    pr = np.zeros(env.V)
    pr[S] = pS
    return coin_project(pr, env.anc, env.desc, coins, diag=DIAG)


def plan_coin(env, p, k, prefix, key, step):
    S, pS, law, cov = _key_law(env, p, k)
    prf = coin_prf(key)
    t_prf = [0.0]

    def coins(ids):                       # one AES block per node, timed apart from the projection
        s2 = time.perf_counter()
        u = prf.at(step, ids)
        t_prf[0] += time.perf_counter() - s2
        return u

    sparse = env.proj != "dense" and env.use_sparse(S.size)
    s = time.perf_counter()
    surv, w, diag = _coin_projection(env, S, pS, coins, sparse)
    TIMERS["proj"] += time.perf_counter() - s - t_prf[0]
    TIMERS["prf"] += t_prf[0]
    if getattr(env, "proj_also", False):  # the other path on the same law: timed apart, must agree exactly
        s = time.perf_counter()
        surv2, w2, _ = _coin_projection(env, S, pS, prf.coins(step), not sparse)
        TIMERS["proj_alt"] += time.perf_counter() - s
        if not (np.array_equal(surv, surv2) and np.array_equal(w, w2)):
            raise AssertionError(f"dense and sparse coin projections differ at step {step}")
    uh = step_uniform(key, b"homograph", step)

    def token(i):
        return _homograph(env, int(surv[i]), law, uh)

    def parse(rem):                       # survivors are ascending key ids
        hits = []
        for L in range(1, min(len(rem), env.max_key_len) + 1):
            r = env.index.key_to_rep.get(rem[:L])
            if r is None:
                continue
            j = int(np.searchsorted(surv, r))
            if j < surv.size and surv[j] == r:
                hits.append(j)
        if len(hits) != 1:
            raise RecvError(f"coin parse found {len(hits)} survivors")
        return hits[0], token(hits[0])

    st = dict(retained=cov, kl_sec=0.0, pool=int(surv.size))
    if diag is not None:
        st.update(kl_kappa=_kl_nats(w, pS[np.searchsorted(S, surv)]),
                  H_key=diag["coin_key_entropy_bits"], H_root=diag["coin_root_entropy_bits"],
                  H_stop=diag["coin_stop_entropy_bits"], H_proj=diag["coin_projected_entropy_bits"],
                  nodes=diag["coin_nodes"])
    return Plan(torch.from_numpy(w), token, parse, st), None


def plan_hubdrop(env, p, k, prefix, key, step):
    """Ablation: drop every key that carries less than HUB_Q of its subtree mass, then group like SyncPool."""
    S, pS, law, cov = _key_law(env, p, k)
    sa, sd = env.index.support_pairs(S)
    ext = np.bincount(sa, weights=pS[sd], minlength=S.size)
    mass = pS + ext
    npos = np.unique(sa)
    drop = npos[pS[npos] / mass[npos] < HUB_Q]
    kept = 1.0 - float(pS[drop].sum())
    p2 = pS.copy()
    p2[drop] = 0.0
    p2 /= p2.sum()
    live = np.flatnonzero(p2 > 0)
    live = live[np.argsort(-p2[live], kind="stable")]
    reps = S[live]
    uh = step_uniform(key, b"homograph", step)
    toks = _rep_tokens(env, reps, law, uh)
    w = p2[live]
    grp, gw, roots = group_items(env, reps, w)
    u = step_uniform(key, b"member", step)
    return _grouped_plan(env, toks, w, grp, gw, roots, pS[live], u, cov * kept,
                         -math.log(max(kept, 1e-300))), None


METHODS = dict(native=plan_native, basic=plan_basic, mwis=plan_mwis, stepwise=plan_stepwise,
               syncpool=plan_syncpool, coin=plan_coin, hubdrop=plan_hubdrop)
CANONICAL = {"native", "stepwise"}


# ---------------------------------------------------------------------------
# Sender and receivers
# ---------------------------------------------------------------------------

def utf8_ok(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def send(env, method, k, prompt, bits, key, max_new=512, extra=8, coder="ac"):
    fn = METHODS[method]
    prompt_ids = env.tok(prompt)["input_ids"]
    st = env.stepper(prompt_ids)
    ac = make_coder(coder, len(bits), key)
    value = int(bits, 2)
    out = bytearray()
    resolved = None
    acc, n_acc = {}, 0
    t_plan = t_fwd = t_law = 0.0
    failure = None
    t0 = time.perf_counter()
    for step in range(max_new + extra):
        s = time.perf_counter()
        p = env.law(st)
        t_law += time.perf_counter() - s
        s = time.perf_counter()
        plan, _ = fn(env, p, k, st.token_ids, key, step)
        t_plan += time.perf_counter() - s
        try:
            i = ac.encode_step(plan.weights, value)
        except CodingError as e:
            failure = f"coding_error: {e}"
            break
        tok_id = plan.token(i)
        if resolved is None:
            pp = p.numpy()
            nz = pp[pp > 0]
            plan.stats["H_full"] = float(-(nz * np.log2(nz)).sum())
            plan.stats["kld_stepwise"] = -math.log(max(plan.stats["retained"], 1e-300))
            for a, b in plan.stats.items():
                acc[a] = acc.get(a, 0.0) + float(b)
            n_acc += 1
        s = time.perf_counter()
        st.append(tok_id)
        t_fwd += time.perf_counter() - s
        out += env.table[tok_id]
        if resolved is None and ac.resolved(value):
            resolved = step + 1
        if resolved is not None:
            can_stop = utf8_ok(bytes(out)) and (coder != "rrc" or ac.resolved(value))
            if can_stop or step + 1 - resolved >= extra:
                if coder == "rrc" and not ac.resolved(value):
                    failure = "rrc_replay_wrap_after_extra"
                break
        if resolved is None and step + 1 >= max_new:
            failure = "max_new_tokens"
            break
    gen = st.generated_ids
    rec = dict(complete=resolved is not None and failure is None, failure=failure,
               n_tokens=resolved or len(gen), n_total=len(gen), utf8_valid=utf8_ok(bytes(out)),
               t_send=time.perf_counter() - t0, ms_plan=1000 * t_plan / max(len(gen), 1),
               ms_fwd=1000 * t_fwd / max(len(gen), 1), ms_law=1000 * t_law / max(len(gen), 1),
               token_ids=gen)
    for a, b in acc.items():
        rec[a] = b / max(n_acc, 1)
    return rec, bytes(out).decode("utf-8", errors="replace"), prompt_ids


def receive(env, method, k, prompt, text, key, bit_length=128, max_steps=600, coder="ac", prefix=False):
    """Decode a stegotext.  prefix=False: the whole bit_length-bit message (send).  prefix=True: the stream
    bits fixed by the decoded decisions (send_stream), via decodable_prefix."""
    fn = METHODS[method]
    prompt_ids = env.tok(prompt)["input_ids"]
    ac = make_coder(coder, bit_length, key)
    if method in CANONICAL:
        ids = env.tok(prompt + text)["input_ids"]
        if ids[: len(prompt_ids)] == prompt_ids:
            gen_ids = ids[len(prompt_ids):]
        elif method == "native":
            gen_ids = env.tok(text)["input_ids"]
        else:
            raise RecvError("retokenized text does not start with the prompt tokens")
        st = env.stepper(prompt_ids)
        for step, t in enumerate(gen_ids):
            p = env.law(st)
            plan, pool = fn(env, p, k, st.token_ids, key, step)
            if method == "native":
                hits = (pool == t).nonzero().flatten()
                idx = int(hits[0]) if hits.numel() else None
            else:
                idx = pool.index_of(t)
            if idx is None:
                raise RecvError(f"canonical token absent from pool at step {step}")
            try:
                ac.decode_step(plan.weights, idx)
            except CodingError as e:
                raise RecvError(str(e))
            st.append(t)
        try:
            return (decodable_prefix(ac) if prefix else ac.message_bits()), len(ids) - len(prompt_ids)
        except CodingError as e:
            raise RecvError(str(e))
    rem = text.encode("utf-8")
    st = env.stepper(prompt_ids)
    step = 0
    while rem:
        if step >= max_steps:
            raise RecvError("receiver exceeded the step limit")
        p = env.law(st)
        plan, _ = fn(env, p, k, st.token_ids, key, step)
        idx, t = plan.parse(rem)
        try:
            ac.decode_step(plan.weights, idx)
        except CodingError as e:
            raise RecvError(str(e))
        rem = rem[len(env.table[t]):]
        st.append(t)
        step += 1
    try:
        return (decodable_prefix(ac) if prefix else ac.message_bits()), step
    except CodingError as e:
        raise RecvError(str(e))


def decodable_prefix(coder) -> str:
    """Leading stream bits fixed by the decisions so far: the common prefix of every integer value that the
    decisions are consistent with.  Arithmetic coding: the code interval itself.  RRC: the final interval is
    mapped back through the inverse rotations; when a rotation cuts the mapped arc, the arc is widened to that
    step's whole interval, so the result is a conservative hull and the prefix is always correct.  Discop: the
    bits carried by the nodes on the decisions' paths, in order."""
    n = coder.bit_length
    if isinstance(coder, DiscopCoder):
        return coder.prefix()
    if isinstance(coder, RRCCoder):
        with decimal.localcontext(coder._ctx):
            a, b = coder.L, coder.R
            for L_prev, d_prev, off in reversed(coder.history):
                a2 = L_prev + coder._mod(a - L_prev - off * d_prev, d_prev)
                b2 = a2 + (b - a)
                if b2 > L_prev + d_prev:
                    a2, b2 = L_prev, L_prev + d_prev
                a, b = a2, b2
            lo = int(a.to_integral_value(rounding=decimal.ROUND_CEILING))
            hi = int(b.to_integral_value(rounding=decimal.ROUND_CEILING)) - 1
    else:
        with decimal.localcontext(coder._context):
            lo = int(coder.low.to_integral_value(rounding=decimal.ROUND_CEILING))
            hi = int(coder.high.to_integral_value(rounding=decimal.ROUND_CEILING)) - 1
    lo, hi = max(lo, 0), min(hi, (1 << n) - 1)
    if hi < lo:
        raise CodingError("no stream value is consistent with the decisions")
    m = n - (lo ^ hi).bit_length()
    return format(lo, f"0{n}b")[:m]


def coder_width_bits(coder) -> float:
    """log2 of the current code-interval width (stream bits not yet fixed)."""
    if isinstance(coder, DiscopCoder):
        return float(coder.bit_length - len(coder.bits))
    if isinstance(coder, RRCCoder):
        w = coder.R - coder.L
    else:
        w = coder.high - coder.low
    return float(w.ln() / Decimal(2).ln()) if w > 0 else float("-inf")


def _g(x) -> float:
    """Six significant digits for the per-step arrays (they feed plots, not the coder)."""
    return float(f"{x:.6g}")


def _entropy_bits(q) -> float:
    q = np.asarray(q, dtype=np.float64)
    q = q[q > 0] / q.sum()
    return float(-(q * np.log2(q)).sum())


def timer_fields(n: int) -> dict:
    """Per-token milliseconds of each sub-timer of the text just sent."""
    return {f"ms_{a}": 1000 * b / max(n, 1) for a, b in sorted(TIMERS.items())}


def send_stream(env, method, k, prompt, stream, key, T, extra=8, coder="rrc", steps=None):
    """Fixed-length stegotext: T coder decisions, then more decisions until the bytes are valid UTF-8 (at
    most `extra`), every one embedding the next stream bits.  The length rule never looks at the stream.

    Timed per token: law (logits -> CPU law), plan (the method: top-k, key law, coins, projection, grouping,
    MWIS, and the token choice), code (the coder step), fwd (one forward with the KV cache, synchronized).
    steps: a dict that receives per-step arrays (computed outside the timed regions)."""
    fn = METHODS[method]
    prompt_ids = env.tok(prompt)["input_ids"]
    TIMERS.clear()
    c0 = time.process_time()
    t0 = time.perf_counter()
    st = env.stepper(prompt_ids)
    _sync()
    t_prefill = time.perf_counter() - t0
    cd = make_coder(coder, len(stream), key)
    value = int(stream, 2)
    out = bytearray()
    acc, n_acc, surprisal = {}, 0, 0.0
    t_plan = t_fwd = t_law = t_code = 0.0
    per = collections.defaultdict(list) if steps is not None else None
    failure = None
    step = 0
    while step < T or not utf8_ok(bytes(out)):
        if step >= T + extra:
            failure = "no_utf8_boundary"
            break
        s = time.perf_counter()
        p = env.law(st)
        d_law = time.perf_counter() - s
        s = time.perf_counter()
        plan, _ = fn(env, p, k, st.token_ids, key, step)
        d_plan = time.perf_counter() - s
        s = time.perf_counter()
        try:
            i = cd.encode_step(plan.weights, value)
        except CodingError as e:
            failure = f"coding_error: {e}"
            break
        d_code = time.perf_counter() - s
        s = time.perf_counter()
        tok_id = plan.token(i)
        d_plan += time.perf_counter() - s
        w = plan.weights.detach().to("cpu", torch.float64).flatten().numpy()
        sur = -math.log2(float(w[i]) / float(w.sum()))
        surprisal += sur
        if DIAG:
            plan.stats["H_full"] = _entropy_bits(p.numpy())
            plan.stats["H_topk"] = _entropy_bits(_LAST_TOPK[0][1].numpy())
            plan.stats["H_coder"] = _entropy_bits(w)
        plan.stats["kld_stepwise"] = -math.log(max(plan.stats["retained"], 1e-300))
        for a_, b_ in plan.stats.items():
            acc[a_] = acc.get(a_, 0.0) + float(b_)
        n_acc += 1
        s = time.perf_counter()
        st.append(tok_id)
        _sync()
        d_fwd = time.perf_counter() - s
        t_law, t_plan, t_code, t_fwd = t_law + d_law, t_plan + d_plan, t_code + d_code, t_fwd + d_fwd
        if per is not None:
            per["sur"].append(_g(sur))
            for a_, b_ in plan.stats.items():
                per[a_].append(_g(b_))
            for a_, b_ in (("ms_law", d_law), ("ms_plan", d_plan), ("ms_code", d_code), ("ms_fwd", d_fwd)):
                per[a_].append(_g(1000 * b_))
        out += env.table[tok_id]
        step += 1
    gen = st.generated_ids
    n = max(len(gen), 1)
    rec = dict(complete=failure is None, failure=failure, n_tokens=len(gen), T=T,
               utf8_valid=utf8_ok(bytes(out)), surprisal_bits=surprisal,
               free_bits=coder_width_bits(cd),
               t_send=time.perf_counter() - t0, cpu_send=time.process_time() - c0,
               ms_prefill=1000 * t_prefill, ms_plan=1000 * t_plan / n, ms_fwd=1000 * t_fwd / n,
               ms_law=1000 * t_law / n, ms_code=1000 * t_code / n, token_ids=gen)
    rec.update(timer_fields(n))
    rec["stream_exhausted"] = rec["free_bits"] < 16
    bits = ""
    if failure is None:
        bits = decodable_prefix(cd)
        if not stream.startswith(bits):
            raise AssertionError("sender's decodable prefix disagrees with the stream")
    rec["bits"], rec["n_bits"] = bits, len(bits)
    for a_, b_ in acc.items():
        rec[a_] = b_ / max(n_acc, 1)
    if per is not None:
        steps.update(per)
    return rec, bytes(out).decode("utf-8", errors="replace"), prompt_ids


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_prompt(text: str, unit: str, size: int = 10) -> str:
    """Stepwise repo's rule (Appendix D.1): the first `size` characters, or the text up to its size-th word
    boundary (the whole text when it has fewer)."""
    if unit == "chars":
        return text[:size] if len(text) > size else text
    boundaries = set(" \n,.?!？！。;；:：")
    seen = 0
    for i, ch in enumerate(text):
        if ch in boundaries:
            seen += 1
        if seen >= size:
            return text[:i]
    return text


def load_prompts(path: str, unit: str, size: int = 10):
    """Prompts in file order.  size = 10 is the Stepwise repo's rule (a text with fewer than ten units is used
    whole).  Any other size is strict: a text that does not reach `size` units is dropped, so a long-context
    prompt is never a short one."""
    out, dropped = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            text = json.loads(line)["text"]
            pr = build_prompt(text, unit, size)
            if "�" in pr or not pr.strip() or (size != 10 and pr == text):
                dropped += 1
                continue
            out.append(pr)
    return out, dropped


def message_bits(seed: int, sample: int, n: int = 128) -> str:
    return format(random.Random(seed * 1_000_003 + sample).getrandbits(n), f"0{n}b")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_fixed(env, args, fh, stats, method, k, s, prompt, stream, key, steps_fh=None):
    """One fixed-length stegotext: send_stream, then the method's receiver decodes the stream prefix."""
    extra = 8
    thr0 = _throttle()
    steps = {} if steps_fh is not None else None
    if method == "las":
        if args.coder != "rrc":
            raise SystemExit("las runs with the shared RRC coder")
        import las_adapter
        rec, text, _ = las_adapter.send_stream(env, k, prompt, stream, key, args.T, extra=extra, steps=steps)
    else:
        rec, text, _ = send_stream(env, method, k, prompt, stream, key, args.T, extra=extra, coder=args.coder,
                                   steps=steps)
    nb = rec.pop("n_bits")
    rec.pop("bits")
    rec.update(kind="case", stop="fixed", tag=args.tag, method=method, k=k, sample=s, coder=args.coder,
               mwis_solver=env.mwis_solver if method == "mwis" else None,
               prompt=prompt, text=text, stream_len=len(stream), n_bits=nb,
               temperature=args.temperature, top_p=args.top_p, proj=args.proj,
               prompt_units=args.prompt_units, lean=args.lean, **run_info())
    ok = rec["complete"]
    rec["bpt"] = nb / rec["n_tokens"] if ok else None
    rec["bpc"] = nb / max(len(text), 1) if ok else None
    rec["util"] = rec["bpt"] / rec["H_full"] if ok and rec.get("H_full") else None
    rec["util_k"] = rec["bpt"] / rec["H_topk"] if ok and rec.get("H_topk") else None
    rec["correct"], rec["recv_error"], rec["recv_bits"], rec["first_error"] = None, None, None, None
    # --recv-first N: the receiver runs on samples < N; Native's receiver (which can fail) runs on every text
    rec["recv_checked"] = bool(ok and not args.no_recv
                               and (args.recv_first < 0 or s < args.recv_first or method == "native"))
    if rec["recv_checked"]:
        t0 = time.perf_counter()
        try:
            if method == "las":
                got, nrecv, _ = las_adapter.receive(env, k, prompt, text, key, len(stream))
            else:
                got, nrecv = receive(env, method, k, prompt, text, key, len(stream),
                                     max_steps=args.T + extra + 64, coder=args.coder, prefix=True)
            rec["correct"] = got == stream[:nb]
            rec["recv_bits"], rec["recv_steps"] = len(got), nrecv
            m = 0
            while m < min(len(got), len(stream)) and got[m] == stream[m]:
                m += 1
            rec["first_error"] = None if stream.startswith(got) else m
        except RecvError as e:
            rec["correct"], rec["recv_error"] = False, str(e)[:200]
        except Exception as e:  # recorded, not swallowed
            rec["correct"], rec["recv_error"] = False, f"EXC {type(e).__name__}: {e}"[:200]
        rec["t_recv"] = time.perf_counter() - t0
    thr1 = _throttle()
    if thr0[0] is not None and thr1[0] is not None:
        rec["throttled_n"], rec["throttled_ms"] = thr1[0] - thr0[0], (thr1[1] - thr0[1]) / 1000
    if steps_fh is not None:   # first, so a case row always has its arrays (a resumed case repeats them)
        steps_fh.write(json.dumps(dict(tag=args.tag, method=method, mwis_solver=rec["mwis_solver"], k=k,
                                       sample=s, **steps)) + "\n")
        steps_fh.flush()
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()
    stats["n"] += 1
    stats["complete"] += bool(ok)
    stats["ok"] += bool(rec["correct"])
    stats["checked"] = stats.get("checked", 0) + rec["recv_checked"]
    if ok:
        stats["bpt"] += rec["bpt"]
        stats["util"] += rec["util"] or 0.0


def run_message(env, args, fh, stats, method, k, s, prompt, key):
    """Legacy protocol (stop once a 128-bit message is decodable); kept for the old runs' scripts."""
    bits = message_bits(args.seed, s)
    rec, text, _ = send(env, method, k, prompt, bits, key, coder=args.coder)
    rec.update(kind="case", tag=args.tag, method=method, k=k, sample=s, coder=args.coder,
               prompt=prompt, text=text, msg=hex(int(bits, 2)))
    rec["bpt"] = len(bits) / rec["n_tokens"] if rec["complete"] else None
    rec["util"] = rec["bpt"] / rec["H_full"] if rec["complete"] and rec.get("H_full") else None
    rec["correct"], rec["recv_error"] = None, None
    if rec["complete"] and not args.no_recv:
        t0 = time.perf_counter()
        try:
            got, nrecv = receive(env, method, k, prompt, text, key, len(bits), coder=args.coder)
            rec["correct"] = got == bits
            rec["recv_steps"] = nrecv
        except RecvError as e:
            rec["correct"], rec["recv_error"] = False, str(e)[:200]
        except Exception as e:  # recorded, not swallowed: summarize.py counts it
            rec["correct"], rec["recv_error"] = False, f"EXC {type(e).__name__}: {e}"[:200]
        rec["t_recv"] = time.perf_counter() - t0
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()
    stats["n"] += 1
    stats["complete"] += bool(rec["complete"])
    stats["ok"] += bool(rec["correct"])
    if rec["complete"]:
        stats["bpt"] += rec["bpt"]
        stats["util"] += rec["util"] or 0.0


def _done_label(method, solver):
    """--skip-done key: MWIS rows from the two solvers are different results."""
    return "mwisx" if method == "mwis" and solver == "exact" else method


def _runnable(args, method, k) -> bool:
    if method in ("basic", "mwis", "stepwise") and (k == 0 or k > args.filter_max_k):
        return method == "mwis" and args.mwis_solver == "exact"
    return True


def warm_up(env, args, prompts):
    """Untimed, unrecorded texts before the run: CUDA kernels, allocator, caches, lazy imports, CPU clocks."""
    T = min(args.T, 32)
    for i in range(args.warmup):
        for k in args.k:
            for method in args.methods:
                if not _runnable(args, method, k):
                    continue
                prompt = prompts[-1 - i % len(prompts)]
                key = sample_key(args.tag + "|warmup", i)
                stream = message_bits(args.seed + 1, i, args.stream_factor * T)
                if method == "las":
                    import las_adapter
                    las_adapter.send_stream(env, k, prompt, stream, key, T)
                else:
                    send_stream(env, method, k, prompt, stream, key, T, coder=args.coder)
    TIMERS.clear()


def main():
    global DIAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True, help="model tag used in keys and output")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--unit", choices=("words", "chars"), required=True)
    ap.add_argument("--prompt-units", type=int, default=10,
                    help="prompt length in units (10: the Stepwise rule; other sizes drop texts that are too short)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--k", type=int, nargs="+", default=[16], help="0 = Full")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--filter-max-k", type=int, default=4096)
    ap.add_argument("--coder", choices=("ac", "rrc", "discop"), default="ac")
    ap.add_argument("--proj", choices=("dense", "sparse", "auto"), default="auto",
                    help="prefix-pair processing for coin and SyncPool (bit-identical outputs)")
    ap.add_argument("--proj-also", action="store_true",
                    help="coin: also run the other projection path on every law, timed apart (ms_proj_alt), "
                         "and stop if the two disagree")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (main setting: 1)")
    ap.add_argument("--top-p", type=float, default=1.0, help="nucleus mass before top-k (main setting: 1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps-out", default=None, help="per-step arrays, one line per text (--stop fixed)")
    ap.add_argument("--stop", choices=("message", "fixed"), default="message",
                    help="message: embed a 128-bit message and stop once it is decodable; "
                         "fixed: T tokens (+ UTF-8 completion) embedding a long random stream")
    ap.add_argument("--T", type=int, default=128, help="tokens per text with --stop fixed (+ UTF-8 completion)")
    ap.add_argument("--stream-factor", type=int, default=64, help="stream length = factor x T bits")
    ap.add_argument("--mwis-solver", choices=("official", "exact"), default="official")
    ap.add_argument("--no-recv", action="store_true")
    ap.add_argument("--recv-first", type=int, default=-1,
                    help="--stop fixed: run the receiver on samples < N only (Native: always); -1 = every sample")
    ap.add_argument("--skip-done", action="store_true", help="skip (tag, method, k, sample) already in --out")
    ap.add_argument("--lean", action="store_true",
                    help="timing run: no diagnostics (entropies, KL) inside or outside the timed code")
    ap.add_argument("--interleave", action="store_true",
                    help="loop k > sample > method, so slow drifts of the machine hit every method alike")
    ap.add_argument("--warmup", type=int, default=0, help="untimed warm-up texts per (method, k) before the run")
    args = ap.parse_args()
    DIAG = not args.lean

    done = set()
    if args.skip_done and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("kind") == "case":
                    done.add((r["tag"], _done_label(r["method"], r.get("mwis_solver")), r["k"], r["sample"]))

    torch.manual_seed(0)
    unknown = set(args.methods) - set(METHODS) - {"las"}
    if unknown or ("las" in args.methods and args.stop != "fixed"):
        raise SystemExit(f"unknown methods {sorted(unknown)} (las needs --stop fixed)")
    if (args.steps_out or args.interleave or args.warmup) and args.stop != "fixed":
        raise SystemExit("--steps-out, --interleave and --warmup need --stop fixed")
    env = Env(args.model_dir, proj=args.proj, temperature=args.temperature, top_p=args.top_p)
    env.mwis_solver = args.mwis_solver
    env.proj_also = args.proj_also
    prompts, dropped = load_prompts(args.prompts, args.unit, args.prompt_units)
    if args.start + args.samples > len(prompts):
        raise SystemExit(f"only {len(prompts)} prompts of {args.prompt_units} {args.unit}")
    meta = dict(kind="meta", tag=args.tag, model_dir=args.model_dir, tok_kind=env.tok_kind, V=env.V,
                prompts=len(prompts), prompts_dropped=dropped, args=vars(args),
                prefix_pairs=int(env.anc.size), visible=int(env.mask.sum()), **machine_meta(args.model_dir))
    print(json.dumps({a: b for a, b in meta.items() if a not in ("model_files", "model_digests")}), flush=True)
    samples = range(args.start, args.start + args.samples)
    cells = [(k, m) for k in args.k for m in args.methods if _runnable(args, m, k)]
    if args.interleave:
        order = [(k, m, s_) for k in args.k for s_ in samples for m in args.methods if (k, m) in cells]
    else:
        order = [(k, m, s_) for k, m in cells for s_ in samples]
    left = collections.Counter((k, m) for k, m, _ in order)
    stats = {c: dict(n=0, ok=0, complete=0, bpt=0.0, util=0.0) for c in cells}
    if args.warmup:
        warm_up(env, args, prompts)
    steps_fh = open(args.steps_out, "a", encoding="utf-8") if args.steps_out else None
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(meta) + "\n")
        for k, method, s_ in order:
            if (args.tag, _done_label(method, args.mwis_solver), k, s_) not in done:
                prompt = prompts[s_ % len(prompts)]
                key = sample_key(args.tag, s_)
                if args.stop == "fixed":
                    stream = message_bits(args.seed, s_, args.stream_factor * args.T)
                    run_fixed(env, args, fh, stats[(k, method)], method, k, s_, prompt, stream, key, steps_fh)
                else:
                    run_message(env, args, fh, stats[(k, method)], method, k, s_, prompt, key)
            left[(k, method)] -= 1
            if left[(k, method)] == 0:
                st = stats[(k, method)]
                c = max(st["complete"], 1)
                print(json.dumps(dict(tag=args.tag, method=method, k=k, n=st["n"], complete=st["complete"],
                                      checked=st.get("checked"), correct=st["ok"], bpt=round(st["bpt"] / c, 3),
                                      util=round(st["util"] / c, 3))), flush=True)
    if steps_fh is not None:
        steps_fh.close()


if __name__ == "__main__":
    main()
