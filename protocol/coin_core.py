"""Coin projection: self-contained core for the coin paper (no LAS runtime imports).

At every step the key law p (token masses merged over byte-identical tokens) is
projected onto a prefix-free support with one shared coin per string that has a
positive-mass proper extension.  The coin stops at v with probability p(v)/M(v),
where M(v) is the mass of v and of all its positive-mass extensions.  A string
survives when it stops and no positive-mass proper prefix of it stopped.  The
survivor v carries

    P~(v) = M(v) * prod_{a < v, p(a) > 0} M(a) / (M(a) - p(a)),

which sums to one for every coin vector and has E_coins[P~(v)] = p(v).

Two implementations return bit-identical results:
  coin_project         masks the whole static prefix-pair table, O(|V| + #pairs)
  coin_project_sparse  visits only the static ancestors of the support,
                       O(|S| + #pairs whose extension is in S)
Both run in float64 with numpy.bincount; the sparse version feeds every bincount
the same terms in the same order as the dense one, so each sum is identical.

Coins: u_v = AES-256_K(step || v) read as a 53-bit uniform, with K derived from the per-text key
(CoinPRF).  Each coin is one AES block computed on demand, and only strings with a positive-mass
extension (the nodes) ask for one, so the coins cost O(#nodes) per step, not O(|V|).
"""
from __future__ import annotations

import functools
import hashlib
import hmac

import numpy as np


# ---------------------------------------------------------------------------
# Shared coins
# ---------------------------------------------------------------------------

class CoinPRF:
    """Random-access coins: u_v(step) = AES-256_K(step || v), first 8 bytes >> 11, over 2^53.

    K = HMAC-SHA256(text key, label).  Distinct (step, v) are distinct AES blocks, so under the PRF
    assumption the coins are independent uniforms; any subset of them costs one AES block each."""

    def __init__(self, key: bytes, label: bytes = b"stegatext-coin-v2"):
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        k = hmac.new(key, label, hashlib.sha256).digest()
        self._enc = Cipher(algorithms.AES(k), modes.ECB()).encryptor()   # ECB: no state across calls

    def at(self, step: int, ids) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return np.zeros(0, dtype=np.float64)
        if ids.min() < 0 or not 0 <= step < 1 << 63:
            raise ValueError("coin index out of range")
        blk = np.empty((ids.size, 2), dtype=">u8")
        blk[:, 0] = step
        blk[:, 1] = ids
        raw = self._enc.update(blk.tobytes())
        hi = np.frombuffer(raw, dtype=">u8")[0::2]
        return (hi >> np.uint64(11)).astype(np.float64) / float(1 << 53)

    def coins(self, step: int):
        """Callable ids -> uniforms for one step (what the projections take)."""
        return functools.partial(self.at, step)


@functools.lru_cache(maxsize=4)
def coin_prf(key: bytes) -> CoinPRF:
    """Sender and receiver of one text reuse one cipher object (key schedule once per text)."""
    return CoinPRF(key)


def _coins_at(coins, ids: np.ndarray) -> np.ndarray:
    """coins: callable(ids) -> uniforms, or an array of uniforms indexed by id (tests)."""
    return coins(ids) if callable(coins) else np.asarray(coins, dtype=np.float64)[ids]


# ---------------------------------------------------------------------------
# Static key index per tokenizer
# ---------------------------------------------------------------------------

class CoinIndex:
    """Unique byte keys of the vocabulary and all their (proper prefix, extension) pairs.

    ``rep_of[i]``: representative token id of token i's byte string (-1 if empty).
    ``anc``/``desc``: pairs ordered by extension rep ascending, then prefix length
    ascending.  ``anc_ptr``: CSR offsets, the static ancestors of rep d are
    ``anc[anc_ptr[d]:anc_ptr[d+1]]``.
    """

    def __init__(self, table):
        self.table = table
        self.vocab_size = len(table)
        self.rep_of = np.full(self.vocab_size, -1, dtype=np.int64)
        self.key_to_rep: dict[bytes, int] = {}
        self.homographs: dict[int, list[int]] = {}
        for i, key in enumerate(table):
            if not key:
                continue
            rep = self.key_to_rep.setdefault(key, i)
            self.rep_of[i] = rep
            self.homographs.setdefault(rep, []).append(i)
        anc, desc = [], []
        self.max_key_len = 1
        for key, rep in self.key_to_rep.items():
            self.max_key_len = max(self.max_key_len, len(key))
            for length in range(1, len(key)):
                a = self.key_to_rep.get(key[:length])
                if a is not None:
                    anc.append(a)
                    desc.append(rep)
        self.anc = np.asarray(anc, dtype=np.int64)
        self.desc = np.asarray(desc, dtype=np.int64)
        if self.desc.size and (np.diff(self.desc) < 0).any():
            raise AssertionError("pair table must be grouped by extension rep ascending")
        counts = np.bincount(self.desc, minlength=self.vocab_size)
        self.anc_ptr = np.zeros(self.vocab_size + 1, dtype=np.int64)
        np.cumsum(counts, out=self.anc_ptr[1:])
        # scratch arrays for the sparse path (reset after each call)
        self._slot = np.full(self.vocab_size, -1, dtype=np.int64)

    def support_pairs(self, S: np.ndarray):
        """Live pairs (a, d) with d in S and a in S, in the dense table's order.

        ``S`` must be sorted ascending.  Returns compact positions (sa, sd) into S.
        """
        lo = self.anc_ptr[S]
        cnt = self.anc_ptr[S + 1] - lo
        tot = int(cnt.sum())
        if tot == 0:
            e = np.zeros(0, dtype=np.int64)
            return e, e
        start = np.repeat(lo - np.r_[0, np.cumsum(cnt)[:-1]], cnt)
        idx = start + np.arange(tot, dtype=np.int64)
        a = self.anc[idx]
        sd = np.repeat(np.arange(S.size, dtype=np.int64), cnt)
        slot = self._slot
        slot[S] = np.arange(S.size, dtype=np.int64)
        try:
            sa = slot[a]
        finally:
            slot[S] = -1
        live = sa >= 0
        return sa[live], sd[live]


# ---------------------------------------------------------------------------
# Projection: dense reference (whole pair table)
# ---------------------------------------------------------------------------

def _h2(q):
    with np.errstate(divide="ignore", invalid="ignore"):
        h2 = -(q * np.log2(q) + (1 - q) * np.log2(np.where(q < 1, 1 - q, 1.0)))
    return np.where((q > 0) & (q < 1), h2, 0.0)


def coin_project(p: np.ndarray, anc: np.ndarray, desc: np.ndarray, coins, diag: bool = True):
    """Project the key law ``p`` with coins (dense: masks the whole pair table).

    ``p``: float64 masses per key index (zeros allowed), summing to one.
    ``anc``/``desc``: every (proper-prefix, extension) pair of key indices.
    ``coins``: callable(ids) -> uniforms in [0, 1) (CoinPRF.coins(step)), or an array indexed by key id;
    asked only for the nodes.  ``diag=False`` skips the diagnostics (timing runs); the result is unchanged.
    Returns (survivor indices ascending, normalized survivor law, diagnostics or None).
    """
    supp = p > 0
    live = supp[anc] & supp[desc]
    a, d = anc[live], desc[live]
    n = p.size
    ext = np.bincount(a, weights=p[d], minlength=n)          # mass of proper extensions
    mass = p + ext                                            # M(v)
    node = np.zeros(n, dtype=bool)
    node[a] = True                                            # has a positive-mass extension
    nodes = np.flatnonzero(node)
    u = np.zeros(n)
    u[nodes] = _coins_at(coins, nodes)
    stop = supp & (~node | (u * mass < p))
    killed = np.zeros(n, dtype=bool)
    killed[d[stop[a]]] = True
    surv = stop & ~killed
    keep = surv[d]                                            # edges into survivors
    la = np.log1p(p[a[keep]] / ext[a[keep]])                  # log M(a)/(M(a)-p(a))
    logf = np.bincount(d[keep], weights=la, minlength=n)
    ids = np.flatnonzero(surv)
    w = mass[ids] * np.exp(logf[ids])
    total = float(w.sum())
    w = w / total
    if not diag:
        return ids, w, None
    nodes = np.flatnonzero(node & supp)
    q = p[nodes] / mass[nodes]
    h2 = _h2(q)
    pp = p[supp]
    has_anc = np.zeros(n, dtype=bool)
    has_anc[d] = True
    roots = np.flatnonzero(supp & ~has_anc)
    mr = mass[roots]
    diag = dict(
        coin_key_entropy_bits=float(-(pp * np.log2(pp)).sum()),
        coin_root_entropy_bits=float(-(mr[mr > 0] * np.log2(mr[mr > 0])).sum()),
        coin_stop_entropy_bits=float((mass[nodes] * h2).sum()),
        coin_projected_entropy_bits=float(-(w[w > 0] * np.log2(w[w > 0])).sum()),
        coin_nodes=int(nodes.size),
        coin_survivors=int(ids.size),
        coin_support_keys=int(supp.sum()),
        coin_unnormalized_total=total,
    )
    return ids, w, diag


# ---------------------------------------------------------------------------
# Projection: sparse (only the support and its static ancestors)
# ---------------------------------------------------------------------------

def coin_project_sparse(S: np.ndarray, pS: np.ndarray, index: CoinIndex, coins, diag: bool = True):
    """Same result as ``coin_project`` from the support only.

    ``S``: support key ids, sorted ascending; ``pS = p[S]`` (all positive); ``coins`` as in
    ``coin_project`` (asked for the node ids only).  Every array below is indexed by position in S.
    """
    m = S.size
    sa, sd = index.support_pairs(S)
    ext = np.bincount(sa, weights=pS[sd], minlength=m)
    mass = pS + ext
    node = np.zeros(m, dtype=bool)
    node[sa] = True
    npos = np.flatnonzero(node)
    uS = np.zeros(m)
    uS[npos] = _coins_at(coins, S[npos])
    stop = ~node | (uS * mass < pS)
    killed = np.zeros(m, dtype=bool)
    killed[sd[stop[sa]]] = True
    surv = stop & ~killed
    keep = surv[sd]
    la = np.log1p(pS[sa[keep]] / ext[sa[keep]])
    logf = np.bincount(sd[keep], weights=la, minlength=m)
    pos = np.flatnonzero(surv)
    w = mass[pos] * np.exp(logf[pos])
    total = float(w.sum())
    w = w / total
    if not diag:
        return S[pos], w, None
    q = pS[npos] / mass[npos]
    h2 = _h2(q)
    has_anc = np.zeros(m, dtype=bool)
    has_anc[sd] = True
    mr = mass[~has_anc]
    diag = dict(
        coin_key_entropy_bits=float(-(pS * np.log2(pS)).sum()),
        coin_root_entropy_bits=float(-(mr[mr > 0] * np.log2(mr[mr > 0])).sum()),
        coin_stop_entropy_bits=float((mass[npos] * h2).sum()),
        coin_projected_entropy_bits=float(-(w[w > 0] * np.log2(w[w > 0])).sum()),
        coin_nodes=int(npos.size),
        coin_survivors=int(pos.size),
        coin_support_keys=int(m),
        coin_unnormalized_total=total,
    )
    return S[pos], w, diag


def last_live_sparse(index: CoinIndex, S: np.ndarray, klen: np.ndarray, longest: bool):
    """(extension, ancestor) per live extension in S: its longest or shortest live prefix.

    Matches c4proto.Env.last_live on supp = indicator of S: extensions ascending.
    Ancestors of one extension are distinct proper prefixes, so their lengths are
    distinct and the shortest / longest is unique.
    """
    sa, sd = index.support_pairs(S)
    if not sd.size:
        e = np.zeros(0, dtype=np.int64)
        return e, e
    # within each extension the static order is prefix length ascending
    if longest:
        last = np.r_[sd[1:] != sd[:-1], True]
    else:
        last = np.r_[True, sd[1:] != sd[:-1]]
    return S[sd[last]], S[sa[last]]
