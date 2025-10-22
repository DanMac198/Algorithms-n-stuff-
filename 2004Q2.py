##Question 2 for the assignment
from typing import List, Tuple

class Analyser:
    """
    Function Description:
        Analyses a list of note sequences (strings of 'a'..'z') to support queries for the
        most frequent contiguous pattern of length K across *distinct songs*, counting each
        song at most once per pattern and treating all key-transposed versions of a pattern
        as the same. After preprocessing, getFrequentPattern(K) runs in O(K) time.

    Approach Description:
        Transposition means that two substrings match if their *consecutive differences*
        (d[i] = s[i+1]-s[i]) are identical. For a length-K pattern, this reduces to a
        length-(K-1) substring in the "delta string". For each K (2..M), we find the length
        (K-1) delta-substring that appears in the largest number of distinct songs.

        Preprocessing:
          1) Convert each song to int array (0..25) and build its delta array (values in -25..25, shifted to 0..50).
          2) For each K, on each song:
             - compute rolling hashes for all windows of length w=K-1 on the delta array;
             - use a *radix sort* to deduplicate per-song (count each pattern once per song);
             - append (hash, song_id, witness_position) to a global buffer.
             Then radix-sort the global buffer by hash and count how many distinct songs contain
             each hash. Track the argmax, and store a *witness* (song_id, start_pos) for this K.
          3) Store only: the original songs (O(NM) space), and per-K witnesses (O(M) space).
             All large buffers are reused and discarded between K’s ⇒ total space O(NM).

        Query:
          For a given K, return the witness substring (length K) from its song. Any transposed
          representative is acceptable; returning the actual occurrence is simplest and valid.

    Input:
        - sequences: list[str], N songs (each length ≤ M), characters in 'a'..'z'.

    Output:
        - getFrequentPattern(K): list[str] of length K (or [] if no song has length ≥ K).

    Time Complexity:
        - __init__: O(N M^2). For each K, we generate O(Σ_i (m_i)) hashes, dedup per-song via radix sort O(m_i),
          then globally radix sort O(Σ_i (m_i)); summed over K yields O(N M^2).
        - getFrequentPattern(K): O(K) to slice/return the witness substring.

    Aux Space Complexity:
        - After preprocessing: O(N M) to store the input sequences (as strings) and O(M) for per-K witnesses.
          Temporary buffers for a single K are O(N M) and are reused, so peak fits O(N M).

    Notes:
        - No Python dicts/sets are used (brief forbids them). We use fixed-size rolling hashes
          (two moduli) plus radix sort (counting sort by bytes) for dedup/frequency counting.
        - We count each song once per pattern by deduplicating the per-song list first.
        - If multiple patterns tie, *any* is returned as allowed by the brief.
    """

    # Two large prime moduli and base for rolling hash over delta alphabet (0..50)
    _MOD1 = 1_000_000_007
    _MOD2 = 1_000_000_009
    _BASE = 911382323  # arbitrary large odd < mod

    def __init__(self, sequences: List[str]):
        self.N = len(sequences)
        self.seqs = sequences[:]                 # store originals (O(NM) target space)
        self.M = 0                               # longest song length
        for s in sequences:
            if len(s) > self.M:
                self.M = len(s)

        # Convert each song to ints [0..25] and build delta arrays [0..50]
        self._vals: List[List[int]] = []         # per-song values (0..25)
        self._deltas: List[List[int]] = []       # per-song deltas (shifted by +25)
        for s in sequences:
            arr = [ord(c) - 97 for c in s]
            self._vals.append(arr)
            if len(arr) >= 2:
                d = [arr[i+1] - arr[i] + 25 for i in range(len(arr)-1)]  # shift to [0..50]
            else:
                d = []
            self._deltas.append(d)

        # Precompute powers for rolling hash up to max (M-1)
        maxW = self.M - 1 if self.M >= 1 else 0
        self._pow1 = [1]*(maxW+1)
        self._pow2 = [1]*(maxW+1)
        for i in range(1, maxW+1):
            self._pow1[i] = (self._pow1[i-1] * Analyser._BASE) % Analyser._MOD1
            self._pow2[i] = (self._pow2[i-1] * Analyser._BASE) % Analyser._MOD2

        # Precompute prefix hashes for every song’s delta array
        # pref1[i] = hash of first i items (0..i-1)
        self._pref1: List[List[int]] = []
        self._pref2: List[List[int]] = []
        for d in self._deltas:
            p1 = [0]*(len(d)+1)
            p2 = [0]*(len(d)+1)
            h1 = 0
            h2 = 0
            for x in d:
                h1 = (h1 * Analyser._BASE + (x + 1)) % Analyser._MOD1  # +1 to avoid leading zero ambiguity
                h2 = (h2 * Analyser._BASE + (x + 1)) % Analyser._MOD2
                p1[len(p1)-len(d)-1] = 0  # noop, keeps linter quiet; we assign below
                p1[p1.__len__()- (len(d) - ((len(d) - (len(p1)-1)) if False else 0))] = 0  # placeholder, ignore
            # The above two lines are placeholders to ensure no slicing; compute in simple loop:
            h1 = 0
            h2 = 0
            p1[0] = 0
            p2[0] = 0
            for i, x in enumerate(d, 1):
                h1 = (h1 * Analyser._BASE + (x + 1)) % Analyser._MOD1
                h2 = (h2 * Analyser._BASE + (x + 1)) % Analyser._MOD2
                p1[i] = h1
                p2[i] = h2
            self._pref1.append(p1)
            self._pref2.append(p2)

        # For each K in [2..M], compute winner witness (song_id, start_pos).
        # If no song has length ≥ K, we store (-1, -1).
        self._winner_song: List[int] = [-1]*(self.M+1)
        self._winner_pos:  List[int] = [-1]*(self.M+1)

        # Temporary buffers reused across K to keep peak memory O(NM)
        # We store tuples as parallel arrays to avoid Python tuple overhead.
        # Per-song windows (dedup buffer): (h1,h2,pos) arrays
        # Global buffer: (h1,h2,song,pos)
        # For radix sort we sort by h1 then by h2 (stable).
        # (Note: we keep integers non-negative; mod primes < 2^31, safe in 32-bit buckets.)

        # Main loop over K
        for K in range(2, self.M + 1):
            w = K - 1  # delta length
            total_windows = 0
            for sidx in range(self.N):
                m = len(self._deltas[sidx])
                if m >= w and w > 0:
                    total_windows += (m - w + 1)

            if total_windows == 0:
                self._winner_song[K] = -1
                self._winner_pos[K]  = -1
                continue

            # Step A: For each song, build list of window hashes and deduplicate per song
            # Store per-song unique items then append to global arrays
            # We will first count total unique to size arrays efficiently.
            # Worst-case unique per song is O(m), so we can allocate a safe upper bound = total_windows.

            gh1 = [0]*total_windows  # global hashes mod1
            gh2 = [0]*total_windows  # global hashes mod2
            gs  = [0]*total_windows  # global song id
            gp  = [0]*total_windows  # global pos (start in original song, not delta!)
            gsz = 0

            # temp buffers for one song
            # maximum m windows for one song; we reuse arrays sized to current song length.
            for sidx in range(self.N):
                d = self._deltas[sidx]
                m = len(d)
                if w == 0 or m < w:
                    continue
                # build per-song hashes and positions
                cnt = m - w + 1
                sh1 = [0]*cnt
                sh2 = [0]*cnt
                sp  = [0]*cnt  # start position in original song (substring of length K starts at i)
                # compute rolling via prefix: hash(i..i+w-1) = pref[i+w] - pref[i]*pow[w]
                p1 = self._pref1[sidx]
                p2 = self._pref2[sidx]
                pow1w = self._pow1[w]
                pow2w = self._pow2[w]
                for i in range(cnt):
                    h1 = (p1[i+w] - (p1[i] * pow1w) % Analyser._MOD1) % Analyser._MOD1
                    h2 = (p2[i+w] - (p2[i] * pow2w) % Analyser._MOD2) % Analyser._MOD2
                    sh1[i] = h1
                    sh2[i] = h2
                    sp[i]  = i  # start in delta; in original song substring starts at i

                # radix sort (stable) by h2 then by h1 to deduplicate
                self._radix_sort_pair_inplace(sh1, sh2)  # sorts by h1 then h2 (we'll implement that)

                # compact unique per song
                if cnt > 0:
                    uh1 = sh1[0]
                    uh2 = sh2[0]
                    up  = sp[0]
                    # Append first
                    gh1[gsz] = uh1; gh2[gsz] = uh2; gs[gsz] = sidx; gp[gsz] = up; gsz += 1
                    last1 = uh1; last2 = uh2
                    for i in range(1, cnt):
                        if sh1[i] != last1 or sh2[i] != last2:
                            uh1 = sh1[i]; uh2 = sh2[i]; up = sp[i]
                            gh1[gsz] = uh1; gh2[gsz] = uh2; gs[gsz] = sidx; gp[gsz] = up; gsz += 1
                            last1 = uh1; last2 = uh2

            # Step B: sort the global list by (h1,h2) and count distinct songs (already unique per-song)
            if gsz == 0:
                self._winner_song[K] = -1
                self._winner_pos[K]  = -1
                continue

            # trim arrays logically by gsz (we avoid slicing; just track gsz)
            self._radix_sort_pair_with_tie(gh1, gh2, gs, gp, gsz)

            # scan to count frequency and keep a witness
            best_count = 0
            best_song  = -1
            best_pos   = -1
            i = 0
            while i < gsz:
                j = i + 1
                # gh1/gh2 equal on [i..j-1]
                while j < gsz and gh1[j] == gh1[i] and gh2[j] == gh2[i]:
                    j += 1
                # since per song is unique, (j-i) = number of songs containing this pattern
                count = j - i
                if count > best_count:
                    best_count = count
                    best_song  = gs[i]
                    best_pos   = gp[i]
                i = j

            self._winner_song[K] = best_song
            self._winner_pos[K]  = best_pos

        # drop heavy temps: keep only sequences/vals for query reconstruction
        # (we already reuse buffers per K; nothing persistent to drop here)
        # optionally, clear deltas/hashes if you want stricter memory, but we may want to reconstruct:
        # We only need original sequences for reconstruction, so we can drop deltas/prefs to reduce space.
        self._deltas = []
        self._pref1 = []
        self._pref2 = []
        self._pow1 = []
        self._pow2 = []

    def getFrequentPattern(self, K: int) -> List[str]:
        """
        Function Description:
            Returns a list of characters (length K) of a most frequent pattern across songs,
            counting at most once per song and treating key-transposed versions as identical.
            If multiple patterns tie, any is returned. If no song has length >= K, returns [].

        Input:
            - K: target pattern length, 2 <= K <= M.

        Output:
            - list[str] of length K (characters 'a'..'z'), or [] if none exist.

        Time Complexity:
            O(K), by slicing the stored witness substring from its song and returning it.
        Aux Space Complexity:
            O(1) extra beyond the output list.
        """
        if K < 2 or K > self.M:
            return []
        sidx = self._winner_song[K]
        pos  = self._winner_pos[K]
        if sidx < 0 or pos < 0:
            return []
        song = self.seqs[sidx]
        if pos + K > len(song):
            # Should not happen; guard for safety if all songs < K
            return []
        # Return the actual occurrence (valid representative)
        out = [None]*K
        for i in range(K):
            out[i] = song[pos + i]
        return out

    # -----------------------------
    # Helpers: radix sort utilities
    # -----------------------------

    def _radix_sort_pair_inplace(self, a: List[int], b: List[int]) -> None:
        """
        Stable radix sort of arrays a,b (same length), sorting primarily by a then by b.
        Both arrays contain non-negative integers less than 2^31 (mod primes).
        We implement LSD radix in base 2^16 for practicality (two passes per 32-bit).
        """
        n = len(a)
        if n <= 1:
            return
        # temp buffers
        ta = [0]*n
        tb = [0]*n

        # First sort by lower 16 bits of b, then upper 16 bits of b
        self._counting_pass_16(b, a, tb, ta, n, 0)  # (key=b low16), carry a alongside
        self._counting_pass_16(tb, ta, b, a, n, 1)  # (key=b high16), swap back to (a,b)

        # Then stable sort by a (two 16-bit passes)
        self._counting_pass_16(a, b, ta, tb, n, 0)  # by a low16
        self._counting_pass_16(ta, tb, a, b, n, 1)  # by a high16 (back to a,b)

    def _radix_sort_pair_with_tie(self, a: List[int], b: List[int],
                                  s: List[int], p: List[int], n: int) -> None:
        """
        Stable radix sort of parallel arrays (a,b,s,p) over first key=a, second=b.
        We perform four counting passes of 16 bits: b low, b high, a low, a high,
        carrying the other arrays along to keep them aligned.
        """
        if n <= 1:
            return
        ta = [0]*n; tb = [0]*n; ts = [0]*n; tp = [0]*n

        # by b low16
        self._counting_pass_16_with_carry(b, a, s, p, tb, ta, ts, tp, n, 0)
        # by b high16
        self._counting_pass_16_with_carry(tb, ta, ts, tp, b, a, s, p, n, 1)
        # by a low16
        self._counting_pass_16_with_carry(a, b, s, p, ta, tb, ts, tp, n, 0)
        # by a high16
        self._counting_pass_16_with_carry(ta, tb, ts, tp, a, b, s, p, n, 1)

    def _counting_pass_16(self, key: List[int], carry: List[int],
                          out_key: List[int], out_carry: List[int],
                          n: int, half: int) -> None:
        """
        Counting sort by 16-bit slice (half=0: low 16, half=1: high 16).
        key/carry -> out_key/out_carry
        """
        MASK = 0xFFFF
        shift = 16*half
        # count
        cnt = [0]*65536
        for i in range(n):
            cnt[(key[i] >> shift) & MASK] += 1
        # prefix
        sumv = 0
        for v in range(65536):
            c = cnt[v]
            cnt[v] = sumv
            sumv += c
        # place
        for i in range(n):
            k = (key[i] >> shift) & MASK
            idx = cnt[k]
            out_key[idx] = key[i]
            out_carry[idx] = carry[i]
            cnt[k] = idx + 1

    def _counting_pass_16_with_carry(self, key: List[int], a: List[int], s: List[int], p: List[int],
                                     out_key: List[int], out_a: List[int], out_s: List[int], out_p: List[int],
                                     n: int, half: int) -> None:
        """
        Counting sort by 16-bit slice on 'key', carrying along a, s, p in lockstep.
        """
        MASK = 0xFFFF
        shift = 16*half
        cnt = [0]*65536
        for i in range(n):
            cnt[(key[i] >> shift) & MASK] += 1
        sumv = 0
        for v in range(65536):
            c = cnt[v]
            cnt[v] = sumv
            sumv += c
        for i in range(n):
            k = (key[i] >> shift) & MASK
            idx = cnt[k]
            out_key[idx] = key[i]
            out_a[idx]   = a[i]
            out_s[idx]   = s[i]
            out_p[idx]   = p[i]
            cnt[k] = idx + 1
