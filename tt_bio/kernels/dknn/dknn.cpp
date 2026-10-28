// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// dknn: the outside-mask part of RFD3's atom neighbour index on the data-movement RISC-Vs, in integers.
// Coordinates arrive as int32 fixed point (|q| <= 2^30), so |dx| < 2^31 and dx^2 + dy^2 + dz^2 < 2^64 is an
// exact uint64 key. Per row: key of every atom, the k-th smallest key (quickselect), then the atoms
// outside the row's mask whose key is <= it; if more than n_fill, the n_fill smallest (ties to the lower index).
// That is what _neighbours_kdtree keeps from a k-nearest query. The row is written ascending, padded with L.
// The same source runs on BRISC and NCRISC; each takes its own row range and its own L1 buffers.
// Candidate mode (CW > 0, a design's static atoms): row i of the candidate table names a static row r, a key bound
// and that row's nearest static atoms; the keys are taken over those and the mover range [m0, m1) only. When the
// k-th of them is under the bound, no other atom can be within it (the host set the bound below every other static
// atom's key at this step's scale), so the row is the full one; otherwise the row is computed in full.
#include "api/dataflow/dataflow_api.h"

namespace {

inline uint64_t kth(uint64_t* a, int32_t n, int32_t t) {   // the t-th smallest (0-based) of a[0..n); reorders a
    int32_t lo = 0, hi = n - 1;
    while (lo < hi) {
        const uint64_t piv = a[(lo + hi) >> 1];
        int32_t i = lo, j = hi;
        while (i <= j) {
            while (a[i] < piv) ++i;
            while (a[j] > piv) --j;
            if (i <= j) {
                const uint64_t s = a[i]; a[i] = a[j]; a[j] = s;
                ++i; --j;
            }
        }
        if (t <= j) hi = j;
        else if (t >= i) lo = i;
        else break;
    }
    return a[t];
}

}  // namespace

void kernel_main() {
    constexpr uint32_t L = get_compile_time_arg_val(0);       // atoms
    constexpr uint32_t K = get_compile_time_arg_val(1);       // neighbours of the k-nearest query
    constexpr uint32_t MW = get_compile_time_arg_val(2);      // mask row words: [n_fill, bits of L...], padded to 8
    constexpr uint32_t WO = get_compile_time_arg_val(3);      // output row width (>= max n_fill), padded to 8
    constexpr uint32_t CB_X = get_compile_time_arg_val(4);    // this RISC-V's CBs: coords, keys, scratch, mask, out
    constexpr uint32_t CW = get_compile_time_arg_val(5);      // candidate row words [r, bound lo, hi, 0, ids...]; 0 = off
    constexpr uint32_t CB_D = CB_X + 1, CB_S = CB_X + 2, CB_M = CB_X + 3, CB_O = CB_X + 4, CB_C = CB_X + 5;
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto m_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();

    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nr = get_arg_val<uint32_t>(1);
    if (nr == 0) return;
    const auto sx = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), L * 16);
    const auto sm = TensorAccessor(m_args, get_common_arg_val<uint32_t>(1), MW * 4);
    const auto so = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2), WO * 4);
    const uint32_t o0 = get_common_arg_val<uint32_t>(3);     // first row of the output (a row range's start)
    const auto sc = TensorAccessor(c_args, get_common_arg_val<uint32_t>(4), (CW > 0 ? CW : 1) * 4);
    const uint32_t m0 = get_common_arg_val<uint32_t>(5), m1 = get_common_arg_val<uint32_t>(6);   // candidate mode movers

    const uint32_t x_l1 = get_write_ptr(CB_X), m_l1 = get_write_ptr(CB_M), o_l1 = get_write_ptr(CB_O);
    const uint32_t c_l1 = CW > 0 ? get_write_ptr(CB_C) : 0;
    noc_async_read(sx.get_noc_addr(0), x_l1, L * 16);
    noc_async_read_barrier();
    const int32_t* x = reinterpret_cast<const int32_t*>(x_l1);
    uint64_t* d = reinterpret_cast<uint64_t*>(get_write_ptr(CB_D));
    uint64_t* s = reinterpret_cast<uint64_t*>(get_write_ptr(CB_S));
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m_l1);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(o_l1);

    // The k-th key is bounded by a sample: the TS-th smallest key of every SS-th atom. Collect only the atoms at or
    // under that bound; if fewer than K made it, the bound was too tight and every atom is collected instead. Either
    // way the k-th key is the k-th of the collected keys, so the row is the same, and quickselect runs on ~2K keys.
    constexpr uint32_t KK = K < L ? K : L;
    constexpr uint32_t SS = 8, NS = (L + SS - 1) / SS, TS = 2 * KK / SS - 1;
    constexpr bool SAMPLED = L >= 4 * KK && TS < NS;

    // Candidate mode: the movers' bounding box, so a row can bound every mover's key from below with one key.
    int32_t lo[3] = {0, 0, 0}, hi[3] = {-1, -1, -1};
    if constexpr (CW > 0) {
        for (uint32_t j = m0; j < m1; ++j) {
            for (uint32_t a = 0; a < 3; ++a) {
                const int32_t v = x[4 * j + a];
                if (j == m0 || v < lo[a]) lo[a] = v;
                if (j == m0 || v > hi[a]) hi[a] = v;
            }
        }
    }
    auto box_key = [&](int32_t qx, int32_t qy, int32_t qz) {   // ~0 (never skip) when there are no movers
        if (m1 <= m0) return ~uint64_t(0);
        const int32_t q[3] = {qx, qy, qz};
        uint64_t b = 0;
        for (uint32_t a = 0; a < 3; ++a) {
            const uint32_t g = q[a] < lo[a] ? uint32_t(lo[a] - q[a]) : q[a] > hi[a] ? uint32_t(q[a] - hi[a]) : 0u;
            b += uint64_t(g) * g;
        }
        return b;
    };

    for (uint32_t i = r0; i < r0 + nr; ++i) {
        uint32_t r = i;
        uint64_t cbound = 0;
        volatile tt_l1_ptr uint32_t* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c_l1);
        if constexpr (CW > 0) {
            noc_async_read(sc.get_noc_addr(i), c_l1, CW * 4);
            noc_async_read_barrier();
            r = cw[0];
            cbound = uint64_t(cw[1]) | (uint64_t(cw[2]) << 32);
        }
        noc_async_read(sm.get_noc_addr(r), m_l1, MW * 4);
        const int32_t qx = x[4 * r], qy = x[4 * r + 1], qz = x[4 * r + 2];
        auto key = [&](uint32_t j) {
            const int32_t ax = x[4 * j] - qx, ay = x[4 * j + 1] - qy, az = x[4 * j + 2] - qz;
            const uint32_t ux = ax < 0 ? uint32_t(-ax) : uint32_t(ax);
            const uint32_t uy = ay < 0 ? uint32_t(-ay) : uint32_t(ay);
            const uint32_t uz = az < 0 ? uint32_t(-az) : uint32_t(az);
            return uint64_t(ux) * ux + uint64_t(uy) * uy + uint64_t(uz) * uz;
        };
        auto collect = [&](uint64_t bound) {   // (key, index) pairs with key <= bound, ascending by index
            uint32_t c = 0;
            for (uint32_t j = 0; j < L; ++j) {
                const uint64_t v = key(j);
                if (v <= bound) {
                    s[2 * c] = v;
                    s[2 * c + 1] = j;
                    ++c;
                }
            }
            return c;
        };
        uint32_t c = 0;
        uint64_t dk = 0;
        bool done = false;
        if constexpr (CW > 0) {
            // the row's static candidates, then the movers; the selection below does not depend on their order
            for (uint32_t t = 4; t < CW && cw[t] < L; ++t) {
                const uint32_t j = cw[t];
                s[2 * c] = key(j);
                s[2 * c + 1] = j;
                ++c;
            }
            // Adding atoms only lowers the k-th key, so a mover above the static candidates' k-th key cannot enter
            // the row: only movers at or under it are kept, and none are scored when the movers' box is beyond it.
            uint64_t mb = ~uint64_t(0);
            if (c >= KK) {
                for (uint32_t t = 0; t < c; ++t) d[t] = s[2 * t];
                mb = dk = kth(d, int32_t(c), int32_t(KK) - 1);
            }
            const uint32_t c0 = c;
            if (box_key(qx, qy, qz) <= mb) {
                for (uint32_t j = m0; j < m1; ++j) {
                    const uint64_t v = key(j);
                    if (v <= mb) {
                        s[2 * c] = v;
                        s[2 * c + 1] = j;
                        ++c;
                    }
                }
            }
            if (c >= KK) {
                if (c != c0) {
                    for (uint32_t t = 0; t < c; ++t) d[t] = s[2 * t];
                    dk = kth(d, int32_t(c), int32_t(KK) - 1);
                }
                done = dk < cbound;
            }
        }
        if (!done) {
            uint64_t bound = ~uint64_t(0);
            if constexpr (SAMPLED) {
                for (uint32_t t = 0; t < NS; ++t) d[t] = key(t * SS);
                bound = kth(d, int32_t(NS), int32_t(TS));
            }
            c = collect(bound);
            if (c < KK) c = collect(~uint64_t(0));
            for (uint32_t t = 0; t < c; ++t) d[t] = s[2 * t];
            dk = kth(d, int32_t(c), int32_t(KK) - 1);
        }
        noc_async_read_barrier();
        const uint32_t n_fill = m[0];
        // outside-mask candidates within the k-th key, compacted in place (still ascending by index)
        uint32_t c2 = 0;
        for (uint32_t t = 0; t < c; ++t) {
            const uint32_t j = uint32_t(s[2 * t + 1]);
            if (s[2 * t] <= dk && !((m[1 + (j >> 5)] >> (j & 31)) & 1u)) {
                s[2 * c2] = s[2 * t];
                s[2 * c2 + 1] = j;
                ++c2;
            }
        }
        c = c2;
        uint32_t w = 0;
        if (c <= n_fill) {
            for (uint32_t t = 0; t < c; ++t) o[w++] = uint32_t(s[2 * t + 1]);
        } else {
            // n_fill smallest of c (at most ~K) pairs: partial selection sort by (key, index), c is small
            for (uint32_t t = 0; t < n_fill; ++t) {
                uint32_t b = t;
                for (uint32_t u = t + 1; u < c; ++u) {
                    if (s[2 * u] < s[2 * b] || (s[2 * u] == s[2 * b] && s[2 * u + 1] < s[2 * b + 1])) b = u;
                }
                const uint64_t k0 = s[2 * t], i0 = s[2 * t + 1];
                s[2 * t] = s[2 * b]; s[2 * t + 1] = s[2 * b + 1];
                s[2 * b] = k0; s[2 * b + 1] = i0;
                o[w++] = uint32_t(s[2 * t + 1]);
            }
        }
        // ascending by index (insertion sort, w <= K), then pad
        for (uint32_t a = 1; a < w; ++a) {
            const uint32_t v = o[a];
            uint32_t b = a;
            while (b > 0 && o[b - 1] > v) { o[b] = o[b - 1]; --b; }
            o[b] = v;
        }
        for (uint32_t t = w; t < WO; ++t) o[t] = L;
        noc_async_write(o_l1, so.get_noc_addr(r - o0), WO * 4);
        noc_async_write_barrier();
    }
}
