/* Native single-pass reduction of one closed native profiler chunk.
 *
 * Reads the raw profiler CSV (gzip or plain), hashes the decompressed bytes,
 * reduces kernel endpoints and zone totals with the same arithmetic as
 * perf/c10_burst_census/reduce_census.py span(), compacts recognized firmware
 * bracket markers into integer endpoints, and writes the payload JSONL records
 * to stdout plus a stats object to --stats-path. It publishes nothing: framing,
 * output caps, the footer, verification and publication stay in fast.py.
 *
 * Exit 0 reduced, 2 refused (cap, malformed, arithmetic), 3 unsupported input
 * shape. A double quote, a stray carriage return or a non-ASCII byte anywhere
 * makes CPython's csv module the only authority on the row split, so fast.py
 * runs its own reducer instead. Nothing reaches stdout before the whole input
 * has been read, so exit 3 always leaves an empty payload behind.
 *
 * Every hash key below is the exact composite key, not a digest of it, except
 * the operation and marker tables which verify the stored entry after probing.
 * No lookup can silently merge two different keys.
 */
#define _GNU_SOURCE
#include <inttypes.h>
#include <openssl/evp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define NCOL 15
#define NRISC 5
#define MAX_ZONES_PER_RISC 64
static const char *const RISC_NAMES[NRISC] = {"BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"};
/* Column order is asserted against the CSV header before any row is parsed. */
static const char *const COLS[NCOL] = {
    "PCIe slot", "core_x", "core_y", "RISC processor type", "timer_id",
    "time[cycles since reset]", "data", "run host ID", "trace id",
    "trace id counter", "zone name", "type", "source line", "source file", "meta data"};
static const char *const HEADER_PREFIX = "ARCH: blackhole, CHIP_FREQ[MHz]: 1350,";
enum { C_DEVICE, C_X, C_Y, C_RISC, C_TIMER, C_TICK, C_DATA, C_CALL, C_TRACE,
       C_REPLAY, C_ZONE, C_TYPE, C_LINE, C_FILE, C_META };
/* timer_id, data, source line, source file, meta data: everything a firmware
 * bracket row carries beyond its identity, core and tick. */
static const int CONSTANT_COLS[5] = {C_TIMER, C_DATA, C_LINE, C_FILE, C_META};
static const int NUMERIC_COLS[8] = {C_DEVICE, C_X, C_Y, C_TIMER, C_TICK, C_DATA, C_CALL, C_LINE};
#define GOLDEN 0x9e3779b97f4a7c15ULL

static void die(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("reduce STOP: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(code);
}
static void *xalloc(size_t n) {
    void *p = calloc(n ? n : 1, 1);
    if (!p) die(2, "out of memory (%zu bytes)", n);
    return p;
}
static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die(2, "out of memory (%zu bytes)", n);
    return q;
}
static inline uint64_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    return x ^ (x >> 33);
}
static uint64_t fnv(const char *s, uint32_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (uint32_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

/* ---------------------------------------------------------------- output */
static unsigned char *ob;
static size_t ob_len, ob_cap;
static uint64_t out_records;

static void ob_flush(void) {
    if (ob_len && fwrite(ob, 1, ob_len, stdout) != ob_len) die(2, "stdout write failed");
    ob_len = 0;
}
static inline void ob_put(const char *s, size_t n) {
    if (ob_len + n > ob_cap) {
        ob_flush();
        if (n > ob_cap) { ob_cap = n * 2; ob = xrealloc(ob, ob_cap); }
    }
    memcpy(ob + ob_len, s, n);
    ob_len += n;
}
static inline void ob_s(const char *s) { ob_put(s, strlen(s)); }
static inline void ob_c(char c) { ob_put(&c, 1); }
static void ob_u64(uint64_t v) {
    char t[24];
    int i = 24;
    if (!v) t[--i] = '0';
    while (v) { t[--i] = (char)('0' + v % 10); v /= 10; }
    ob_put(t + i, (size_t)(24 - i));
}
/* The chunk is known ASCII and quote-free by the time anything is emitted. */
static void ob_json(const char *s, uint32_t n) {
    ob_c('"');
    for (uint32_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\\') ob_s("\\\\");
        else if (c == '"') ob_s("\\\"");
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); ob_s(b); }
        else ob_c((char)c);
    }
    ob_c('"');
}

/* ------------------------------------------------------- string interning */
static char *arena;
static size_t arena_len = 1, arena_cap;   /* offset 0 is the empty string */
typedef struct { uint32_t off, len; } Str;
static Str *strs;
static uint32_t nstr = 1, str_cap;        /* id 0 is "" */
static uint32_t *str_ix, str_mask;
static inline const char *sptr(uint32_t id) { return arena + strs[id].off; }
static inline uint32_t slen(uint32_t id) { return strs[id].len; }

static uint32_t intern(const char *s, uint32_t n) {
    if (!n) return 0;
    uint64_t h = fnv(s, n);
    uint32_t i = (uint32_t)h & str_mask;
    for (;;) {
        uint32_t slot = str_ix[i];
        if (!slot) break;
        uint32_t id = slot;
        if (strs[id].len == n && !memcmp(arena + strs[id].off, s, n)) return id;
        i = (i + 1) & str_mask;
    }
    if (nstr + 1 >= str_cap) { str_cap *= 2; strs = xrealloc(strs, (size_t)str_cap * sizeof *strs); }
    if (nstr >= (1u << 20)) die(2, "more than 1M distinct interned strings");
    if (arena_len + n + 1 > arena_cap) {
        while (arena_len + n + 1 > arena_cap) arena_cap *= 2;
        arena = xrealloc(arena, arena_cap);   /* ids are offsets: the arena may move */
    }
    memcpy(arena + arena_len, s, n);
    arena[arena_len + n] = 0;
    strs[nstr].off = (uint32_t)arena_len;
    strs[nstr].len = n;
    arena_len += n + 1;
    str_ix[i] = nstr;
    return nstr++;
}

/* --------------------------------------------- exact-key open addressing */
typedef struct { uint64_t *key; uint32_t *val; uint32_t mask; } Map;
static void map_init(Map *m, uint64_t want) {
    uint64_t n = 16;
    while (n < want * 2) n *= 2;
    if (n > (1ULL << 31)) die(2, "hash table request too large");
    m->key = xalloc((size_t)n * sizeof *m->key);
    m->val = xalloc((size_t)n * sizeof *m->val);
    m->mask = (uint32_t)n - 1;
}
/* key must be the exact composite key, never a digest: equality is identity. */
static inline uint32_t map_find(Map *m, uint64_t key, uint32_t *found) {
    uint32_t i = (uint32_t)mix(key) & m->mask;
    for (;;) {
        if (!m->val[i]) { *found = 0; return i; }
        if (m->key[i] == key) { *found = m->val[i]; return i; }
        i = (i + 1) & m->mask;
    }
}

/* ----------------------------------------------------------------- limits */
static uint64_t max_rows = 4000000, max_operations = 4096, max_core_riscs = 600000,
                max_markers = 1024, max_line_bytes = 16384, max_raw_bytes = 512ULL << 20,
                max_verbatim_bytes = 64ULL << 20;
static int compact_firmware = 1;

/* ------------------------------------------------------------ operations */
typedef struct {
    uint32_t call, device, trace, replay;   /* interned */
    uint32_t domain;
    int32_t khead, ktail, shead, stail, fhead, ftail;
    uint32_t kcount, scount, fcount;
} Op;
static Op *ops;
static uint32_t nops;
static uint32_t *op_ix, op_mask;            /* hash -> op index + 1, verified */

typedef struct {
    uint32_t device, trace, replay, min_call, max_call;
    uint64_t min_tick, max_tick;
    int has_tick;
} Dom;
static Dom *doms;
static uint32_t ndom;

/* -------------------------------------------------------------- entries */
typedef struct {
    uint32_t cr; uint8_t have;
    uint64_t start, end;
    int32_t next;
} End;
typedef struct {
    uint32_t cr, zone;
    uint64_t value;
    int32_t next;
} Sum;
static End *kern, *fw;
static Sum *sums;
static uint32_t nkern = 1, nfw = 1, nsum = 1, kern_cap, fw_cap, sum_cap;  /* index 0 unused */
static uint32_t *kern_of, *fw_of;           /* cr id -> entry index, 0 = none */
static Map sum_map;
/* Dense (operation, core, RISC) ids. Every endpoint key below is exact. */
typedef struct { uint32_t op; uint16_t cx, cy; uint8_t risc; } CoreRisc;
static CoreRisc *crs;
static uint32_t ncr = 1;
static Map cr_map;

/* -------------------------------------------------------------- markers */
typedef struct {
    uint32_t risc, zone, type;
    int8_t risc_ix;
    int8_t route;                 /* 0 verbatim, 1 kernel, 2 sum, 3 firmware */
    int8_t is_start, is_end;
    uint64_t rows;
    uint32_t payload[5];
    int payload_constant;
} Marker;
static Marker *markers;
static uint32_t nmarker = 1;
static uint32_t *marker_ix, marker_mask;

/* ------------------------------------------------------------- verbatim */
typedef struct { uint64_t row; uint32_t off, len; } Verbatim;
static Verbatim *verb;
static uint32_t nverb, verb_cap;
static char *verb_arena;
static size_t verb_len, verb_cap_bytes;

static uint64_t rows_total, transitions, returns_total;
static uint64_t firmware_pairs, firmware_conflicts, firmware_verbatim;
static uint64_t unsupported_programs, unplaced_programs, unplaced_markers;
static uint32_t header_line;   /* interned, for the chunk provenance record */

static void fjson(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '\\' || c == '"') { fputc('\\', f); fputc((char)c, f); }
        else if (c < 0x20) fprintf(f, "\\u%04x", c);
        else fputc((char)c, f);
    }
    fputc('"', f);
}

static int block_is_ascii(const unsigned char *p, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        if (w & 0x8080808080808080ULL) return 0;
    }
    for (; i < n; i++)
        if (p[i] >= 0x80) return 0;
    return 1;
}

static int risc_index(const char *s, uint32_t n) {
    for (int i = 0; i < NRISC; i++)
        if (strlen(RISC_NAMES[i]) == n && !memcmp(RISC_NAMES[i], s, n)) return i;
    return -1;
}
static inline int all_digits(const char *s, uint32_t n) {
    if (!n) return 0;
    for (uint32_t i = 0; i < n; i++)
        if ((unsigned)(s[i] - '0') > 9u) return 0;
    return 1;
}
static uint64_t to_u64(const char *s, uint32_t n, const char *what) {
    uint64_t v = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (v > (UINT64_MAX - (uint64_t)(s[i] - '0')) / 10) die(2, "%s exceeds 64 bits", what);
        v = v * 10 + (uint64_t)(s[i] - '0');
    }
    return v;
}
/* Digit strings compare exactly by (length, bytes) at any magnitude. */
static int call_cmp(uint32_t a, uint32_t b) {
    if (slen(a) != slen(b)) return slen(a) < slen(b) ? -1 : 1;
    return memcmp(sptr(a), sptr(b), slen(a));
}
static int ends_with(uint32_t id, const char *suffix) {
    uint32_t n = (uint32_t)strlen(suffix);
    return slen(id) >= n && !memcmp(sptr(id) + slen(id) - n, suffix, n);
}
static int str_eq(uint32_t id, const char *s) {
    uint32_t n = (uint32_t)strlen(s);
    return slen(id) == n && !memcmp(sptr(id), s, n);
}

/* ---------------------------------------------------------- per-program */
typedef struct { uint64_t key; uint32_t ix; } Ord;
static int ord_cmp(const void *a, const void *b) {
    uint64_t x = ((const Ord *)a)->key, y = ((const Ord *)b)->key;
    return x < y ? -1 : x > y ? 1 : 0;
}
static Ord *kord, *sord, *ford;
static uint32_t kord_cap, sord_cap, ford_cap;
static void ord_need(Ord **p, uint32_t *cap, uint32_t want) {
    if (want > *cap) { *cap = want * 2 + 16; *p = xrealloc(*p, (size_t)*cap * sizeof **p); }
}
#define CORE_KEY(cx, cy, risc) (((uint64_t)(cx) << 48) | ((uint64_t)(cy) << 32) | ((uint64_t)(risc) << 24))
#define CR_CORE_KEY(i) CORE_KEY(crs[i].cx, crs[i].cy, crs[i].risc)

typedef struct { uint32_t zone; uint64_t value; } ZoneAcc;
static struct {
    uint64_t resident, blocked;
    uint32_t cores, nzone;
    ZoneAcc zone[MAX_ZONES_PER_RISC];
    int present;
} th[NRISC];

static void emit_program(uint32_t o) {
    Op *op = &ops[o];
    for (int r = 0; r < NRISC; r++) {
        th[r].resident = th[r].blocked = 0;
        th[r].cores = th[r].nzone = 0;
        th[r].present = 0;
    }
    uint64_t start = UINT64_MAX, end = 0, longest = 0;
    for (int32_t i = op->khead; i >= 0; i = kern[i].next) {
        End *e = &kern[i];
        CoreRisc *c = &crs[e->cr];
        if (e->have != 3 || e->end <= e->start)
            die(2, "Missing/reversed endpoints: call %s core (%u,%u) %s",
                sptr(op->call), c->cx, c->cy, RISC_NAMES[c->risc]);
        uint64_t own = e->end - e->start;
        if (e->start < start) start = e->start;
        if (e->end > end) end = e->end;
        if (own > longest) longest = own;
        th[c->risc].present = 1;
        th[c->risc].cores++;
        th[c->risc].resident += own;
    }
    /* Zone totals in insertion order, so first-seen zone order is exact. */
    ord_need(&sord, &sord_cap, op->scount);
    uint32_t nunplaced = 0, ns = 0;
    for (int32_t i = op->shead; i >= 0; i = sums[i].next) {
        Sum *s = &sums[i];
        CoreRisc *c = &crs[s->cr];
        sord[ns].key = CORE_KEY(c->cx, c->cy, c->risc) | ns;
        sord[ns].ix = (uint32_t)i;
        ns++;
        if (!th[c->risc].present) { nunplaced++; continue; }
        if (!kern_of[s->cr])
            die(2, "Sum without own RISC endpoints: call %s core (%u,%u) %s",
                sptr(op->call), c->cx, c->cy, RISC_NAMES[c->risc]);
        ZoneAcc *z = th[c->risc].zone;
        uint32_t n = th[c->risc].nzone, j = 0;
        while (j < n && z[j].zone != s->zone) j++;
        if (j == n) {
            if (n >= MAX_ZONES_PER_RISC) die(2, "more than %d distinct zones on one RISC", MAX_ZONES_PER_RISC);
            z[n].zone = s->zone;
            z[n].value = 0;
            th[c->risc].nzone = n + 1;
        }
        z[j].value += s->value;
        th[c->risc].blocked += s->value;
    }
    for (int r = 0; r < NRISC; r++)
        if (th[r].present && th[r].blocked > th[r].resident)
            die(2, "Zone sums exceed RISC residency: call %s %s", sptr(op->call), RISC_NAMES[r]);

    out_records++;
    ob_s("{\"kind\":\"program\",\"global_call_id\":");
    ob_s(sptr(op->call));
    ob_s(",\"device\":");
    ob_s(sptr(op->device));
    ob_s(",\"trace\":");
    if (slen(op->trace)) ob_json(sptr(op->trace), slen(op->trace)); else ob_s("null");
    ob_s(",\"replay\":");
    if (slen(op->replay)) ob_json(sptr(op->replay), slen(op->replay)); else ob_s("null");
    ob_s(",\"unplaced_zone_sums\":[");
    uint32_t written = 0;
    for (int32_t i = op->shead; i >= 0; i = sums[i].next) {
        Sum *s = &sums[i];
        CoreRisc *c = &crs[s->cr];
        if (th[c->risc].present) continue;
        if (written++) ob_c(',');
        ob_s("{\"core\":[");
        ob_u64(c->cx); ob_c(','); ob_u64(c->cy);
        ob_s("],\"risc\":"); ob_json(RISC_NAMES[c->risc], (uint32_t)strlen(RISC_NAMES[c->risc]));
        ob_s(",\"zone\":"); ob_json(sptr(s->zone), slen(s->zone));
        ob_s(",\"cycles\":"); ob_u64(s->value); ob_c('}');
    }
    if (written != nunplaced) die(2, "internal unplaced count mismatch");
    unplaced_markers += nunplaced;
    unplaced_programs += nunplaced != 0;
    ob_c(']');

    ord_need(&ford, &ford_cap, op->fcount);
    uint32_t nf = 0;
    for (int32_t i = op->fhead; i >= 0; i = fw[i].next) {
        if (fw[i].have == 3) firmware_pairs++;
        if (!kern_of[fw[i].cr]) {
            ford[nf].key = CR_CORE_KEY(fw[i].cr);
            ford[nf].ix = (uint32_t)i;
            nf++;
        }
    }
    qsort(ford, nf, sizeof *ford, ord_cmp);
    ob_s(",\"firmware_only_cores\":[");
    for (uint32_t i = 0; i < nf; i++) {
        End *e = &fw[ford[i].ix];
        CoreRisc *c = &crs[e->cr];
        if (i) ob_c(',');
        ob_s("{\"core\":["); ob_u64(c->cx); ob_c(','); ob_u64(c->cy);
        ob_s("],\"risc\":"); ob_json(RISC_NAMES[c->risc], (uint32_t)strlen(RISC_NAMES[c->risc]));
        ob_s(",\"firmware_start_cycle\":");
        if (e->have & 1) ob_u64(e->start); else ob_s("null");
        ob_s(",\"firmware_end_cycle\":");
        if (e->have & 2) ob_u64(e->end); else ob_s("null");
        ob_c('}');
    }
    ob_c(']');

    if (!op->kcount) {
        unsupported_programs++;
        ob_s(",\"status\":\"unsupported_no_kernel_endpoints\",\"summary\":null,\"cores\":[]}\n");
        return;
    }
    ob_s(",\"status\":");
    ob_s(nunplaced ? "\"reduced_with_unplaced_sums\"" : "\"reduced\"");
    ob_s(",\"summary\":{\"start_cycle\":");
    ob_u64(start);
    ob_s(",\"end_cycle\":"); ob_u64(end);
    ob_s(",\"span_cycles\":"); ob_u64(end - start);
    ord_need(&kord, &kord_cap, op->kcount);
    uint32_t nk = 0;
    for (int32_t i = op->khead; i >= 0; i = kern[i].next) {
        kord[nk].key = CR_CORE_KEY(kern[i].cr);
        kord[nk].ix = (uint32_t)i;
        nk++;
    }
    qsort(kord, nk, sizeof *kord, ord_cmp);
    qsort(sord, ns, sizeof *sord, ord_cmp);
    uint64_t active = 0;
    for (uint32_t i = 0; i < nk; i++)
        if (!i || (kord[i].key >> 32) != (kord[i - 1].key >> 32)) active++;
    ob_s(",\"active_cores\":"); ob_u64(active);
    ob_s(",\"longest_single_core_risc_span\":"); ob_u64(longest);
    ob_s(",\"threads\":{");
    int first = 1;
    for (int r = 0; r < NRISC; r++) {
        if (!th[r].present) continue;
        if (!first) ob_c(',');
        first = 0;
        ob_json(RISC_NAMES[r], (uint32_t)strlen(RISC_NAMES[r]));
        ob_s(":{\"cores\":"); ob_u64(th[r].cores);
        ob_s(",\"resident_core_cycles\":"); ob_u64(th[r].resident);
        ob_s(",\"zone_core_cycles\":{");
        for (uint32_t j = 0; j < th[r].nzone; j++) {
            if (j) ob_c(',');
            ob_json(sptr(th[r].zone[j].zone), slen(th[r].zone[j].zone));
            ob_c(':'); ob_u64(th[r].zone[j].value);
        }
        ob_s("},\"unclassified_core_cycles\":");
        ob_u64(th[r].resident - th[r].blocked);
        ob_c('}');
    }
    ob_s("}}");

    /* cores[]: kernel endpoints sorted by (core, RISC), merged with their sums. */
    ob_s(",\"cores\":[");
    uint32_t sp = 0;
    for (uint32_t i = 0; i < nk; i++) {
        End *e = &kern[kord[i].ix];
        CoreRisc *c = &crs[e->cr];
        uint64_t ck = kord[i].key;
        while (sp < ns && (sord[sp].key & ~0xffffffULL) < ck) sp++;
        if (i) ob_c(',');
        ob_s("{\"core\":["); ob_u64(c->cx); ob_c(','); ob_u64(c->cy);
        ob_s("],\"risc\":"); ob_json(RISC_NAMES[c->risc], (uint32_t)strlen(RISC_NAMES[c->risc]));
        ob_s(",\"start_cycle\":"); ob_u64(e->start);
        ob_s(",\"end_cycle\":"); ob_u64(e->end);
        uint64_t resident = e->end - e->start, blocked = 0;
        ob_s(",\"resident_cycles\":"); ob_u64(resident);
        ob_s(",\"zone_cycles\":{");
        uint32_t k = sp, m = 0;
        while (k < ns && (sord[k].key & ~0xffffffULL) == ck) {
            Sum *s = &sums[sord[k].ix];
            if (m++) ob_c(',');
            ob_json(sptr(s->zone), slen(s->zone));
            ob_c(':'); ob_u64(s->value);
            blocked += s->value;
            k++;
        }
        ob_c('}');
        if (blocked > resident)
            die(2, "Zones exceed own core/RISC residency: call %s core (%u,%u) %s",
                sptr(op->call), c->cx, c->cy, RISC_NAMES[c->risc]);
        ob_s(",\"unclassified_cycles\":"); ob_u64(resident - blocked);
        uint32_t fi = fw_of[e->cr];
        ob_s(",\"firmware_start_cycle\":");
        if (fi && (fw[fi].have & 1)) ob_u64(fw[fi].start); else ob_s("null");
        ob_s(",\"firmware_end_cycle\":");
        if (fi && (fw[fi].have & 2)) ob_u64(fw[fi].end); else ob_s("null");
        ob_c('}');
    }
    ob_s("]}\n");
}

/* ------------------------------------------------------------------ main */
int main(int argc, char **argv) {
    const char *path = NULL, *stats_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--stats-path") && i + 1 < argc) stats_path = argv[++i];
        else if (!strcmp(argv[i], "--no-compact-firmware")) compact_firmware = 0;
        else if (!strcmp(argv[i], "--max-rows") && i + 1 < argc) max_rows = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--max-operations") && i + 1 < argc) max_operations = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--max-core-riscs") && i + 1 < argc) max_core_riscs = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--max-markers") && i + 1 < argc) max_markers = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--max-line-bytes") && i + 1 < argc) max_line_bytes = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--max-raw-bytes") && i + 1 < argc) max_raw_bytes = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--max-verbatim-bytes") && i + 1 < argc) max_verbatim_bytes = strtoull(argv[++i], 0, 10);
        else if (!path) path = argv[i];
        else die(2, "unexpected argument %s", argv[i]);
    }
    if (!path || !stats_path) die(2, "usage: reduce RAW --stats-path FILE [limits]");
    if (!max_rows || !max_operations || !max_core_riscs || !max_markers || !max_line_bytes
        || !max_raw_bytes || !max_verbatim_bytes) die(2, "Limits must be positive");
    /* The exact composite keys below pack the operation index into 16 bits and
     * the (operation, core, RISC) id into 20 bits. */
    if (max_operations > 0xffff) die(2, "max_operations above 65535");
    if (max_core_riscs >= (1u << 20)) die(2, "max_core_riscs at or above 1048576");

    ob_cap = 1 << 20; ob = xalloc(ob_cap);
    arena_cap = 1 << 16; arena = xalloc(arena_cap);
    str_cap = 4096; strs = xalloc((size_t)str_cap * sizeof *strs);
    str_mask = (1u << 17) - 1; str_ix = xalloc(((size_t)str_mask + 1) * sizeof *str_ix);
    ops = xalloc((size_t)(max_operations + 1) * sizeof *ops);
    doms = xalloc((size_t)(max_operations + 1) * sizeof *doms);
    op_mask = 1; while (op_mask < max_operations * 2) op_mask *= 2;
    op_ix = xalloc((size_t)op_mask * sizeof *op_ix); op_mask -= 1;
    markers = xalloc((size_t)(max_markers + 1) * sizeof *markers);
    marker_mask = 1; while (marker_mask < max_markers * 2) marker_mask *= 2;
    marker_ix = xalloc((size_t)marker_mask * sizeof *marker_ix); marker_mask -= 1;
    crs = xalloc((size_t)(max_core_riscs + 1) * sizeof *crs);
    kern_of = xalloc((size_t)(max_core_riscs + 1) * sizeof *kern_of);
    fw_of = xalloc((size_t)(max_core_riscs + 1) * sizeof *fw_of);
    map_init(&cr_map, max_core_riscs);
    map_init(&sum_map, max_core_riscs);
    kern_cap = fw_cap = sum_cap = 1 << 16;
    kern = xalloc((size_t)kern_cap * sizeof *kern);
    fw = xalloc((size_t)fw_cap * sizeof *fw);
    sums = xalloc((size_t)sum_cap * sizeof *sums);
    verb_cap = 1024; verb = xalloc((size_t)verb_cap * sizeof *verb);
    verb_cap_bytes = 1 << 16; verb_arena = xalloc(verb_cap_bytes);

    gzFile gz = gzopen(path, "rb");
    if (!gz) die(2, "cannot open %s", path);
    gzbuffer(gz, 1 << 20);
    EVP_MD_CTX *md = EVP_MD_CTX_new();
    if (!md || !EVP_DigestInit_ex(md, EVP_sha256(), NULL)) die(2, "sha256 init failed");

    size_t cap = 4u << 20, len = 0, pos = 0;
    char *buf = xalloc(cap);
    uint64_t raw_bytes = 0, longest_line = 0;
    int saw_header = 0, saw_columns = 0, eof = 0;
    uint32_t cur_op = 0, prev_call = 0, prev_device = 0, prev_trace = 0, prev_replay = 0;
    int have_prev = 0;
    uint64_t prev_core_key = UINT64_MAX;
    uint32_t prev_cr = 0;

    while (!eof) {
        if (pos) { memmove(buf, buf + pos, len - pos); len -= pos; pos = 0; }
        if (len == cap) { cap *= 2; buf = xrealloc(buf, cap); }
        int got = gzread(gz, buf + len, (unsigned)(cap - len));
        if (got < 0) { int e; const char *m = gzerror(gz, &e); die(2, "gzip read failed: %s", m); }
        if (!got) eof = 1;
        else {
            if (!EVP_DigestUpdate(md, buf + len, (size_t)got)) die(2, "sha256 update failed");
            raw_bytes += (uint64_t)got;
            if (raw_bytes > max_raw_bytes) die(2, "max_raw_bytes exceeded");
            /* Refuse before anything is parsed: on these shapes CPython's csv module
             * is the only authority on the row split. Nothing has reached stdout. */
            if (!block_is_ascii((unsigned char *)buf + len, (size_t)got))
                die(3, "non-ASCII byte in input; use the Python reducer");
            if (memchr(buf + len, '"', (size_t)got))
                die(3, "double quote in input; use the Python reducer");
            len += (size_t)got;
        }
        for (;;) {
            char *nl = memchr(buf + pos, '\n', len - pos);
            if (!nl) {
                if (eof && len > pos) die(2, "Truncated line: missing final newline");
                break;
            }
            size_t line_len = (size_t)(nl - (buf + pos));
            if (line_len + 1 > longest_line) longest_line = line_len + 1;
            if (longest_line > max_line_bytes) die(2, "max_line_bytes exceeded");
            char *line = buf + pos;
            pos += line_len + 1;
            if (line_len && line[line_len - 1] == '\r') line_len--;
            if (memchr(line, '\r', line_len))
                die(3, "carriage return inside a row; use the Python reducer");
            if (!saw_header) {
                saw_header = 1;
                if (line_len < strlen(HEADER_PREFIX) || memcmp(line, HEADER_PREFIX, strlen(HEADER_PREFIX)))
                    die(2, "Unsupported profiler header");
                header_line = intern(line, (uint32_t)line_len);
                continue;
            }
            /* Field split: comma delimited, spaces after a delimiter skipped, as
             * csv.reader(skipinitialspace=True) does on quote-free ASCII input. */
            const char *f[NCOL];
            uint32_t fl[NCOL];
            int nf = 0;
            {
                size_t i = 0;
                while (nf < NCOL) {
                    while (i < line_len && line[i] == ' ') i++;
                    size_t s = i;
                    while (i < line_len && line[i] != ',') i++;
                    f[nf] = line + s;
                    fl[nf] = (uint32_t)(i - s);
                    nf++;
                    if (i >= line_len) break;
                    i++;
                    if (i == line_len) {                   /* trailing empty field */
                        if (nf < NCOL) { f[nf] = line + i; fl[nf] = 0; nf++; }
                        else nf = NCOL + 1;
                        break;
                    }
                }
                if (i < line_len) nf = NCOL + 1;
            }
            if (!saw_columns) {
                saw_columns = 1;
                if (nf != NCOL) die(2, "Unexpected CSV columns");
                for (int i = 0; i < NCOL; i++)
                    if (strlen(COLS[i]) != fl[i] || memcmp(COLS[i], f[i], fl[i]))
                        die(2, "Unexpected CSV columns");
                continue;
            }
            rows_total++;
            if (rows_total > max_rows) die(2, "max_rows exceeded");
            if (nf != NCOL) die(2, "Malformed CSV row %" PRIu64 ": %d columns", rows_total, nf);
            for (int i = 0; i < 8; i++)
                if (!all_digits(f[NUMERIC_COLS[i]], fl[NUMERIC_COLS[i]]))
                    die(2, "Not an integer field in row %" PRIu64 " column %d", rows_total, NUMERIC_COLS[i]);
            if (fl[C_TRACE] || fl[C_REPLAY]) {
                if (!fl[C_TRACE] || !fl[C_REPLAY]) die(2, "Incomplete trace/replay identity");
                if (!all_digits(f[C_TRACE], fl[C_TRACE]) || !all_digits(f[C_REPLAY], fl[C_REPLAY]))
                    die(2, "Not an integer trace identity in row %" PRIu64, rows_total);
            }
            uint64_t tick = to_u64(f[C_TICK], fl[C_TICK], "tick");

            int changed = !have_prev
                || fl[C_CALL] != slen(prev_call) || memcmp(f[C_CALL], sptr(prev_call), fl[C_CALL])
                || fl[C_DEVICE] != slen(prev_device) || memcmp(f[C_DEVICE], sptr(prev_device), fl[C_DEVICE])
                || fl[C_TRACE] != slen(prev_trace) || memcmp(f[C_TRACE], sptr(prev_trace), fl[C_TRACE])
                || fl[C_REPLAY] != slen(prev_replay) || memcmp(f[C_REPLAY], sptr(prev_replay), fl[C_REPLAY]);
            if (changed) {
                transitions++;
                uint32_t call = intern(f[C_CALL], fl[C_CALL]);
                uint32_t device = intern(f[C_DEVICE], fl[C_DEVICE]);
                uint32_t trace = intern(f[C_TRACE], fl[C_TRACE]);
                uint32_t replay = intern(f[C_REPLAY], fl[C_REPLAY]);
                if ((slen(call) > 1 && sptr(call)[0] == '0') || (slen(device) > 1 && sptr(device)[0] == '0'))
                    die(2, "Ambiguous zero-padded identity encoding in row %" PRIu64, rows_total);
                uint64_t h = mix(((uint64_t)call << 42) ^ ((uint64_t)device << 28)
                                 ^ ((uint64_t)trace << 14) ^ replay);
                uint32_t i = (uint32_t)h & op_mask, hit = 0;
                for (;;) {
                    uint32_t slot = op_ix[i];
                    if (!slot) break;
                    Op *c = &ops[slot];
                    if (c->call == call && c->device == device && c->trace == trace && c->replay == replay) {
                        hit = slot;
                        break;
                    }
                    i = (i + 1) & op_mask;
                }
                if (hit) { cur_op = hit; returns_total++; }
                else {
                    if (nops >= max_operations) die(2, "max_operations exceeded");
                    cur_op = ++nops;
                    Op *op = &ops[cur_op];
                    op->call = call; op->device = device; op->trace = trace; op->replay = replay;
                    op->khead = op->ktail = op->shead = op->stail = op->fhead = op->ftail = -1;
                    uint32_t d = 0;
                    for (uint32_t j = 1; j <= ndom; j++)
                        if (doms[j].device == device && doms[j].trace == trace && doms[j].replay == replay) { d = j; break; }
                    if (!d) {
                        d = ++ndom;
                        doms[d].device = device; doms[d].trace = trace; doms[d].replay = replay;
                        doms[d].min_call = doms[d].max_call = call;
                    }
                    op->domain = d;
                    op_ix[i] = cur_op;
                }
                prev_call = call; prev_device = device; prev_trace = trace; prev_replay = replay;
                have_prev = 1;
                Dom *d = &doms[ops[cur_op].domain];
                if (call_cmp(call, d->min_call) < 0) d->min_call = call;
                if (call_cmp(call, d->max_call) > 0) d->max_call = call;
            }
            Dom *dom = &doms[ops[cur_op].domain];
            if (!dom->has_tick) { dom->min_tick = dom->max_tick = tick; dom->has_tick = 1; }
            else if (tick < dom->min_tick) dom->min_tick = tick;
            else if (tick > dom->max_tick) dom->max_tick = tick;

            /* marker census; the row's routing is decided once per marker combo */
            uint64_t mh = mix(fnv(f[C_RISC], fl[C_RISC]) ^ (fnv(f[C_ZONE], fl[C_ZONE]) * GOLDEN)
                              ^ fnv(f[C_TYPE], fl[C_TYPE]));
            uint32_t mi = 0, i = (uint32_t)mh & marker_mask;
            for (;;) {
                uint32_t slot = marker_ix[i];
                if (!slot) break;
                Marker *c = &markers[slot];
                if (slen(c->risc) == fl[C_RISC] && !memcmp(sptr(c->risc), f[C_RISC], fl[C_RISC])
                    && slen(c->zone) == fl[C_ZONE] && !memcmp(sptr(c->zone), f[C_ZONE], fl[C_ZONE])
                    && slen(c->type) == fl[C_TYPE] && !memcmp(sptr(c->type), f[C_TYPE], fl[C_TYPE])) {
                    mi = slot;
                    break;
                }
                i = (i + 1) & marker_mask;
            }
            if (!mi) {
                if (nmarker > max_markers) die(2, "max_markers exceeded");
                mi = nmarker++;
                Marker *m = &markers[mi];
                m->risc = intern(f[C_RISC], fl[C_RISC]);
                m->zone = intern(f[C_ZONE], fl[C_ZONE]);
                m->type = intern(f[C_TYPE], fl[C_TYPE]);
                m->risc_ix = (int8_t)risc_index(f[C_RISC], fl[C_RISC]);
                m->is_start = str_eq(m->type, "ZONE_START");
                m->is_end = str_eq(m->type, "ZONE_END");
                m->rows = 0;
                m->payload_constant = 1;
                for (int j = 0; j < 5; j++) m->payload[j] = intern(f[CONSTANT_COLS[j]], fl[CONSTANT_COLS[j]]);
                /* Same precedence as analyze.read_raw: a -KERNEL zone is an
                 * endpoint, then any ZONE_TOTAL is a sum, then a -FW bracket. */
                if (m->risc_ix < 0) m->route = 0;
                else if (ends_with(m->zone, "-KERNEL")) m->route = 1;
                else if (str_eq(m->type, "ZONE_TOTAL")) m->route = 2;
                else if (compact_firmware && ends_with(m->zone, "-FW") && (m->is_start || m->is_end)) m->route = 3;
                else m->route = 0;
                marker_ix[i] = mi;
            }
            Marker *mrk = &markers[mi];
            mrk->rows++;

            int reduced = 0;
            if (mrk->route) {
                uint64_t cx = to_u64(f[C_X], fl[C_X], "core_x"), cy = to_u64(f[C_Y], fl[C_Y], "core_y");
                if (cx > 0xffff || cy > 0xffff) die(2, "core coordinate out of range");
                uint64_t ckey = ((uint64_t)cur_op << 40) | (cx << 24) | (cy << 8) | (uint64_t)mrk->risc_ix;
                if (ckey != prev_core_key) {
                    uint32_t found, slot = map_find(&cr_map, ckey, &found);
                    if (!found) {
                        if (ncr > max_core_riscs) die(2, "max_core_riscs exceeded");
                        found = ncr++;
                        crs[found].op = cur_op;
                        crs[found].cx = (uint16_t)cx;
                        crs[found].cy = (uint16_t)cy;
                        crs[found].risc = (uint8_t)mrk->risc_ix;
                        cr_map.key[slot] = ckey;
                        cr_map.val[slot] = found;
                    }
                    prev_core_key = ckey;
                    prev_cr = found;
                }
                uint32_t cr = prev_cr;
                Op *op = &ops[cur_op];
                if (mrk->route == 1) {
                    uint32_t ki = kern_of[cr];
                    if (!ki) {
                        if (nkern == kern_cap) { kern_cap *= 2; kern = xrealloc(kern, (size_t)kern_cap * sizeof *kern); }
                        ki = nkern++;
                        kern[ki].cr = cr; kern[ki].have = 0; kern[ki].next = -1;
                        kern_of[cr] = ki;
                        if (op->ktail < 0) op->khead = (int32_t)ki; else kern[op->ktail].next = (int32_t)ki;
                        op->ktail = (int32_t)ki;
                        op->kcount++;
                    }
                    uint8_t bit = mrk->is_start ? 1 : mrk->is_end ? 2 : 0;
                    if (!bit || (kern[ki].have & bit))
                        die(2, "Duplicate/invalid kernel marker: call %s core (%" PRIu64 ",%" PRIu64 ") %s",
                            sptr(op->call), cx, cy, RISC_NAMES[mrk->risc_ix]);
                    if (bit == 1) kern[ki].start = tick; else kern[ki].end = tick;
                    kern[ki].have |= bit;
                    reduced = 1;
                } else if (mrk->route == 2) {
                    uint64_t skey = ((uint64_t)cr << 20) | mrk->zone;
                    uint32_t found, slot = map_find(&sum_map, skey, &found);
                    if (found)
                        die(2, "Duplicate sum marker: call %s core (%" PRIu64 ",%" PRIu64 ") %s %s",
                            sptr(op->call), cx, cy, RISC_NAMES[mrk->risc_ix], sptr(mrk->zone));
                    if (nsum == sum_cap) { sum_cap *= 2; sums = xrealloc(sums, (size_t)sum_cap * sizeof *sums); }
                    uint32_t si = nsum++;
                    sums[si].cr = cr; sums[si].zone = mrk->zone; sums[si].next = -1;
                    sums[si].value = to_u64(f[C_DATA], fl[C_DATA], "data");
                    sum_map.key[slot] = skey;
                    sum_map.val[slot] = si;
                    if (op->stail < 0) op->shead = (int32_t)si; else sums[op->stail].next = (int32_t)si;
                    op->stail = (int32_t)si;
                    op->scount++;
                    reduced = 1;
                } else {
                    uint32_t fi = fw_of[cr];
                    if (!fi) {
                        if (nfw == fw_cap) { fw_cap *= 2; fw = xrealloc(fw, (size_t)fw_cap * sizeof *fw); }
                        fi = nfw++;
                        fw[fi].cr = cr; fw[fi].have = 0; fw[fi].next = -1;
                        fw_of[cr] = fi;
                        if (op->ftail < 0) op->fhead = (int32_t)fi; else fw[op->ftail].next = (int32_t)fi;
                        op->ftail = (int32_t)fi;
                        op->fcount++;
                    }
                    uint8_t bit = mrk->is_start ? 1 : 2;
                    if (fw[fi].have & bit) firmware_conflicts++;   /* keep the row whole instead */
                    else {
                        if (bit == 1) fw[fi].start = tick; else fw[fi].end = tick;
                        fw[fi].have |= bit;
                        reduced = 1;
                        for (int j = 0; j < 5; j++) {
                            uint32_t p = mrk->payload[j];
                            if (slen(p) != fl[CONSTANT_COLS[j]]
                                || memcmp(sptr(p), f[CONSTANT_COLS[j]], slen(p))) {
                                /* The combo is not constant after all: keep this row whole too. */
                                mrk->payload_constant = 0;
                                firmware_verbatim++;
                                reduced = 0;
                                break;
                            }
                        }
                    }
                }
            }
            if (!reduced) {
                if (nverb == verb_cap) { verb_cap *= 2; verb = xrealloc(verb, (size_t)verb_cap * sizeof *verb); }
                if (verb_len + line_len + 1 > verb_cap_bytes) {
                    while (verb_len + line_len + 1 > verb_cap_bytes) verb_cap_bytes *= 2;
                    if (verb_cap_bytes > max_verbatim_bytes)
                        die(3, "verbatim retention above max_verbatim_bytes; use the Python reducer");
                    verb_arena = xrealloc(verb_arena, verb_cap_bytes);
                }
                verb[nverb].row = rows_total;
                verb[nverb].off = (uint32_t)verb_len;
                verb[nverb].len = (uint32_t)line_len;
                memcpy(verb_arena + verb_len, line, line_len);
                verb_len += line_len;
                verb_arena[verb_len++] = 0;
                nverb++;
            }
        }
    }
    gzclose(gz);
    if (!saw_header || !saw_columns || !rows_total) die(2, "Empty chunk");

    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned dlen = 0;
    if (!EVP_DigestFinal_ex(md, digest, &dlen) || dlen != 32) die(2, "sha256 final failed");
    EVP_MD_CTX_free(md);

    for (uint32_t o = 1; o <= nops; o++) emit_program(o);
    for (uint32_t i = 0; i < nverb; i++) {
        char *line = verb_arena + verb[i].off;
        size_t n = verb[i].len, p = 0;
        out_records++;
        ob_s("{\"kind\":\"unreduced_marker\",\"row\":");
        ob_u64(verb[i].row);
        ob_s(",\"values\":[");
        for (int c = 0; c < NCOL; c++) {
            while (p < n && line[p] == ' ') p++;
            size_t s = p;
            while (p < n && line[p] != ',') p++;
            if (c) ob_c(',');
            ob_json(line + s, (uint32_t)(p - s));
            if (p < n) p++;
        }
        ob_s("]}\n");
    }
    ob_flush();
    if (fflush(stdout)) die(2, "stdout flush failed");

    FILE *st = fopen(stats_path, "w");
    if (!st) die(2, "cannot write %s", stats_path);
    fputs("{\"header\":", st);
    fjson(st, sptr(header_line));
    fputs(",\"raw_sha256\":\"", st);
    for (int i = 0; i < 32; i++) fprintf(st, "%02x", digest[i]);
    fprintf(st, "\",\"raw_bytes\":%" PRIu64 ",\"raw_rows\":%" PRIu64 ",\"records\":%" PRIu64
                ",\"programs\":%u,\"call_transitions\":%" PRIu64 ",\"noncontiguous_returns\":%" PRIu64
                ",\"distinct_core_riscs\":%u,\"firmware_bracket_pairs\":%" PRIu64
                ",\"firmware_bracket_conflicts\":%" PRIu64 ",\"firmware_rows_retained_verbatim\":%" PRIu64
                ",\"verbatim_records\":%u,\"programs_without_endpoints\":%" PRIu64
                ",\"programs_with_unplaced_sums\":%" PRIu64 ",\"unplaced_sum_markers\":%" PRIu64
                ",\"compact_firmware\":%s,\"ascii_only_input\":true",
            raw_bytes, rows_total, out_records, nops, transitions, returns_total,
            ncr - 1, firmware_pairs, firmware_conflicts, firmware_verbatim,
            nverb, unsupported_programs, unplaced_programs, unplaced_markers,
            compact_firmware ? "true" : "false");
    fputs(",\"counter_ranges\":[", st);
    for (uint32_t i = 1; i <= ndom; i++) {
        Dom *d = &doms[i];
        fprintf(st, "%s{\"device\":%s,\"trace\":", i > 1 ? "," : "", sptr(d->device));
        if (slen(d->trace)) fjson(st, sptr(d->trace)); else fputs("null", st);
        fputs(",\"replay\":", st);
        if (slen(d->replay)) fjson(st, sptr(d->replay)); else fputs("null", st);
        fprintf(st, ",\"min_call_id\":%s,\"max_call_id\":%s,\"min_tick\":%" PRIu64 ",\"max_tick\":%" PRIu64 "}",
                sptr(d->min_call), sptr(d->max_call), d->min_tick, d->max_tick);
    }
    fputs("],\"markers\":[", st);
    for (uint32_t i = 1; i < nmarker; i++) {
        Marker *m = &markers[i];
        fputs(i > 1 ? ",{\"risc\":" : "{\"risc\":", st);
        fjson(st, sptr(m->risc));
        fputs(",\"zone\":", st);
        fjson(st, sptr(m->zone));
        fputs(",\"type\":", st);
        fjson(st, sptr(m->type));
        fprintf(st, ",\"rows\":%" PRIu64 ",\"route\":%d,\"payload_constant\":%s,\"payload\":[",
                m->rows, m->route, m->payload_constant ? "true" : "false");
        for (int j = 0; j < 5; j++) { if (j) fputc(',', st); fjson(st, sptr(m->payload[j])); }
        fputs("]}", st);
    }
    fputs("]}\n", st);
    if (fflush(st) || fclose(st)) die(2, "cannot flush %s", stats_path);
    return 0;
}
