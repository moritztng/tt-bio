/* tt_profile.h -- by-stage wall-clock accounting for RELION's accelerated E-step.
 *
 * Why this file exists. RELION's ALTCPU path has no observable stage breakdown, for two independent
 * reasons, and both had to be found by reading the source:
 *
 *   1. `src/ml_optimiser.h` registers TIMING_ESP_DIFF1 / TIMING_ESP_DIFF2 / TIMING_ESP_WSUM, and
 *      `src/acc/acc_ml_optimiser_impl.h` does tic/toc them -- but only when
 *      `op.part_id == baseMLO->exp_my_first_part_id` (or `thread_id == 0`), which never fires on this
 *      code path, so `Timer::printTimes` skips them: it only prints tags with `counts[i] > 0`.
 *   2. The CTIC/CTOC macros that bracket every interesting region are defined empty at the top of
 *      `src/acc/cpu/cpu_benchmark_utils.h`, and the block that would give them a body is inside a
 *      `/* ... *\/` comment.
 *
 * So a -DTIMING ALTCPU build reports `expectationSomeParticles` as one opaque number (156.746 s of a
 * 161.2 s iteration) and nothing inside it. Every stage share in this lineage was therefore derived
 * from sampling counts rather than measured.
 *
 * What this does. Gives CTIC/CTOC a thread-safe body, so every region RELION already brackets is
 * accumulated. Regions are keyed on the `const char*` of the label, which is always a string literal
 * at every call site, so the pointer is stable and unique per site and there is no hashing per call.
 * Per-thread open regions live on a thread_local stack, so nesting works and a missing CTOC truncates
 * one thread's stack instead of corrupting the totals.
 *
 * The accumulated total is a sum over threads, so it exceeds the elapsed wall by roughly the thread
 * count. That is intended: the question is which stage owns the work, and a share of the CPU-time sum
 * answers it without needing the regions to be serialised. `oneParticle` is the denominator.
 *
 * Set TT_RELION_PROFILE=1 to print the table at exit. Accumulation is always on: it is a handful of
 * relaxed atomic adds per particle per region against a stage that costs tens of milliseconds.
 */
#ifndef TT_PROFILE_H_
#define TT_PROFILE_H_

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unistd.h>
#include <utility>
#include <vector>

namespace TTProf
{

struct Rec
{
	const char *label;
	std::atomic<unsigned long long> ns;
	std::atomic<unsigned long long> n;
	Rec(const char *l) : label(l), ns(0), n(0) {}
};

// A vector of pointers, never of objects: the vector may reallocate and a Rec holds atomics.
// Both statics are deliberately leaked. The report runs from an atexit handler, and atexit handlers
// interleave with static destructors in one LIFO list: a `static std::vector` here is constructed
// AFTER the handler is registered and therefore destroyed BEFORE it runs, so the report reads freed
// memory and the process exits 139 after printing a correct table. Measured that way once.
inline std::vector<Rec *> &table()
{
	static std::vector<Rec *> *t = new std::vector<Rec *>();
	return *t;
}

inline std::mutex &table_mutex()
{
	static std::mutex *m = new std::mutex();
	return *m;
}

inline void report();

inline Rec *rec_for(const char *label)
{
	// One registration per call site, then pointer-keyed lookup. The table has a few dozen entries,
	// so the linear scan is cheaper than a hash and needs no allocation on the hot path.
	std::lock_guard<std::mutex> g(table_mutex());
	static bool registered_atexit = false;
	if (!registered_atexit)
	{
		registered_atexit = true;
		std::atexit(&report);
	}
	for (size_t i = 0; i < table().size(); i++)
		if (table()[i]->label == label)
			return table()[i];
	table().push_back(new Rec(label));
	return table().back();
}

typedef std::chrono::steady_clock clk;

inline std::vector<std::pair<Rec *, clk::time_point> > &open_stack()
{
	static thread_local std::vector<std::pair<Rec *, clk::time_point> > s;
	return s;
}

// Pointer-keyed cache so only the first tic per site per thread takes the mutex.
inline Rec *cached_rec(const char *label)
{
	static thread_local std::vector<std::pair<const char *, Rec *> > cache;
	for (size_t i = 0; i < cache.size(); i++)
		if (cache[i].first == label)
			return cache[i].second;
	Rec *r = rec_for(label);
	cache.push_back(std::make_pair(label, r));
	return r;
}

inline void tic(const char *label)
{
	open_stack().push_back(std::make_pair(cached_rec(label), clk::now()));
}

inline void toc(const char *label)
{
	std::vector<std::pair<Rec *, clk::time_point> > &s = open_stack();
	const clk::time_point now = clk::now();
	// Scan from the top for the matching open region rather than assuming the top is it: RELION has
	// early returns between some CTIC/CTOC pairs, and a mismatched pop would then charge one stage's
	// time to another for the rest of the run.
	for (size_t k = s.size(); k-- > 0;)
	{
		if (s[k].first->label == label)
		{
			const unsigned long long ns =
			    (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(now - s[k].second).count();
			s[k].first->ns.fetch_add(ns, std::memory_order_relaxed);
			s[k].first->n.fetch_add(1, std::memory_order_relaxed);
			s.resize(k);
			return;
		}
	}
	// No matching tic: a CTOC without its CTIC. Drop it rather than guess.
}

inline void report()
{
	const char *on = std::getenv("TT_RELION_PROFILE");
	if (!on || on[0] == '0' || on[0] == '\0')
		return;
	std::lock_guard<std::mutex> g(table_mutex());
	unsigned long long total = 0;
	for (size_t i = 0; i < table().size(); i++)
		if (table()[i]->label && std::strcmp(table()[i]->label, "oneParticle") == 0)
			total = table()[i]->ns.load();
	std::printf("\nTTPROF rank_pid=%d stages=%zu denominator=oneParticle\n", (int)getpid(), table().size());
	std::printf("TTPROF %-42s %14s %12s %9s\n", "region", "cpu-seconds", "calls", "%onePart");
	for (size_t i = 0; i < table().size(); i++)
	{
		const unsigned long long ns = table()[i]->ns.load();
		const unsigned long long n = table()[i]->n.load();
		std::printf("TTPROF %-42s %14.4f %12llu %8.2f%%\n", table()[i]->label, ns / 1e9, n,
		            total ? 100.0 * (double)ns / (double)total : 0.0);
	}
	std::fflush(stdout);
}

} // namespace TTProf

#endif /* TT_PROFILE_H_ */
