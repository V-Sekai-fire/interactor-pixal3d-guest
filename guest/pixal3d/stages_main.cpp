#include <api.hpp>

#include "trellis2.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-rd.h"
#include "ggml.h"
#include "gguf.h"
#include "pump/pump.h"
#include "rd_compute.h"

static rdc::Device g_dev;
static bool g_attached = false;
static std::string g_log;

static void logf_(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf_(const char *fmt, ...) {
	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	g_log += buf;
	g_log += '\n';
	std::printf("pixal3d: %s\n", buf);
}

static void hook_wait_gpu(void *) {
	pump::wait_gpu();
}

static void hook_coop(void *) {
	pump::coop();
}

static bool hook_upload(void *, const std::string &path, uint64_t file_offset, uint64_t bytes, ::RID rid,
		uint64_t dst_offset) {
	return pump::upload(path, file_offset, bytes, rid, dst_offset);
}

static size_t read_into(const std::string &path, uint64_t offset, uint8_t *dst, uint64_t bytes) {
	const uint64_t chunk = uint64_t(16) << 20;
	uint64_t done = 0;
	while (done < bytes) {
		const uint64_t n = std::min(bytes - done, chunk);
		std::vector<uint8_t> v = pump::read(path, offset + done, n);
		std::memcpy(dst + done, v.data(), v.size());
		done += v.size();
		if (v.size() < n) {
			break;
		}
	}
	return size_t(done);
}

static bool hook_read(void *, const std::string &path, uint64_t file_offset, uint64_t bytes, void *dst) {
	return read_into(path, file_offset, static_cast<uint8_t *>(dst), bytes) == bytes;
}

static void on_ggml_abort(const char *message) {
	std::fflush(stdout);
	pump::fail(std::string("ggml abort: ") + message);
}

// pixal3d_stages.elf: stages of the Pixal3D pipeline on ggml-rd, each gated against a PyTorch
// reference dump (DINOv3 encoder tap by tap; sparse-structure flow forward). The model code is vendor/pixal3d-ggml's; only where bytes come
// from changes (trellis2_set_io), because the guest cannot open files.
//
// Host side:
//   pixal3d_attach(rd, total_mb)                     once; rd may be null (ggml-rd CPU kernels)
//   pixal3d_dino_start(model_gguf, ref_gguf, opts)   opts: "atol=2e-3 rtol=2e-3 control=perturb"
//   pixal3d_pump(data)                               every frame until DONE / ERROR
//   pixal3d_result_size(), pixal3d_result_chunk(off, n)   the report text
static std::vector<uint8_t> g_result;

struct Job {
	std::string model;
	std::string ref;
	double atol = 2e-3;
	double rtol = 2e-3;
	bool perturb = false;
	double rel_l2 = 2e-3;
	int stage = 0; // 0 dino, 1 ssflow, 2 ssdec
};
static Job g_job;

static size_t io_read(void *, const char *path, uint64_t offset, void *dst, size_t bytes) {
	return read_into(path, offset, static_cast<uint8_t *>(dst), bytes);
}

struct Stats {
	double max_abs = 0.0;
	double rel_l2 = 0.0;
	size_t bad = 0;
	size_t n = 0;
};

static Stats compare(const std::vector<float> &got, const std::vector<float> &ref, double atol, double rtol) {
	Stats s;
	s.n = ref.size();
	double num = 0.0;
	double den = 0.0;
	for (size_t i = 0; i < ref.size(); ++i) {
		const double d = std::fabs(double(got[i]) - double(ref[i]));
		num += d * d;
		den += double(ref[i]) * double(ref[i]);
		s.max_abs = std::max(s.max_abs, d);
		if (!(d <= atol + rtol * std::fabs(double(ref[i])))) {
			++s.bad;
		}
	}
	s.rel_l2 = den > 0.0 ? std::sqrt(num / den) : 0.0;
	return s;
}

// Returns true when every tap passes; counts taps the reference lacks as failures.
static bool run_gate(trellis2_dino_model *model, gguf_context *ref_gguf, ggml_context *ref_ctx, bool perturb,
		size_t &n_taps, size_t &n_fail, size_t &n_missing) {
	ggml_tensor *pix = ggml_get_tensor(ref_ctx, "pixel_values");
	if (pix == nullptr || pix->type != GGML_TYPE_F32) {
		logf_("reference has no f32 pixel_values");
		return false;
	}
	std::vector<float> pixels(size_t(ggml_nelements(pix)));
	std::memcpy(pixels.data(), pix->data, pixels.size() * sizeof(float));
	const int size = int(std::lround(std::sqrt(double(pixels.size()) / 3.0)));
	if (perturb) {
		for (size_t i = 0; i < pixels.size(); ++i) {
			pixels[i] += 0.05F * std::sin(float(i));
		}
	}
	trellis2_dino_cond cond;
	trellis2_dino_taps taps;
	std::string err;
	if (!trellis2_dino_encode(model, pixels.data(), size, cond, &taps, &err)) {
		logf_("encode failed: %s", err.c_str());
		return false;
	}
	n_taps = taps.names.size();
	for (size_t i = 0; i < taps.names.size(); ++i) {
		ggml_tensor *r = ggml_get_tensor(ref_ctx, taps.names[i].c_str());
		if (r == nullptr || r->type != GGML_TYPE_F32 || size_t(ggml_nelements(r)) != taps.data[i].size()) {
			logf_("[%-18s] no matching reference tensor", taps.names[i].c_str());
			++n_missing;
			continue;
		}
		std::vector<float> ref(size_t(ggml_nelements(r)));
		std::memcpy(ref.data(), r->data, ref.size() * sizeof(float));
		const Stats s = compare(taps.data[i], ref, g_job.atol, g_job.rtol);
		logf_("[%-18s] n=%zu max_abs=%.3e rel_l2=%.3e bad=%zu %s", taps.names[i].c_str(), s.n, s.max_abs, s.rel_l2,
				s.bad, s.bad == 0 ? "ok" : "FAIL");
		if (s.bad != 0) {
			++n_fail;
		}
	}
	(void)ref_gguf;
	return n_taps != 0 && n_fail == 0 && n_missing == 0;
}

static void dino_job(void *) {
	static trellis2_io io = { io_read, nullptr };
	trellis2_set_io(&io);
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		pump::fail("pixal3d: no ggml-rd device (call pixal3d_attach first)");
	}
	std::string err;
	trellis2_dino_model *model = trellis2_dino_load(g_job.model, true, &err);
	if (model == nullptr) {
		pump::fail("pixal3d: model load failed: " + err);
	}
	logf_("backend: %s", trellis2_dino_backend_name(model));
	ggml_context *ref_ctx = nullptr;
	gguf_init_params gp;
	gp.no_alloc = false;
	gp.ctx = &ref_ctx;
	std::string ref_path = g_job.ref;
	struct Rd {
		std::string *path;
	} rd{ &ref_path };
	gguf_context *ref_gguf = gguf_init_from_callback(
			[](void *user, void *out, uint64_t off, size_t n) -> size_t {
				return read_into(*static_cast<Rd *>(user)->path, off, static_cast<uint8_t *>(out), n);
			},
			&rd, size_t(16) << 20, UINT64_MAX >> 1, gp);
	if (ref_gguf == nullptr) {
		pump::fail("pixal3d: cannot read reference " + g_job.ref);
	}
	size_t n_taps = 0;
	size_t n_fail = 0;
	size_t n_missing = 0;
	const bool pass = run_gate(model, ref_gguf, ref_ctx, g_job.perturb, n_taps, n_fail, n_missing);
	std::string line;
	if (g_job.perturb) {
		line = std::string("CONTROL ") + (pass ? "MISSED" : "DETECTED");
	} else {
		line = std::string("RESULT: ") + (pass ? "PASS" : "FAIL");
	}
	line += " taps=" + std::to_string(n_taps) + " failed=" + std::to_string(n_fail) + " missing=" + std::to_string(n_missing);
	logf_("%s", line.c_str());
	g_result.assign(line.begin(), line.end());
	gguf_free(ref_gguf);
	ggml_free(ref_ctx);
	trellis2_dino_free(model);
}

static std::vector<uint8_t> read_all(const std::string &path) {
	std::vector<uint8_t> bundle;
	const uint64_t chunk = uint64_t(16) << 20;
	for (;;) {
		const size_t at = bundle.size();
		bundle.resize(at + chunk);
		const size_t got = read_into(path, at, bundle.data() + at, chunk);
		bundle.resize(at + got);
		if (got < chunk) {
			return bundle;
		}
	}
}

// Sparse-structure flow forward against the SSFREF01 bundle (x, t, cond, reference output).
static void ssflow_job(void *) {
	static trellis2_io io = { io_read, nullptr };
	trellis2_set_io(&io);
	if (ggml_backend_reg_dev_count(ggml_backend_rd_reg()) == 0) {
		pump::fail("pixal3d: no ggml-rd device (call pixal3d_attach first)");
	}
	const std::vector<uint8_t> bundle = read_all(g_job.ref);
	if (bundle.size() < 36 || std::memcmp(bundle.data(), "SSFREF01", 8) != 0) {
		pump::fail("pixal3d: bad SSFREF01 bundle " + g_job.ref);
	}
	int32_t dims[5];
	float t = 0.0F;
	std::memcpy(dims, bundle.data() + 8, sizeof dims);
	std::memcpy(&t, bundle.data() + 28, sizeof t);
	const size_t res = size_t(dims[0]);
	const size_t n = res * res * res;
	const size_t c_in = size_t(dims[1]);
	const size_t c_out = size_t(dims[2]);
	const size_t l_kv = size_t(dims[3]);
	const size_t c_ctx = size_t(dims[4]);
	const size_t floats = c_in * n + l_kv * c_ctx + c_out * n;
	if (bundle.size() != 32 + floats * sizeof(float)) {
		pump::fail("pixal3d: SSFREF01 bundle size does not match its header");
	}
	std::vector<float> all(floats);
	std::memcpy(all.data(), bundle.data() + 32, floats * sizeof(float));
	std::vector<float> x(all.begin(), all.begin() + long(c_in * n));
	std::vector<float> cond(all.begin() + long(c_in * n), all.begin() + long(c_in * n + l_kv * c_ctx));
	std::vector<float> ref(all.begin() + long(c_in * n + l_kv * c_ctx), all.end());
	if (g_job.perturb) {
		for (size_t i = 0; i < x.size(); ++i) {
			x[i] += 0.05F * std::sin(float(i));
		}
	}
	std::string err;
	trellis2_ss_flow_model *model = trellis2_ss_flow_load(g_job.model, true, &err);
	if (model == nullptr) {
		pump::fail("pixal3d: model load failed: " + err);
	}
	logf_("backend: %s", trellis2_ss_flow_backend_name(model));
	std::vector<float> out(c_out * n, 0.0F);
	if (!trellis2_ss_flow_forward(model, x.data(), t, cond.data(), int(l_kv), int(c_ctx), out.data(), &err)) {
		pump::fail("pixal3d: forward failed: " + err);
	}
	trellis2_ss_flow_free(model);
	const Stats s = compare(out, ref, 1e30, 0.0);
	const bool pass = s.rel_l2 <= g_job.rel_l2;
	logf_("ss_flow out n=%zu max_abs=%.3e rel_l2=%.3e tol=%.1e", s.n, s.max_abs, s.rel_l2, g_job.rel_l2);
	std::string line = g_job.perturb ? std::string("CONTROL ") + (pass ? "MISSED" : "DETECTED") : std::string("RESULT: ") + (pass ? "PASS" : "FAIL");
	line += " rel_l2=" + std::to_string(s.rel_l2);
	logf_("%s", line.c_str());
	g_result.assign(line.begin(), line.end());
}

// Sparse-structure decoder against the SSDEC001 bundle (latent, reference logits).
static void ssdec_job(void *) {
	static trellis2_io io = { io_read, nullptr };
	trellis2_set_io(&io);
	if (ggml_backend_reg_dev_count(ggml_backend_rd_reg()) == 0) {
		pump::fail("pixal3d: no ggml-rd device (call pixal3d_attach first)");
	}
	const std::vector<uint8_t> bundle = read_all(g_job.ref);
	if (bundle.size() < 24 || std::memcmp(bundle.data(), "SSDEC001", 8) != 0) {
		pump::fail("pixal3d: bad SSDEC001 bundle " + g_job.ref);
	}
	int32_t dims[4];
	std::memcpy(dims, bundle.data() + 8, sizeof dims);
	const size_t n_in = size_t(dims[0]) * size_t(dims[1]) * size_t(dims[1]) * size_t(dims[1]);
	const size_t n_out = size_t(dims[2]) * size_t(dims[3]) * size_t(dims[3]) * size_t(dims[3]);
	if (bundle.size() != 24 + (n_in + n_out) * sizeof(float)) {
		pump::fail("pixal3d: SSDEC001 bundle size does not match its header");
	}
	std::vector<float> latent(n_in);
	std::vector<float> ref(n_out);
	std::memcpy(latent.data(), bundle.data() + 24, n_in * sizeof(float));
	std::memcpy(ref.data(), bundle.data() + 24 + n_in * sizeof(float), n_out * sizeof(float));
	if (g_job.perturb) {
		for (size_t i = 0; i < latent.size(); ++i) {
			latent[i] += 0.05F * std::sin(float(i));
		}
	}
	std::string err;
	trellis2_ss_dec_model *model = trellis2_ss_dec_load(g_job.model, true, &err);
	if (model == nullptr) {
		pump::fail("pixal3d: model load failed: " + err);
	}
	logf_("backend: %s", trellis2_ss_dec_backend_name(model));
	std::vector<float> out(n_out, 0.0F);
	if (!trellis2_ss_dec_decode(model, latent.data(), out.data(), &err)) {
		pump::fail("pixal3d: decode failed: " + err);
	}
	trellis2_ss_dec_free(model);
	const Stats s = compare(out, ref, 1e30, 0.0);
	// A voxel whose reference logit lies within 1.0 of zero (occupancy 0.27 to 0.73) may flip; a flip
	// of a more confident voxel fails.
	size_t flipped = 0;
	size_t hard = 0;
	size_t occupied = 0;
	for (size_t i = 0; i < n_out; ++i) {
		const bool flip = (out[i] > 0.0F) != (ref[i] > 0.0F);
		flipped += flip ? 1 : 0;
		hard += flip && std::fabs(ref[i]) > 1.0F ? 1 : 0;
		occupied += ref[i] > 0.0F ? 1 : 0;
	}
	const bool pass = s.rel_l2 <= g_job.rel_l2 && hard == 0;
	logf_("ss_dec out n=%zu max_abs=%.3e rel_l2=%.3e tol=%.1e occupied=%zu flipped=%zu hard=%zu", s.n, s.max_abs, s.rel_l2,
			g_job.rel_l2, occupied, flipped, hard);
	std::string line = g_job.perturb ? std::string("CONTROL ") + (pass ? "MISSED" : "DETECTED") : std::string("RESULT: ") + (pass ? "PASS" : "FAIL");
	line += " rel_l2=" + std::to_string(s.rel_l2) + " flipped=" + std::to_string(flipped) + " hard=" + std::to_string(hard);
	logf_("%s", line.c_str());
	g_result.assign(line.begin(), line.end());
}

static std::vector<std::string> split_ws(const std::string &s) {
	std::vector<std::string> r;
	size_t i = 0;
	while (i < s.size()) {
		while (i < s.size() && s[i] == ' ') {
			++i;
		}
		size_t j = i;
		while (j < s.size() && s[j] != ' ') {
			++j;
		}
		if (j > i) {
			r.push_back(s.substr(i, j - i));
		}
		i = j;
	}
	return r;
}

static Variant pixal3d_attach(Object rd, int64_t total_mb) {
	if (rd.is_valid()) {
		g_dev.adopt(rd);
	}
	g_attached = g_dev.ok();
	static bool registered = false;
	if (!registered) {
		registered = true;
		ggml_backend_register(ggml_backend_rd_reg());
	}
	ggml_backend_rd_attach(g_attached ? &g_dev : nullptr, size_t(total_mb) << 20);
	ggml_rd_hooks h;
	h.wait_gpu = hook_wait_gpu;
	h.coop = hook_coop;
	h.upload = hook_upload;
	h.read = hook_read;
	ggml_backend_rd_set_hooks(h);
	ggml_set_abort_callback(on_ggml_abort);
	return Variant(String(g_attached ? "attached device=" + g_dev.device_name() : std::string("no RD device")));
}

static Variant pixal3d_start_stage(int stage, String model_gguf, String ref_gguf, String opts);
static Variant pixal3d_dino_start(String model_gguf, String ref_gguf, String opts) {
	return pixal3d_start_stage(0, model_gguf, ref_gguf, opts);
}
static Variant pixal3d_ssflow_start(String model_gguf, String ref_bin, String opts) {
	return pixal3d_start_stage(1, model_gguf, ref_bin, opts);
}
static Variant pixal3d_ssdec_start(String model_gguf, String ref_bin, String opts) {
	return pixal3d_start_stage(2, model_gguf, ref_bin, opts);
}
static Variant pixal3d_start_stage(int stage, String model_gguf, String ref_gguf, String opts) {
	for (const char *k : { "GGML_RD_FAULT", "GGML_RD_BARRIER_ALL", "GGML_RD_TIMESTAMPS" }) {
		unsetenv(k);
	}
	g_job = Job{};
	g_job.stage = stage;
	g_job.model = model_gguf.utf8();
	g_job.ref = ref_gguf.utf8();
	for (const std::string &kv : split_ws(opts.utf8())) {
		const size_t eq = kv.find('=');
		if (eq == std::string::npos) {
			continue;
		}
		const std::string k = kv.substr(0, eq);
		const std::string v = kv.substr(eq + 1);
		if (k == "atol") {
			g_job.atol = std::atof(v.c_str());
		} else if (k == "rtol") {
			g_job.rtol = std::atof(v.c_str());
		} else if (k == "rel_l2") {
			g_job.rel_l2 = std::atof(v.c_str());
		} else if (k == "control") {
			g_job.perturb = v == "perturb";
		} else {
			setenv(k.c_str(), v.c_str(), 1);
		}
	}
	g_log.clear();
	g_result.clear();
	if (!pump::start(g_job.stage == 2 ? &ssdec_job : g_job.stage == 1 ? &ssflow_job : &dino_job, nullptr, size_t(16) << 20)) {
		return Variant(String("FAIL a job is running"));
	}
	return Variant(String("STARTED dino"));
}

static Variant pixal3d_pump(PackedArray<uint8_t> in) {
	return pump::step(in);
}

static Variant pixal3d_result_size() {
	return Variant(int64_t(g_result.size()));
}

static Variant chunk_of(const std::vector<uint8_t> &data, int64_t offset, int64_t bytes) {
	const size_t off = size_t(std::max<int64_t>(offset, 0));
	if (off >= data.size()) {
		return Variant(PackedArray<uint8_t>(std::vector<uint8_t>()));
	}
	const size_t n = std::min<size_t>({ size_t(std::max<int64_t>(bytes, 0)), size_t(8) << 20, data.size() - off });
	return Variant(PackedArray<uint8_t>(data.data() + off, n));
}

static Variant pixal3d_result_chunk(int64_t offset, int64_t bytes) {
	return chunk_of(g_result, offset, bytes);
}

static Variant pixal3d_output() {
	return Variant(String(g_log));
}

static Variant pixal3d_stats() {
	std::string r = ggml_backend_rd_stats();
	r += " syncs=" + std::to_string(g_dev.syncs());
	r += " submits=" + std::to_string(g_dev.submits());
	r += " permanent_slots=" + std::to_string(g_dev.permanent_slots());
	return Variant(String(r));
}

int main() {
	ADD_API_FUNCTION(pixal3d_attach, "String", "Object rd, int total_mb", "Attach the host's RenderingDevice");
	ADD_API_FUNCTION(pixal3d_dino_start, "String", "String model_gguf, String ref_gguf, String opts", "Gate the DINOv3 encoder against a reference dump on the pump");
	ADD_API_FUNCTION(pixal3d_ssflow_start, "String", "String model_gguf, String ref_bin, String opts", "Gate the sparse-structure flow forward against an SSFREF01 bundle on the pump");
	ADD_API_FUNCTION(pixal3d_ssdec_start, "String", "String model_gguf, String ref_bin, String opts", "Gate the sparse-structure decoder against an SSDEC001 bundle on the pump");
	ADD_API_FUNCTION(pixal3d_pump, "Array", "PackedByteArray data", "Resume the job once: [header, text, rid]");
	ADD_API_FUNCTION(pixal3d_result_size, "int", "", "Bytes of the report line");
	ADD_API_FUNCTION(pixal3d_result_chunk, "PackedByteArray", "int offset, int bytes", "Report bytes (<= 8 MiB)");
	ADD_API_FUNCTION(pixal3d_output, "String", "", "What the job logged");
	ADD_API_FUNCTION(pixal3d_stats, "String", "", "ggml-rd and rd_compute counters");
	halt();
}
