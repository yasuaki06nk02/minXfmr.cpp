#ifdef __ARM_NEON

#include "cpu_backend.h"
#include "../../tensor/tensor.h"
#include <arm_neon.h>

// NEON-optimized implementations would go here. This file is intentionally
// small and guarded by __ARM_NEON so that it only compiles on ARM targets.

// Example placeholder: a NEON-accelerated dequant routine could be added.
#include <cstring>

// Dequantize one Q4_K block (256 outputs) using NEON intrinsics.
// This function is intentionally conservative: it mirrors the scalar
// implementation but vectorizes the inner unpack and convert operations.
void tensor_dequant_q4_k_block_neon(const uint8_t* blk, float* dst256) {
	uint16_t hd = 0, hm = 0;
	std::memcpy(&hd, blk + 0, sizeof(hd));
	std::memcpy(&hm, blk + 2, sizeof(hm));
	const float d = tensor_fp16_to_fp32(hd);
	const float dmin = tensor_fp16_to_fp32(hm);

	const uint8_t* scales = blk + 4;
	const uint8_t* q = blk + 16;

	auto get_scale_min_k4_local = [](int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
		if (j < 4) {
			d = q[j] & 63;
			m = q[j + 4] & 63;
		} else {
			d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
			m = (q[j + 4] >> 4) | ((q[j - 4] >> 6) << 4);
		}
	};

	int is = 0;
	for (int j = 0; j < (int)TENSOR_Q4_K_QK_K; j += 64) {
		uint8_t sc = 0, m = 0;
		get_scale_min_k4_local(is + 0, scales, sc, m);
		const float d1 = d * sc;
		const float m1 = dmin * m;

		get_scale_min_k4_local(is + 1, scales, sc, m);
		const float d2 = d * sc;
		const float m2 = dmin * m;

		// process 32 bytes in chunks of 8
		for (int l = 0; l < 32; l += 8) {
			uint8x8_t bytes = vld1_u8(q + l);
			uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0F));
			uint8x8_t hi = vshrq_n_u8(bytes, 4);

			uint16x8_t lo16 = vmovl_u8(lo);
			uint16x8_t hi16 = vmovl_u8(hi);

			// lower half (elements 0..3)
			uint16x4_t lo16_low = vget_low_u16(lo16);
			uint16x4_t lo16_high = vget_high_u16(lo16);
			uint16x4_t hi16_low = vget_low_u16(hi16);
			uint16x4_t hi16_high = vget_high_u16(hi16);

			uint32x4_t lo32_0 = vmovl_u16(lo16_low);
			uint32x4_t lo32_1 = vmovl_u16(lo16_high);
			uint32x4_t hi32_0 = vmovl_u16(hi16_low);
			uint32x4_t hi32_1 = vmovl_u16(hi16_high);

			float32x4_t lof0 = vcvtq_f32_u32(lo32_0);
			float32x4_t lof1 = vcvtq_f32_u32(lo32_1);
			float32x4_t hif0 = vcvtq_f32_u32(hi32_0);
			float32x4_t hif1 = vcvtq_f32_u32(hi32_1);

			float32x4_t res_lo0 = vsubq_f32(vmulq_n_f32(lof0, d1), vdupq_n_f32(m1));
			float32x4_t res_lo1 = vsubq_f32(vmulq_n_f32(lof1, d1), vdupq_n_f32(m1));
			float32x4_t res_hi0 = vsubq_f32(vmulq_n_f32(hif0, d2), vdupq_n_f32(m2));
			float32x4_t res_hi1 = vsubq_f32(vmulq_n_f32(hif1, d2), vdupq_n_f32(m2));

			vst1q_f32(dst256 + j + l, res_lo0);
			vst1q_f32(dst256 + j + l + 4, res_lo1);
			vst1q_f32(dst256 + j + 32 + l, res_hi0);
			vst1q_f32(dst256 + j + 32 + l + 4, res_hi1);
		}

		q += 32;
		is += 2;
	}
}

#endif // __ARM_NEON
