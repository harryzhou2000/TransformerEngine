/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_dgelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream) {
  NVTE_API_CALL(nvte_dgelu);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, dgelu<fp32, fp32>>(grad, input, output, stream);
}

void nvte_dgeglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_dgeglu);
  using namespace transformer_engine;
  Empty e = {};
  dgated_act_fn<fp32, Empty, gelu<fp32, fp32>, dgelu<fp32, fp32>>(grad, input, output, e,
                                                                  stream);
}
