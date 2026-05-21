/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_dqgelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_dqgelu);
  using namespace transformer_engine;
  dact_fn<fp32, Empty, dqgelu<fp32, fp32>>(grad, input, output, stream);
}

void nvte_dqgeglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_dqgeglu);
  using namespace transformer_engine;
  Empty e = {};
  dgated_act_fn<fp32, Empty, qgelu<fp32, fp32>, dqgelu<fp32, fp32>>(grad, input, output, e,
                                                                   stream);
}
