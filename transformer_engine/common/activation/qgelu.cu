/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/math.h"
#include "./activation_template.h"

void nvte_qgelu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_qgelu);
  using namespace transformer_engine;
  act_fn<fp32, Empty, qgelu<fp32, fp32>>(input, output, stream);
}

void nvte_qgeglu(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_qgeglu);
  using namespace transformer_engine;
  Empty e = {};
  gated_act_fn<fp32, Empty, qgelu<fp32, fp32>>(input, output, e, stream);
}
