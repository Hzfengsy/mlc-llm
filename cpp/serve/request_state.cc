/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/request_state.cc
 */

#include "request_state.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** RequestModelState ******************/

TVM_REGISTER_OBJECT_TYPE(RequestModelStateNode);

RequestModelState::RequestModelState(int model_id, Array<Data> inputs) {
  ObjectPtr<RequestModelStateNode> n = make_object<RequestModelStateNode>();
  n->model_id = model_id;
  n->request_id = -1;
  n->inputs = std::move(inputs);
  data_ = std::move(n);
}

int RequestModelStateNode::GetInputLength() const {
  int total_length = 0;
  for (Data input : inputs) {
    total_length += input->GetLength();
  }
  return total_length;
}

TVM_REGISTER_OBJECT_TYPE(RequestStateNode);

RequestState::RequestState(Request request, int num_models) {
  ObjectPtr<RequestStateNode> n = make_object<RequestStateNode>();
  Array<RequestModelState> mstates;
  mstates.reserve(num_models);
  for (int i = 0; i < num_models; ++i) {
    mstates.push_back(RequestModelState(i, request->inputs));
  }
  n->request = std::move(request);
  n->mstates = std::move(mstates);
  n->tadd = std::chrono::high_resolution_clock::now();
  data_ = std::move(n);
}

bool RequestStateNode::GenerationFinished(int max_single_sequence_length) const {
  // - Case 0. There is remaining draft output ==> Unfinished
  //   All draft outputs are supposed to be processed before finish.
  for (RequestModelState mstate : mstates) {
    if (!mstate->draft_output_tokens.empty()) {
      return false;
    }
  }

  // - Decode committed tokens.
  const std::vector<int32_t>& committed_tokens = mstates[0]->committed_tokens;

  // Case 1. Any of the stop strings appears in output ==> Finished
  // Todo: handle stop_str by tokenizing. So that we don't detokenize during check

  // Case 2. Any of the stop tokens appears in the committed tokens ===> Finished
  if (std::any_of(
          request->generation_cfg->stop_tokens.begin(), request->generation_cfg->stop_tokens.end(),
          [&committed_tokens](int32_t token) { return token == committed_tokens.back(); })) {
    return true;
  }
  // Case 3. Generation reaches the specified max generation length ==> Finished
  if (static_cast<int>(committed_tokens.size()) >= request->generation_cfg->max_new_tokens) {
    return true;
  }
  // Case 4. Total length of the request reaches the maximum single sequence length ==> Finished
  if (request->input_total_length + static_cast<int>(committed_tokens.size()) >=
      max_single_sequence_length) {
    return true;
  }
  return false;
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc