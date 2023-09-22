#ifndef SHITTIM_RL_MATHOPS_H_
#define SHITTIM_RL_MATHOPS_H_

#include<cstddef>
#include<cmath>

namespace shittim {
namespace rl {

template<class Logits, class Probs>
Probs& softmax(const Logits& logits, Probs& probs, const double beta = 1.0)
{
  assert(probs.size() == logits.size());
  double max = *std::max_element(logits.begin(), logits.end());
  double sum = 0.;
  for(std::size_t i = 0; i < logits.size(); ++i)
  {
    sum += (probs[i] = std::exp(beta * (logits[i] - max)));
  }
  for(std::size_t i = 0; i < logits.size(); ++i)
  {
    probs[i] /= sum;
  }
  return probs;
}
template<class Probs, class Grads>
void dsoftmax(const std::size_t label, const Probs& probs, Grads& grads, const double beta = 1.0)
{
  assert(grads.size() == probs.size());
  assert(label < probs.size());

  for(std::size_t i = 0; i < grads.size(); ++i)
  {
    grads[i] = - beta * ((i == label ? 1. : 0.) - probs[i]);
  }
}
template<class Probs, class Grads>
void dsoftmax(const Probs& tprobs, const Probs& probs, Grads& grads, const double beta = 1.0)
{
  assert(grads.size() == tprobs.size());
  assert(grads.size() == probs.size());

  for(std::size_t i = 0; i < grads.size(); ++i)
  {
    grads[i] = - beta * (tprobs[i] - probs[i]);
  }
}

template<class Probs>
double crossentropy_loss(const std::size_t label, const Probs& probs)
{
  assert(label < probs.size());
  return -std::log(probs[label]);
}

template<class Probs>
double crossentropy_loss(const Probs& target_probs, const Probs& probs)
{
  assert(target_probs.size() == probs.size());
  double loss = 0;
  for(std::size_t i = 0; i < target_probs.size(); ++i)
  {
    loss += -target_probs[i] * std::log(probs[i]);
  }
  return loss;
}

template<class Container>
std::size_t argmax(const Container& container)
{
  return std::distance(container.begin(), std::max_element(container.begin(), container.end()));
}

} // namespace rl
} // namespace shittim

#endif // SHITTIM_RL_MATHOPS_H_