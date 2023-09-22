#ifndef SHITTIM_RL_SPEC_H_
#define SHITTIM_RL_SPEC_H_

#include<vector>

#include"shittim/rl/random.h"

namespace shittim {
namespace rl {

// minimum equirements of 'Spec' kinds
// - member type: `point_type`
// - member function: `point_type sample() const`

template<class T>
class BoundedSpec
{
public:
  using point_type = T;
  BoundedSpec() = default;
  virtual ~BoundedSpec() = default;
  BoundedSpec(const point_type lb, const point_type ub) : lb_(lb), ub_(ub) {}
  point_type& lower_bounds() { return lb_; }
  point_type& upper_bounds() { return ub_; }
  const point_type& lower_bounds() const { return lb_; }
  const point_type& upper_bounds() const { return ub_; }
  virtual point_type sample() const = 0;
  point_type lb_;
  point_type ub_;
};

class ContinuousSpec : public BoundedSpec<std::vector<float>>
{
public:
  ContinuousSpec(const point_type lb, const point_type ub) : BoundedSpec(lb, ub) {}
  point_type sample() const
  {
    point_type p;
    for(std::size_t i = 0; i < lb_.size(); ++i)
    {
      p.push_back(Random::uniform_real(lb_[i], ub_[i]));
    }
    return p;
  }
};

class ContinuousScalarSpec : public BoundedSpec<float>
{
public:
  ContinuousScalarSpec(const point_type lb, const point_type ub) : BoundedSpec(lb, ub) {}
  point_type sample() const
  {
    return Random::uniform_real(lb_, ub_);
  }
};

class DiscreteSpec : public BoundedSpec<std::vector<int>>
{
public:
  DiscreteSpec() = default;
  DiscreteSpec(const point_type lb, const point_type ub) : BoundedSpec(lb, ub) {}
  point_type sample() const
  {
    point_type p;
    for(std::size_t i = 0; i < lb_.size(); ++i)
    {
      p.push_back(Random::uniform_int(lb_[i], ub_[i]));
    }
    return p;
  }
};

class DiscreteScalarSpec : public BoundedSpec<int>
{
public:
  DiscreteScalarSpec(const point_type lb, const point_type ub) : BoundedSpec(lb, ub) {}
  point_type sample() const
  {
    return Random::uniform_int(lb_, ub_);
  }
};

} // namespace rl
} // namespace shittim

#endif // SHITTIM_RL_SPEC_H_