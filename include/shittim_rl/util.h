#ifndef SHITTIM_RL_UTIL_H_
#define SHITTIM_RL_UTIL_H_

// utility classes to use in environments and agents

#include<type_traits>

#include"shittim/rl/traits.h"

namespace shittim {
namespace rl {

template<class BoundaryType>
class BoundaryTraits
{
  static_assert(traits::has_point_type<BoundaryType>::value, "aronarl::util::BoundaryTraits: boundary type must have a member type 'point_type'.");
  using point_type = typename BoundaryType::point_type;
public:
  template<class P = point_type>
  static auto contains(const BoundaryType& bounds, const P& p)
    -> traits::enable_if_t<!traits::is_container<P>::value, bool>
  {
    return bounds.lower_bounds() <= p && p <= bounds.upper_bounds();
  }

  template<class P = point_type>
  static auto contains(const BoundaryType& bounds, const P& p)
    -> traits::enable_if_t<traits::is_container<P>::value, bool>
  {
    bool is_inside = true;
    for(std::size_t i = 0; i < p.size(); ++i)
    {
      is_inside &= bounds.lower_bounds()[i] <= p[i] && p[i] <= bounds.upper_bounds()[i];
    }
    return is_inside;
  }

  template<class P = point_type>
  static auto contains(const BoundaryType& bounds, const point_type& p, const std::size_t index)
    -> traits::enable_if_t<traits::is_container<P>::value, bool>
  {
    return bounds.lower_bounds()[index] <= p[index] && p[index] <= bounds.upper_bounds()[index];
  }
};

template<class T>
class Discretizer
{
  Discretizer(const std::vector<T>& lb, const std::vector<T>& ub, const std::vector<std::size_t>& dims)
    : lb_(lb), ub_(ub), dims_(dims), ndim_(dims_.size())
    , size_(std::accumulate(dims_.begin(), dims_.end(), 1ull, std::multiplies<std::size_t>())) {}

  const std::vector<T>& lb() const noexcept { return lb_; }
  const std::vector<T>& ub() const noexcept { return ub_; }
  const std::vector<std::size_t>& dims() const noexcept { return dims_; }
  std::size_t ndim() const noexcept {return ndim_; }
  std::size_t size() const noexcept { return size_; }
  
  template<class Point>
  std::size_t operator()(const Point& point) const noexcept { return to_index(point); }

  template<class Index>
  std::vector<T> operator[](const std::size_t index) const noexcept { return index_to_point(index); }

  template<class P>
  std::size_t to_index(const P& p) const noexcept
  {
    std::size_t index = 0;
    for(std::size_t i = 0; i < ndim_; ++i)
    {
      index = dims_[i] > 1 ? index * dims_[i] + discretize(p[i], lb_[i], ub_[i], dims_[i]) : index;
    }
    return index;
  }
  template<class P, class I>
  inline const I& to_indices(const P& p, I& indices) const noexcept
  {
    for(std::size_t i = 0; i < ndim_; ++i)
    {
      indices[i] = dims_[i] > 1 ? discretize(p[i], lb_[i], ub_[i], dims_[i]) : 0;
    }
    return indices;
  }
  template<class P>
  std::vector<std::size_t> to_indices(const P& p) const noexcept
  {
    std::vector<std::size_t> indices(ndim_);
    return to_indices(p, indices);
  }
  std::size_t indices_to_index(const std::vector<std::size_t>& indices) const noexcept
  {
    std::size_t index = 0;
    for(std::size_t i = 0; i < ndim_; ++i)
    {
        index = index * dims_[i] + indices[i];
    }
    return index;
  }
  inline std::vector<std::size_t> index_to_indices(std::size_t index) const noexcept
  {
    std::vector<std::size_t> indices(ndim_);
    for(std::size_t i = 1; i <= ndim_; ++i)
    {
      indices[ndim_ - i] = index % dims_[ndim_ - i];
      index /= dims_[ndim_ - i];
    }
    return indices;
  }
  std::vector<T> index_to_point(std::size_t index) const noexcept
  {
    std::vector<std::size_t> indices = index_to_indices(index);
    std::vector<T> point(ndim_);
    for(std::size_t i = 0; i < ndim_; ++i)
    {
        point[i] = lb_[i] + ((ub_[i] - lb_[i]) / dims_[i]) * indices[i];
    }
    return point;
  }
  static std::size_t discretize(T x, T l, T u, std::size_t n) noexcept
  {
    int index = (n * (x - l)) / (u - l);
    return index < 0 ? 0ul : (index >= n ? n - 1 : static_cast<std::size_t>(index));
  }
private:
  const std::vector<T> lb_;
  const std::vector<T> ub_;
  const std::vector<std::size_t> dims_;
  const std::size_t ndim_;
  const std::size_t size_;
};

} // namespace rl
} // namespace shittim

#endif // SHITTIM_RL_UTIL_H_