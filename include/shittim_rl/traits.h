#ifndef SHITTIM_RL_TRAITS_H_
#define SHITTIM_RL_TRAITS_H_

#include<type_traits>

namespace shittim {
namespace rl {
namespace traits {

template<class ... Args>
using void_t = void;

template<bool C, class U = void>
using enable_if_t = typename std::enable_if<C, U>::type;

template<class Default, class AlwaysVoid, template <class ...> class Op, class ... Args>
struct detected_impl
{
  using value_t = std::false_type;
  using type = Default;
};

template<class Default, template <class ...> class Op, class ... Args>
struct detected_impl <Default, void_t<Op<Args ...>>, Op, Args ...>
{
  using value_t = std::true_type;
  using type = Op<Args ...>;
};

template<template <class ...> class Op, class ... Args>
using is_detected = typename detected_impl<void, void, Op, Args ...>::value_t;

template<template <class ...> class Op, class ... Args>
using is_detected_t = typename detected_impl<void, void, Op, Args ...>::type;

template<class T> using container_detector = void_t<typename T::value_type,
                                                    typename T::size_type,
                                                    decltype(std::declval<T>().begin()),
                                                    decltype(std::declval<T>().end()),
                                                    decltype(std::declval<T>().size())>;

template<class T> using is_container = is_detected<container_detector, T>;

template<class T> using observation_detector = typename T::Observation;
template<class T> using action_detector = typename T::Action;
template<class T> using reward_detector = typename T::Reward;
template<class T> using discount_detector = typename T::Discount;

template<class T> using has_observation = is_detected<observation_detector, T>;
template<class T> using has_action = is_detected<action_detector, T>;
template<class T> using has_reward = is_detected<reward_detector, T>;
template<class T> using has_discount = is_detected<discount_detector, T>;

template<class T> using point_type_detector = typename T::point_type;
template<class T> using has_point_type = is_detected<point_type_detector, T>;

template<class T> using lower_bounds_detecter = decltype(std::declval<T>().lower_bounds());
template<class T> using upper_bounds_detecter = decltype(std::declval<T>().upper_bounds());

template<class T> using has_lower_bounds = is_detected<lower_bounds_detecter, T>;
template<class T> using has_upper_bounds = is_detected<upper_bounds_detecter, T>;

} // namespace traits
} // namespace rl
} // namespace shittim

#endif // SHITTIM_RL_TRAITS_H_