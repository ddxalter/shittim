#ifndef SHITTIM_RL_CORE_H_
#define SHITTIM_RL_CORE_H_

#include<deque>
#include<tuple>

namespace shittim {
namespace rl {

enum class StepType { first, mid, last };

template<class O, class R = double, class D = double>
struct Step
{
public:
  using Observation = O;
  using Reward = R;
  using Discount = D;
  Step(const Observation& observation, const Reward& reward, const Discount& discount, const StepType& status)
    : observation_(observation), reward_(reward), discount_(discount), status_(status) {}
  const Observation& observation() const { return observation_; };
  const Reward& reward() const { return reward_; };
  const Discount& discount() const { return discount_; };
  bool is_done() const { return status_ == StepType::last && !(discount_ > 0); }
  bool is_truncated() const { return status_ == StepType::mid && (discount_ > 0); }
  bool is_terminated() const { return status_ == StepType::last; }
  static Step make_start(const Observation& observation)
  {
    return Step(observation, Reward{0}, Discount{0}, StepType::first);
  }
  static Step make_transition(const Observation& observation, const Reward& reward)
  {
    return Step(observation, reward, Discount{1}, StepType::mid);
  }
  static Step make_termination(const Observation& observation, const Reward& reward)
  {
    return Step(observation, reward, Discount{0}, StepType::last);
  }
  static Step make_truncation(const Observation& observation, const Reward& reward)
  {
    return Step(observation, reward, Discount{1}, StepType::last);
  }
private:
  Observation observation_;
  Reward reward_;
  Discount discount_;
  StepType status_;
};

template<class ObsSpec, class ActSpec, class R = double, class D = double>
class ReinforcementLearning
{
public:
  using ObservationSpec = ObsSpec;
  using ActionSpec = ActSpec;
  using Observation = typename ObsSpec::point_type;
  using Action = typename ActSpec::point_type;
  using Reward = R;
  using Discount = D;
  using Step = Step<Observation, Reward, Discount>;
  using TrainTuple = std::tuple<const Observation&, const Action&, const Observation&, const Reward&, const Discount&>;

  class TypeDefinitions
  {
  public:
    using ObservationSpec = typename ReinforcementLearning::ObservationSpec;
    using ActionSpec = typename ReinforcementLearning::ActionSpec;
    using Observation = typename ReinforcementLearning::ObservationSpec::point_type;
    using Action = typename ReinforcementLearning::ActionSpec::point_type;
    using Reward = typename ReinforcementLearning::Reward;
    using Discount = typename ReinforcementLearning::Discount;
    using Step = typename ReinforcementLearning::Step;
    using TrainTuple = typename ReinforcementLearning::TrainTuple;
  };

  class EnvironmentBase : public TypeDefinitions
  {
  public:
    EnvironmentBase() = default;
    virtual ~EnvironmentBase() = default;
    // required
    virtual Observation reset() = 0;
    virtual Step step(const Action&) = 0;
    virtual ObservationSpec observation_spec() const = 0;
    virtual ActionSpec action_spec() const = 0;
  };

  class AgentBase : public TypeDefinitions
  {
  public:
    AgentBase() = default;
    virtual ~AgentBase() = default;
    // required
    virtual Action sample_action(const Observation&) = 0;
    // optional
    virtual void train_step(const TrainTuple&) {};
    virtual void init(const ObservationSpec&, const ActionSpec&) {}
    virtual void reset(const Observation&) {}
    virtual void train_episode() {}
    virtual double episode_loss() { return 0; }
  };
};

} // namespace rl
} // namespace shittim

#endif // SHITTIM_RL_CORE_H_
