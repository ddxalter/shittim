#ifndef SHITTIM_RL_TRAINER_H_
#define SHITTIM_RL_TRAINER_H_

#include<iostream>

#include"arona/progress.h"

#include"shittim/rl/traits.h"

namespace shittim {
namespace rl {

template<class E, class A>
class Trainer
{
  static_assert(traits::has_observation<E>::value, "aronarl::Trainer: Environment must have a type 'Observation'.");
  static_assert(traits::has_action<E>::value, "aronarl::Trainer: Environment must have a type 'Action'.");
  static_assert(traits::has_reward<E>::value, "aronarl::Trainer: Environment must have a type 'Reward'.");

  using Observation = typename E::Observation;
  using Action = typename E::Action;
  using Reward = typename E::Reward;
  using Step = typename E::Step;
  using Count = std::size_t;
  using Loss = double;
  using Rewards = std::vector<Reward>;
  using Counts = std::vector<Count>;
  using Losses = std::vector<Loss>;
  using EpisodeResult = std::tuple<Reward, Count, Loss>;
  using SessionResult = std::tuple<Rewards, Counts, Losses>;
public:
  E& env;
  A& agent;
  std::ostream* ostream_ = nullptr;

  Trainer(E& env, A& agent) : env(env), agent(agent) {}

  void enable_monitoring_output(std::ostream& ostream) { ostream_ = &ostream; }
  void disable_monitoring_output() { ostream_ = nullptr; }

  EpisodeResult run_episode(int max_steps)
  {
    Reward episode_rewards = 0;
    Count step_count = 0;
    Observation observation = env.reset();
    agent.reset(observation);
    for(step_count = 0; step_count < max_steps; ++step_count)
    {
      Action action = agent.sample_action(observation);
      Step step_tuple = env.step(action);
      agent.train_step({observation, action, step_tuple.observation(), step_tuple.reward(), step_tuple.discount()});
      episode_rewards += step_tuple.reward();
      if(step_tuple.is_done()) { ++step_count; break; };
      observation = std::move(step_tuple.observation());
    }
    agent.train_episode();
    return {episode_rewards, step_count, agent.episode_loss()};
  }
  SessionResult run_session(const Count max_episodes, const Count max_steps)
  {
    if(ostream_ != nullptr)
    {
      *ostream_ << "aronarl::Trainer::run: [start session] max_episodes=" << max_episodes << ", max_steps=" << max_steps << std::endl;
    }
    Rewards result_rewards;
    Counts result_steps;
    Losses result_losses;
    for(int episode = 0; episode < max_episodes; ++episode)
    {
      if(ostream_ != nullptr)
      {
        *ostream_ << arona::Progress(episode, max_episodes);
      }
      auto result = run_episode(max_steps);
      result_rewards.emplace_back(std::get<0>(result));
      result_steps.emplace_back(std::get<1>(result));
      result_losses.emplace_back(std::get<2>(result));
    }
    return {result_rewards, result_steps, result_losses};
  }
};

} // namespace rl
} // namespace shittim

#endif // SHITTIM_RL_TRAINER_H_