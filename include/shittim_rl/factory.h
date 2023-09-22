#ifndef SHITTIM_RL_FACTORY_H_
#define SHITTIM_RL_FACTORY_H_

#include"arona/factory.h"

namespace shittim {
namespace rl {

template<class Base, class ... Args>
using Factory = arona::Factory<Base, Args ...>;

} // namespace rl
} // namespace shittim

#endif // SHITTIM_RL_FACTORY_H_