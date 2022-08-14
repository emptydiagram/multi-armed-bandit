using Random

using Pkg
Pkg.add("Plots")
Pkg.add("Distributions")

using Distributions
using Plots

Action = UInt
Reward = Float64

SEED = 628
bp_rng = MersenneTwister(SEED)


# f::Function{T, (V, T)}
# domain::AbstractVector{T}
function argmax_rand_tiebreak(f::Function, domain::AbstractVector)
  max1((f1, i1), (f2, i2)) = isless(f1, f2) ? (f2, i2) : (f1, i1)
  mapped = map(x -> (f(x), x), domain)
  # println("argmax_rand_tiebreak, mapped = ", mapped)
  max_pairs = nothing
  for (v, x) in mapped
    if max_pairs === nothing || v > max_pairs[1][1]
      max_pairs = [(v, x)]
    elseif v == max_pairs[1][1]
      push!(max_pairs, (v, x))
    end
  end
  rand_mp_idx = rand(1:length(max_pairs))
  # if length(max_pairs) > 1
  #   println(" :: number of max_pairs = ", length(max_pairs))
  #   println(" :: chose random idx = ", rand_mp_idx)
  # end
  return max_pairs[rand_mp_idx][2]
end


# K-armed bandit problem, player that uses sample-average to estimate action-value
mutable struct EpsGreedySampleAverageBanditPlayer
  total_reward_action::Dict{Action, Float64}
  total_num_sels_action::Dict{Action, UInt}
  total_reward::Float64
  total_num_sels::UInt
  K::UInt
  eps::Float64

  function EpsGreedySampleAverageBanditPlayer(eps::Float64, K::UInt)
    eps = eps
    K = K
    total_reward_action = Dict{Action, Float64}()
    total_num_sels_action = Dict{Action, UInt}()
    total_reward = 0
    total_num_sels = UInt(0)
    return new(total_reward_action, total_num_sels_action, total_reward, total_num_sels, K, eps)
  end
end

function reward_update(player::EpsGreedySampleAverageBanditPlayer, action::Action, reward::Reward)
  # println(":: updating for (reward, action) = ($reward, $action)")
  if !haskey(player.total_reward_action, action)
    player.total_reward_action[action] = 0.
  end
  player.total_reward_action[action] += reward

  if !haskey(player.total_num_sels_action, action)
    player.total_num_sels_action[action] = 0
  end
  player.total_num_sels_action[action] += 1

  player.total_reward += reward
  player.total_num_sels += 1
end

function action_value(player::EpsGreedySampleAverageBanditPlayer, action::Action)
  if !haskey(player.total_num_sels_action, action) || player.total_num_sels_action[action] == 0.
    return 0.
  end
  return player.total_reward_action[action] / player.total_num_sels_action[action]
end

function choose_action(player::EpsGreedySampleAverageBanditPlayer)
  if rand() <= player.eps
    return rand(1:player.K)
  end
    # action = argmax(act -> action_value(player, act), 1:player.K)
    action = argmax_rand_tiebreak(act -> action_value(player, act), 1:player.K)
    # println("choose_action, taking greedy action = ", action)
    return action
end

function generate_random_bandit_problem(K)
    random_q_values = randn(bp_rng, Float64, Int(K))
    return random_q_values
end

function run_bandit_problems(num_runs::Int, num_run_steps::Int, eps::Float64, K::UInt)
  rewards_mean = zeros(Float64, num_run_steps)
  rewards_stddev = zeros(Float64, num_run_steps)

  for i in 1:num_runs
    q_values= generate_random_bandit_problem(K)
    player = EpsGreedySampleAverageBanditPlayer(eps, K)
    Sn = 0.
    for j in 1:num_run_steps
      action = choose_action(player)
      # reward = q_values[action] + randn(bp_rng, Float64)
      reward_gaussian = Normal(q_values[action], 1.)
      reward = rand(reward_gaussian)
      reward_update(player, action, reward)

      # mu_n = mu_{n-1} + (x_n - mu_{n-1}) / n
      # S_n = S_{n-1} + (x_n - mu_{n-1}) * (x_n - mu_n)
      # sigma_n = sqrt(S_n / n)
      avg_reward_curr = player.total_reward / player.total_num_sels
      if j == 1
        rewards_mean[j] = avg_reward_curr
        Sn += 0.
      else
        rewards_mean[j] = rewards_mean[j-1] + (avg_reward_curr - rewards_mean[j-1]) / Float64(j)
        Sn += (avg_reward_curr - rewards_mean[j-1]) * (avg_reward_curr - rewards_mean[j])
      end
      stddev = sqrt(Sn / j)
      rewards_stddev[j] = stddev

    end
    # println("run $i results")
    # println("estimated Q_t(a) | actual q_*(a)")
    # for action in 1:K
    #   println("  $action: $(action_value(player, action)) | $(q_values[action])")
    # end

    # for i in 1:length(avg_rewards)
    #   println("  avg rew $i: $(avg_rewards[i])")
    # end
    # println("  avg rew end: $(avg_rewards_curr[end])")
  end
  # for i in eachindex(avg_rewards)
  #   println("  avg rew $i: $(avg_rewards[i])")
  # end
  # println("  avg rew end: $(avg_rewards[end])")
  return (rewards_mean, rewards_stddev)
end
