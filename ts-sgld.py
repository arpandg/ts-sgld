

#===============================================================================================================

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import itertools
import jax
import jax.numpy as jnp
from functools import partial
from scipy.special import entr
import json
import time

# N/n * ss = 0.001

@jax.jit
def prior_term(lambda_1, lambda_2, l, r):
  p1 = jnp.prod(jnp.exp(lambda_1 * l) * lambda_1)
  p2 = jnp.prod(jnp.exp(lambda_2 * r) * lambda_2)
  return jnp.log(p1 * p2)

@jax.jit
def prob_types(l):
  e_x = jnp.exp(l)
  if l.ndim == 1:
    return e_x / e_x.sum(axis=0) + eps
  return e_x / (e_x.sum(axis=1).reshape(-1, 1) + eps)

@jax.jit
def prob_slots(r, result):
  return (jnp.exp(r) * result + (1 - result)) / (1 + jnp.exp(r))

@jax.jit
def cond_term(l, r, result):
  prob_type = prob_types(l)
  prob_slot = prob_slots(r, result)
  P = jnp.log(jnp.matmul(prob_type, prob_slot))
  return P

def calculate_second_derivative(l, r, result, verbose=False):
  # Check notes for formula
  prob_type = prob_types(l)
  prob_slot = prob_slots(r, result)
  P = np.matmul(prob_type, prob_slot)
  # print(P)

  # del_w = np.zeros(w.shape)

  SM = prob_type.reshape((-1,1))
  jac = jnp.diagflat(prob_type) - jnp.dot(SM, SM.T)
  del_l = (1/(P+eps)) * jnp.matmul(jac, prob_slot)

  # del_w = (1/P) * del_w
  del_r = (1/(P+eps)) * prob_type * ((np.exp(r) * (2 * result - 1)) / ((1 + np.exp(r))**2 + eps))
  if verbose:
    print(prob_type, prob_slot, P)
    print(del_l, del_r)

  return del_l, del_r

def calculate_first_derivative(lambda_1, lambda_2, l, r):

  # Check notes for this formula

  del_w = lambda_1
  # del_r = np.divide((beta * (1 - 2 * np.power(r, beta))) , (r * (1 - np.power(r, beta)) + eps))
  del_r = lambda_2

  return del_w, del_r

def second_derivate_jax_fn(i, params, verbose, cond_term_dx, data):
  inputs = params
  delta_l = inputs["delta_l"]
  delta_r = inputs["delta_r"]
  l = inputs["l"]
  r = inputs["r"]
  user = data[i][0]
  time = data[i][1]
  result = data[i][2]
  # l_upd_1, r_upd_1 = calculate_second_derivative(l[user, :], r[:, time], result, verbose=verbose)
  l_upd, r_upd = cond_term_dx(l[user, :], r[:, time], result)
  # print(l_upd, l_upd_1)
  # print(r_upd, r_upd_1)
  # print(l_upd == l_upd_1)
  # print(r_upd == r_upd_1)
  # delta_l[user, :] += l_upd * (N/len(data))
  # delta_r[:, time] += r_upd * (N/len(data))
  delta_l = delta_l.at[user, :].set(l_upd * (N/len(data)) + delta_l[user, :])
  delta_r = delta_r.at[:, time].set(r_upd * (N/len(data)) + delta_r[:, time])
  if verbose:
    print(data[i])
    print(delta_l)
    print(delta_r)
    print(10*"=")
  inputs["delta_l"] = delta_l
  inputs["delta_r"] = delta_r
  return inputs

# @jax.jit
@partial(jax.jit, static_argnames=['verbose'])
def update_params(l, r, data, lr, subkey, verbose=False):
  # print(data)
  delta_l = jnp.zeros(l.shape)
  delta_r = jnp.zeros(r.shape)

  # prior_term_dx = jax.grad(prior_term, argnums=(2, 3))
  cond_term_dx = jax.grad(cond_term, argnums=(0, 1))

  # Calculate the second part of the equation using samples

  second_derivate_jax = partial(second_derivate_jax_fn, verbose=verbose, cond_term_dx=cond_term_dx, data=data)
  inputs = {
      "delta_l": delta_l,
      "delta_r": delta_r,
      "l": l,
      "r": r
  }
  inputs = jax.lax.fori_loop(0, len(data), second_derivate_jax, inputs)
  delta_l = inputs["delta_l"]
  delta_r = inputs["delta_r"]
  l = inputs["l"]
  r = inputs["r"]

  # Calculate the first part of the equation
  lambda_1 = jnp.full(l.shape, 0.1)
  lambda_2 = jnp.full(r.shape, 0.1)
  delta_l2_1, delta_r2_1 = calculate_first_derivative(lambda_1, lambda_2, l, r)
  delta_l += delta_l2_1
  delta_r += delta_r2_1

  # Add noise
  delta_l = delta_l * lr / 2 + jax.random.normal(subkey, shape=l.shape) * lr
  delta_r = delta_r * lr / 2 + jax.random.normal(subkey, shape=r.shape) * lr
  # delta_l = delta_l * lr / 2 + np.random.normal(0, lr, l.shape)
  # delta_r = delta_r * lr / 2 + np.random.normal(0, lr, r.shape)

  l += delta_l
  r += delta_r
  # print(delta_r, delta_l)

  return l, r

isjax = True

def initialize_matrices(isjax=False):
  w = np.random.random((users, types))
  r = np.zeros((types, reward), dtype=np.float32)
  # r = np.full((types, reward), -10, dtype=np.float32)
  if isjax:
    w = jnp.array(w)
    r = jnp.array(r)
  return w, r

def get_batch(data, batch_size=1000):
  return [data[idx] for idx in np.random.choice(len(data), size=batch_size)]

class AnnealerDec:
  def __init__(self, verbose=False, start_lr = 0.001, interrupts_rate=100):
    self.verbose = verbose
    self.lr = start_lr
    self.start_lr = start_lr
    self.int_rate = interrupts_rate
    self.cur_count = 0
    self.b = 0.1
  def update(self, norms, run_full=False):
    self.cur_count += self.int_rate
    self.lr = self.start_lr/ (0.1 * self.cur_count + self.b)
    return 0
  
#=====================================================================================================================
def softmax_with_temperature(logits, temperature=0.05):
  """
  Applies softmax function to logits with temperature scaling.

  Args:
    logits: A list or numpy array of logits.
    temperature: A float value controlling the sharpness of the distribution.

  Returns:
    A numpy array representing the probability distribution.
  """
  logits = np.array(logits) / temperature
  exp_logits = np.exp(logits - np.max(logits)) # subtract max for numerical stability
  probabilities = exp_logits / np.sum(exp_logits)
  return probabilities

def sample_from_softmax(probabilities):
  """
  Samples an index from a probability distribution.

  Args:
    probabilities: A list or numpy array representing the probability distribution.

  Returns:
    An integer representing the sampled index.
  """
  return np.random.choice(len(probabilities), p=probabilities)

#=====================================================================================================================


def gibbs_sampling(data, logger, interrupt = 250, num_epochs = 3000, lr=0.0002, full_run = False):
  annealer = AnnealerDec(start_lr=lr, interrupts_rate=interrupt)
  matrices = []
  w, r = initialize_matrices(isjax)
  key = jax.random.PRNGKey(np.random.randint(0, 10000))
  # print(evaluate(w, r, transition_matrix), annealer.lr)
  logger.start(annealer.lr)
  Rs = []
  for epoch in range(num_epochs):
    if isjax:
      # key, subkey = jax.random.split(key)
      subkey = jax.random.fold_in(key, epoch)
      # print(key, subkey)

      batched_data = jnp.array(get_batch(data))
      # print(w.shape, r.shape, jax.tree.map(jnp.shape, get_batch(data)), type(annealer.lr), subkey)
      w, r = update_params(w, r, batched_data, annealer.lr, subkey, verbose=False)
    else:
      w, r = update_params(w, r, get_batch(data), annealer.lr, verbose=False)
    # print(w, r)
    # norm = evaluate(w, r, transition_matrix)
    # logger.add(norm)
    # print(norm)
    if epoch % interrupt == 0:
      # print(norm)
      if epoch != 0:
        stop_code = annealer.update(logger.value[-1][-1], run_full=full_run)
      else:
        stop_code = 1
      U = prob_types(w)
      R = prob_slots(r, 1)
      Rs.append(R)
      #print(U[:20,:], R)
      matrices.append(np.matmul(U, R))
      #print(matrices[-1][:20,:])
      print("Epoch, LR, Type", epoch, annealer.lr, types)
      # print(np.matmul(U, R))
      if stop_code == -1:
        break
  np.set_printoptions(threshold=np.inf)
  for reward_mat in Rs[-3:]:
    print(reward_mat)
  return matrices[-3:]

class Logger():
  def __init__(self, verbose=False):
    self.verbose = verbose
    self.value = []
    self.ctr = 0
  def data_gen(self):
    self.ctr += 1
  def start(self, lr):
    self.value.append([lr, self.ctr, []])
  def add(self, val):
    self.value[-1][-1].append(val)
  def plot(self):
    # plt.plot([b for a in self.value for b in a[-1]])
    plt.plot(list(itertools.chain.from_iterable([a[-1] for a in self.value])))
  def plot_several(self):
    for ind, val in enumerate(self.value):
      plt.plot(val[-1], label=ind)
    plt.legend()

def get_matrix_final(data, params):
  # data.extend(thompson_sampling(transition_matrix, matrices, batch_size, phase_number))
  # # data.extend(boltzmann_expl(transition_matrix, matrices[-1], batch_size))
  # batches.append(len(data))
  # print("Data Generated Through Thompson Sampling")
  global types
  types = params[1]
  lr_1 = params[0]
  logger = Logger()
  matrices = gibbs_sampling(data, logger, full_run=False, lr=lr_1)
  # print("Gibbs Sampling")
  # intermediate_mats.append(matrices[-1])
  # logger.plot_several()
  # regrets = regret_calculator(transition_matrix, data, batches)
  log_dict = {}
  log_dict["matrix"] = matrices[-1].tolist()
  return matrices[-1], json.dumps(log_dict)

def get_loss(mat, test_data):
  loss = 0
  for data_pt in test_data:
    loss += abs(mat[data_pt[0], data_pt[1]] - data_pt[2])
  return loss/len(test_data)

def get_best_params(data):
  sum_mat_full = np.zeros((users, reward))
  cnt_mat_full = np.zeros((users, reward))
  for data_pt in data:
    sum_mat_full[data_pt[0], data_pt[1]] += data_pt[2]
    cnt_mat_full[data_pt[0], data_pt[1]] += 1
  
  lrs = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
  possible_types = [3, 5, 7]
  train_data = data[:int(0.8*len(data))]
  test_data = data[int(0.8*len(data)):]
  best_params = []
  best_loss = 10000
  for lr in lrs:
    for chosen_type in possible_types:
      mat, _ = get_matrix_final(train_data, [lr, chosen_type])
      if np.any(np.isnan(mat)):
            continue
      loss = get_loss(mat, test_data)
      if loss < best_loss:
        best_loss = loss
        best_params = [lr, chosen_type]
      print(f"LR: {lr}, Loss: {loss}")
  
  mat_completed = sum_mat_full / (cnt_mat_full+eps)
  mask = np.array(cnt_mat_full, dtype=bool)
  mat_completed = mat_completed * mask
  print("========Summary=======")
  print(mat_completed[:20, :])
  print(cnt_mat_full[:20, :])
  print()
  return best_params

#params = get_best_params(data)
#matrix, log_dict = get_matrix_final(data, params)
#params_dict = json.dumps(params)
#print(matrix, log_dict, params_dict)

#===============================================================================================================

def hyperparameter_tune(raw_data, num_of_users, time_slots):
  global N, users, reward, types, eps, data
  users = num_of_users
  reward = time_slots
  types = 5 # user's persona (latent space length)
  # step_size = 0.001
  eps = 1e-15

  # Insert data method here
  data = raw_data
  N = len(data)
  params = get_best_params(data)
  return params

def final_run(raw_data, num_of_users, time_slots, parameters):

  global N, users, reward, types, eps, data
  users = num_of_users
  reward = time_slots
  types = 5 # user's persona (latent space length)
  # step_size = 0.001
  eps = 1e-15

  # Insert data method here
  data = raw_data
  N = len(data)
  t1 = time.time()

  # [Awadhesh] Load the param values
  params = parameters
  matrix, log_dict = get_matrix_final(data, params)
  # matrix = get_matrix_final(data, params)
  if np.any(np.isnan(matrix)):
    matrix = np.nan_to_num(matrix, nan=0.0)
    print("NaN is replaced with 0.0")
  params_dict = json.dumps(params)
  ml_exec_time = time.time() - t1
  return matrix, params_dict, log_dict, ml_exec_time

if __name__ == "__main__":
  data = [(0, 1, 1)] # Input data here
  num_users = 10 # Number of users
  time_slots = 7
  params = hyperparameter_tune(data, num_users, time_slots)
  matrix, _, _, _ = final_run(data, num_users, time_slots, params)