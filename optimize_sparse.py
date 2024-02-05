###|
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds

baseurl = './cm/variance/' ## other option: cifar100_epochs
type_decision = 'variance'
import glob

# Get a list of all files starting with "confusion_matrix"
files = glob.glob(baseurl+"confusion_matrix*")

# Load each file into a numpy array and append it to a list
confusion_matrices = []
for file in files:
    confusion_matrices.append(np.load(file))

d =  confusion_matrices[0].shape[0]
n = len(confusion_matrices)
confusion_matrices = np.concatenate(confusion_matrices).reshape(n,d,d)

## slice it to dxd
d = 80
confusion_matrices = confusion_matrices[::2,:d,:d] ### slice the data and take every 2nd decision problem
n = confusion_matrices.shape[0]
n_train = confusion_matrices.sum(axis=2)[0,0]

# prompt: fill one of the columns for i row except the ith column with 100 - sum of row
for k in range(confusion_matrices.shape[0]):
  for i in range(d):
      j = np.random.choice([x for x in range(d) if x!=i])
      confusion_matrices[k,i,j] += n_train - np.sum(confusion_matrices[k,i,:])

priors = np.ones((n,d))
priors_prob = priors/priors.sum(axis=0)
actsels_prob = confusion_matrices/confusion_matrices.sum(axis=2).reshape(n,d,1)
actsels = actsels_prob

actsels_prob = np.asarray(actsels_prob)
actsels = np.asarray(actsels)

posterior_probs_by_category = {}
act_probs_by_category = {}
num_actions = actsels_prob.shape[-1]

for categ in range(n): # parametrizes the prior
  act_probs_by_category[categ] = []
  posterior_probs_by_category[categ] = []
  for action in range(num_actions):
    # compute p(a) = prior*p(a|x)
    act_probs_by_category[categ].append(priors_prob[categ,:]@actsels_prob[categ,:,action])
    # compute p(x|a) = p(a|x)*prior/p(a)
    posterior_probs_by_category[categ].append(actsels_prob[categ,:,action]*priors_prob[categ,:]/float(actsels_prob[categ,:,action]@priors_prob[categ,:]))

  act_probs_by_category[categ] = np.array(act_probs_by_category[categ])
  posterior_probs_by_category[categ] = np.array(posterior_probs_by_category[categ])

def NIAS(feasible_vars, pos_beliefs=posterior_probs_by_category): # check if, given posterior belief, does the agent choose the best action
  '''
  RETURNS a list of NIAS inequality values: [\sum_x p_k(x|a) (u_k(x,b)-u_k(x,a)) for a in range(num_actions) for k in range(num_categs)]
  - Every element of list must be non-positive for NIAS to be feasible


  pos_beliefs.values() -> np arrays
  feasible_vars -> np array
  feasible_vars[0:num_categs*num_states*num_actions] -> utilities u_k(x,a)
  feasible_vars[num_categs*num_states*num_actions + 1 : num_categs*num_states*num_actions + num_categs] -> C_k
  feasible_vars[num_categs*num_states*num_actions + num_categs + 1 : num_categs*num_states*num_actions + num_categs + num_categs] -> \lambda_k

  feasible_vars.shape[0] = num_categs*(num_states + 2)
  '''

  belief_shape = pos_beliefs[0].shape
  num_actions = belief_shape[0]
  num_states = belief_shape[1]
  num_categs = len(list(pos_beliefs.keys()))

  assert (feasible_vars.shape)[0] == num_categs*(num_states + 2),"Error: Check dimension of feasible_vars"

  utils = np.ones((num_categs*num_states*num_actions,))*0
  ### fill all diagnol entries with corresponding utilities from feasible_vars[0:num_categs*num_states]
  for i in range(num_categs):
    for j in range(num_states):
        utils[i*num_states*num_actions+j*num_states+j ] = feasible_vars[i*num_states+ j]



  ineq_mat = []
  ineq_counter = 0
  for categ in pos_beliefs.keys(): # NIAS for every category
    for a in range(num_actions): # NIAS for every action in every category
      for b in range(num_actions): # NIAS (compare optimal action a to every non-optimal action b for every category)
        if a!=b:
          ## every element must be less than 0 for NIAS to go through
          ## \sum_{x} p(x|a)*( u(x,b) - u(x,a) ) <= 0
          offset_a = categ*num_states*num_actions + a*num_states
          offset_b = categ*num_states*num_actions + b*num_states

          ineq_mat.append(pos_beliefs[categ][a,:]@np.array(utils[offset_b: offset_b + num_states] - utils[offset_a: offset_a + num_states]))
          ## \sum_x p_k(x|a)*(u_k(x,b) - u_k(x,a)))


          # ineq_mat[ineq_counter, offset_a: offset_a + num_states] = -pos_beliefs[categ][a,:].reshape((1,num_states))
          # ineq_mat[ineq_counter, offset_b: offset_b + num_states] = +pos_beliefs[categ][a,:].reshape((1,num_states))
          #ineq_counter = ineq_counter + 1

          # try:
          #   ineq_mat[ineq_counter, offset_a: offset_a + num_states] = -pos_beliefs[categ][a,:].reshape((1,num_states))
          #   ineq_mat[ineq_counter, offset_b: offset_b + num_states] = +pos_beliefs[categ][a,:].reshape((1,num_states))
          # except:
          #   print(num_categs*num_actions*(num_actions-1),ineq_counter,offset_a,offset_b)
  return ineq_mat

def NIAS_diagonal_utils_margin(feasible_vars, pos_beliefs=posterior_probs_by_category): # check if, given posterior belief, does the agent choose the best action
  '''
  RETURNS a list of NIAS inequality values: [\sum_x p_k(x|a) (u_k(x,b)-u_k(x,a)) for a in range(num_actions) for k in range(num_categs)]
  - Every element of list must be non-positive for NIAS to be feasible


  pos_beliefs.values() -> np arrays
  feasible_vars -> np array
  feasible_vars[0:num_categs*num_states] -> utilities u_k(x,x), x\in\{1,2,..\}
  feasible_vars[num_categs*num_states + 1 : num_categs*num_states num_categs] -> C_k
  feasible_vars[num_categs*num_states + num_categs + 1 : num_categs*num_states + num_categs + num_categs] -> \lambda_k

  feasible_vars.shape[0] = num_categs*(num_states + 2)
  '''

  belief_shape = pos_beliefs[0].shape
  num_actions = belief_shape[0]
  num_states = belief_shape[1]
  num_categs = len(list(pos_beliefs.keys()))

  assert (feasible_vars.shape)[0] == num_categs*(num_states + 2),"Error: Check dimension of feasible_vars"

  # # utils = np.ones((num_categs*num_states*num_actions,))*0
  # ### fill all diagnol entries with corresponding utilities from feasible_vars[0:num_categs*num_states]
  # for i in range(num_categs):
  #   for j in range(num_states):
  #       utils[i*num_states*num_actions+j*num_states+j ] = feasible_vars[i*num_states+ j]

  # make NIAS as a linear inequality (A.feasible vars <= 0)

  num_ineqs = num_categs*num_actions*(num_actions-1)
  ineq_mat = np.zeros((num_ineqs,feasible_vars.shape[0]))

  ineq_counter = 0
  for categ in pos_beliefs.keys(): # NIAS for every category
    for a in range(num_actions): # NIAS for every action in every category
      for b in range(num_actions): # NIAS (compare optimal action a to every non-optimal action b for every category)
        if a!=b:
          ## every element must be less than 0 for NIAS to go through
          ## \sum_{x} p(x|a)*( u(x,b) - u(x,a) ) <= 0

          ineq_mat[ineq_counter,a] = -pos_beliefs[categ][a,a]; ## posterior p(x=a|a).u(x=a,a)
          ineq_mat[ineq_counter,b] =  pos_beliefs[categ][a,b]; ## posterior p(x=b|a).u(x=b,b)

          ineq_counter = ineq_counter + 1;
  return ineq_mat

def NIAC(feasible_vars, pos_beliefs=posterior_probs_by_category,act_prob=act_probs_by_category,act_sels= actsels, priors = priors_prob): # check for relative optimality: for each category, does the chosen action selection policy maximize lambda*expected utility - cost
  '''
  RETURNS a list of NIAC inequality values: [   {exputil(p_{k'}(a|x),u_k(x,a)) - exputil(p_k(a|x),u_k(x,a)) - lambda_k (C_{k'} - C_k) } for k,k' in range(num_categs)**2 and k != k']
  - Every element of list must be non-positive for NIAC to be feasible
  pos_beliefs.values() -> np arrays
  feasible_vars -> np array
  feasible_vars[0:num_categs*num_states*num_actions] -> utilities u_k(x,a)
  feasible_vars[num_categs*num_states*num_actions + 1 : num_categs*num_states*num_actions + num_categs] -> costs C_k
  feasible_vars[num_categs*num_states*num_actions + num_categs + 1 : num_categs*num_states*num_actions + num_categs + num_categs] -> lambda vals \lambda_k

  feasible_vars.shape[0] = num_categs*(num_states*num_actions + 2)

  RETURNS a list of NIAC inequality values: [   {exputil(p_{k'}(a|x),u_k(x,a)) - exputil(p_k(a|x),u_k(x,a)) - lambda_k (C_{k'} - C_k) } for k,k' in range(num_categs)**2 and k != k']
  - Every element of list must be non-positive for NIAC to be feasible

  '''

  belief_shape = pos_beliefs[0].shape
  num_actions = belief_shape[0]
  num_states = belief_shape[1]
  num_categs = len(list(pos_beliefs.keys()))

  assert (feasible_vars.shape)[0] == num_categs*(num_states + 2),"Error: Check dimension of feasible_vars"


  utils = np.ones((num_categs*num_states*num_actions,))*0
    ### fill all diagnol entries with corresponding utilities from feasible_vars[0:num_categs*num_states]
  for i in range(num_categs):
    for j in range(num_states):
        utils[i*num_states*num_actions+j*num_states+j ] = feasible_vars[i*num_states+ j]

  costs = feasible_vars[num_categs*num_states: num_categs*num_states + num_categs]
  lambdas = feasible_vars[num_categs*num_states + num_categs:]

  ineq_mat = []
  for categ1 in pos_beliefs.keys(): # NIAC is for a pair of categories - categ1 has higher expected utility
    for categ2 in pos_beliefs.keys():
      if categ1 != categ2:
        offset = categ1*num_states*num_actions
        exputil1 = sum([ act_prob[categ1][a]*(pos_beliefs[categ1][a,:]@utils[ offset + a*num_states: offset + (a+1)*num_states]) for a in range(num_actions) ])
        # \sum_{x,a} p_k(x|a)p_k(a)u_k(x,a)

        cross_pos_beliefs = np.zeros(pos_beliefs[0].shape)
        cross_act_probs = np.zeros((num_actions,))
        act_sel_2 = np.array(act_sels[categ2])

        for a in range(num_actions):
          cross_pos_beliefs[a,:] = priors[categ1]*np.array(act_sel_2[:,a]).reshape((num_states,))  #### (prior(x)p_{k'}(a|x))
          cross_act_probs[a] = sum(cross_pos_beliefs[a,:]) #### sum_{x} (prior(x')p_{k'}(a|x'))
          cross_pos_beliefs[a,:] = cross_pos_beliefs[a,:]/cross_act_probs[a] #### (prior(x)p_{k'}(a|x))/sum_{x} (prior(x')p_{k'}(a|x'))

        offset = categ1*num_actions*num_states
        exputil2 = sum([ cross_act_probs[a]*max( [cross_pos_beliefs[a,:]@utils[offset + a1*num_states: offset + (a1+1)*num_states] for a1 in range(num_actions)]  ) for a in range(num_actions)])
        # \sum_{a} p_{k'}(a) (\max_{b} \sum_{x} p_{k'}(x|a) u_{k}(x,a)) - cross max utility gained by using action selection from k' in category k

        ## Debugging blurb: print(categ1,categ2,'utils',exputil1,exputil2)
        ineq_mat.append(  exputil2 - exputil1 - lambdas[categ1]*(costs[categ2] - costs[categ1])  )

  return ineq_mat

def sparsest_utility(feasible_vars,pos_beliefs = posterior_probs_by_category):
  '''
  RETURNS \sum_{x,a,k} |u_k(x,a)| -> sum of absolute values of the utilities (just the sum since utilities are constrained to be >= 0)

  pos_beliefs.values() -> np arrays
  feasible_vars -> np array
  feasible_vars[0:num_categs*num_states*num_actions] -> utilities u_k(x,a)
  feasible_vars[num_categs*num_states*num_actions + 1 : num_categs*num_states*num_actions + num_categs] -> costs C_k
  feasible_vars[num_categs*num_states*num_actions + num_categs + 1 : num_categs*num_states*num_actions + num_categs + num_categs] -> lambda vals \lambda_k

  feasible_vars.shape[0] = num_categs*(num_states*num_actions + 2)

  '''

  belief_shape = pos_beliefs[0].shape
  num_actions = belief_shape[0]
  num_states = belief_shape[1]
  num_categs = len(list(pos_beliefs.keys()))

  assert (feasible_vars.shape)[0] == num_categs*(num_states + 2),"Error: Check dimension of feasible_vars"
  return float(sum(np.absolute(feasible_vars[:num_categs*num_actions])))

def grad_sparsest_utility(feasible_vars,pos_beliefs = posterior_probs_by_category):
  '''
  RETURNS \sum_{x,a,k} |u_k(x,a)| -> sum of absolute values of the utilities (just the sum since utilities are constrained to be >= 0)

  pos_beliefs.values() -> np arrays
  feasible_vars -> np array
  feasible_vars[0:num_categs*num_states*num_actions] -> utilities u_k(x,a)
  feasible_vars[num_categs*num_states*num_actions + 1 : num_categs*num_states*num_actions + num_categs] -> costs C_k
  feasible_vars[num_categs*num_states*num_actions + num_categs + 1 : num_categs*num_states*num_actions + num_categs + num_categs] -> lambda vals \lambda_k

  feasible_vars.shape[0] = num_categs*(num_states*num_actions + 2)

  '''

  belief_shape = pos_beliefs[0].shape
  num_actions = belief_shape[0]
  num_states = belief_shape[1]
  num_categs = len(list(pos_beliefs.keys()))

  assert (feasible_vars.shape)[0] == num_categs*(num_actions + 2),"Error: Check dimension of feasible_vars"
  return np.ones((num_categs*(num_states + 2),))

thresh_nias = -0.0001
thresh_niac = -0.001
thresh_vars = 0.001
belief_shape = posterior_probs_by_category[0].shape

num_actions = belief_shape[0]
num_states = belief_shape[1]
num_categs = len(list(posterior_probs_by_category.keys()))


Nfeval =1
# minimize sum of absolute utilities subject to NIAS <= thresh_nias, NIAC <= thresh_niac
def callbackF(Xi):
    global Nfeval
    print(f"{Nfeval} {Xi[0]}")
    Nfeval += 1

dim_diagonal_utils = num_categs*(num_states + 2)
dim_ineqs = num_categs*(num_states)*(num_states-1)
feasible_vars_cand = np.zeros((dim_diagonal_utils,))

#### TAKES A LOT OF TIME ####
res_sparsest = minimize(fun = sparsest_utility,
                        x0= 0.5*np.ones((num_categs*(num_states + 2),)),
                        bounds =  [(thresh_vars,1000)]*(num_categs*(num_states + 2)), # every element is between [0,1]
                        jac = grad_sparsest_utility,
                        #constraints = (NonlinearConstraint(NIAS, -np.inf, thresh_nias),NonlinearConstraint(NIAC, -np.inf, thresh_niac),),
                        constraints = (LinearConstraint(NIAS_diagonal_utils_margin(feasible_vars_cand), -np.inf*np.ones(dim_ineqs), np.zeros(dim_ineqs) ),NonlinearConstraint(NIAC, -np.inf, 0),),

                        options={'disp': True,'maxiter':60},
                        callback=callbackF,

                        )


vals = res_sparsest.x
sparse_utils = vals[:num_categs*num_actions]
costs = vals[num_categs*num_states:num_categs*num_states + num_categs]
lambdas = vals[num_categs*num_actions + num_categs:]

##### shape utilities into a dictionary:
sparse_utils_dict = {}
for categ in range(num_categs):
  utils = np.ones((num_states,num_actions))*0
  for j in range(num_states):
    utils[j,j ] = sparse_utils[categ*num_states+ j]
  sparse_utils_dict[categ] = utils

sparse_utils_dict['Summary']='Storing sparsest utilities of 8 categories - rows are actions, columns are states'


sparse_final_utils_costs_lambdas_data = {
    'utils':sparse_utils_dict,
    'inattention_costs':costs,
    'lagrange_cost_multipliers':lambdas
}


# Save Sparse utils and costs and lambdas
with open(baseurl + 'dl_final_sparse_diagonal' + type_decision + '.pkl', 'wb') as f:
  pickle.dump(sparse_final_utils_costs_lambdas_data,f)
