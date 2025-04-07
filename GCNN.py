import torch
import torch.nn as nn

def create_adj(size):
  a = torch.rand(size,size)
  a[a>0.5] = 1
  a[a<=0.5] = 0

  # for illustration we set the diagonal elemtns to zero
  for i in range(a.shape[0]):
    for j in range(a.shape[1]):
      if i==j:
        a[i,j] = 0
  return a

def calc_degree_matrix(a):
  return torch.diag(a.sum(dim=-1))

def create_graph_lapl(a):
  return calc_degree_matrix(a)-a

def calc_degree_matrix_norm(a):
  return torch.diag(torch.pow(a.sum(dim=-1),-0.5))

def create_graph_lapl_norm(a):
  size = a.shape[-1]
  D_norm = calc_degree_matrix_norm(a)
  L_norm = torch.ones(size) - (D_norm @ a @ D_norm )
  return L_norm

def find_eigmax(L):
   with torch.no_grad():
       e1, _ = torch.eig(L, eigenvectors=False)
       return torch.max(e1[:, 0]).item()

def chebyshev_Lapl(X, Lapl, thetas, order):
 list_powers = []
 nodes = Lapl.shape[0]

 T0 = X.float()

 eigmax = find_eigmax(Lapl)
 L_rescaled = (2 * Lapl / eigmax) - torch.eye(nodes)

 y = T0 * thetas[0]
 list_powers.append(y)
 T1 = torch.matmul(L_rescaled, T0)
 list_powers.append(T1 * thetas[1])

 #Computation of: T_k = 2*L_rescaled*T_k-1  -  T_k-2
 for k in range(2, order):
     T2 = 2 * torch.matmul(L_rescaled, T1) - T0
     list_powers.append((T2 * thetas[k]))
     T0, T1 = T1, T2
 y_out = torch.stack(list_powers, dim=-1)
 #the powers may be summed or concatenated. i use concatenation here
 y_out = y_out.view(nodes, -1) # -1 = order* features_of_signal
 return y_out

features = 3
out_features = 50
a = create_adj(10)
L = create_graph_lapl_norm(a)
x = torch.rand(10, features)
power_order = 4 # p-hops
thetas = nn.Parameter(torch.rand(4))

out = chebyshev_Lapl(x,L,thetas,power_order)

print('cheb approx out powers concatenated:', out.shape)
#because we used concatenation  of the powers
#the out features will be power_order * features
linear = nn.Linear(4*3, out_features)

layer_out = linear(out)
print('Layers output:', layer_out.shape)