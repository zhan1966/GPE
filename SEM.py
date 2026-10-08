import jax.scipy as jsp
import jax.numpy as jnp
import time
from jax import config
config.update("jax_enable_x64", True)
# config.update('jax_traceback_filtering', 'off')

def JacobiGQ(alpha,beta,N):
    '''
    Purpose: Compute the N'th order Gauss quadrature points, x,
    and weights, w, associated with the Jacobi polynomial, of type (alpha,beta) > -1 ( <> -0.5).
    '''
    if N==0:
        x = -(alpha-beta)/(alpha+beta+2)
        w = 2
    else:
        # Form symmetric matrix from recurrence.
        h1 = 2*jnp.linspace(0,N,N+1)+alpha+beta;
        J = jnp.diag(-1/2*(alpha**2-beta**2)/(h1+2)/h1) +\
            jnp.diag(2/(h1[:N]+2)*jnp.sqrt(jnp.linspace(1,N,N)*(jnp.linspace(1,N,N)+alpha+beta)*\
            (jnp.linspace(1,N,N)+alpha)*(jnp.linspace(1,N,N)+beta)/(h1[:N]+1)/(h1[:N]+3)),1)
        if (alpha+beta<10*jnp.finfo(jnp.float32).eps):
            J = J.at[0,0].set(0.0);
        J = J + jnp.transpose(J);
        # Compute quadrature by eigenvalue solve
        D,V = jsp.linalg.eigh(J)
        x = jnp.real(D)
        w = jnp.transpose(V[0,:])**2*2**(alpha+beta+1)/(alpha+beta+1)*\
            jsp.special.gamma(alpha+1)*jsp.special.gamma(beta+1)/jsp.special.gamma(alpha+beta+1)
    return x,w
    
def JacobiGL(alpha,beta,N):
    '''
    Purpose: Compute the N'th order Gauss Lobatto quadrature
    points, x, associated with the Jacobi polynomial of type (alpha,beta) > -1 ( <> -0.5).
    '''
    if N==1:
        x = jnp.linspace(-1,1,2)
    else:
        xint,wint = JacobiGQ(alpha+1,beta+1,N-2);
        x = jnp.append(-1.0,xint)
        x = jnp.append(x,1.0)
    return x
    
def JacobiP(x,alpha,beta,N):
    '''
    Purpose: Evaluate Jacobi Polynomial of type (alpha,beta) > -1
              (alpha+beta <> -1) at points x for order N and returns P[1:length(xp))]
    Note   : They are normalized to be orthonormal.
    Turn points into row if needed.
    '''
    PL = jnp.zeros([N+1,len(x)])
    # Initial values P_0(x) and P_1(x)
    gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*jsp.special.gamma(alpha+1)*\
            jsp.special.gamma(beta+1)/jsp.special.gamma(alpha+beta+1);
    PL = PL.at[0,:].set(1/jnp.sqrt(gamma0))
    if N==0:
        P = jnp.transpose(PL)
    else:
        gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0;
        PL = PL.at[1,:].set(((alpha+beta+2)*x/2 + (alpha-beta)/2)/jnp.sqrt(gamma1))
        if N==1:
            P = jnp.transpose(PL[N+1,:])
        else:
            # Repeat value in recurrence.
            aold = 2/(2+alpha+beta)*jnp.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
            for i in range(1,N):
                h1 = 2*i+alpha+beta;
                anew = 2/(h1+2)*jnp.sqrt((i+1)*(i+1+alpha+beta)*(i+1+alpha)*(i+1+beta)/(h1+1)/(h1+3))
                bnew = - (alpha**2-beta**2)/h1/(h1+2)
                PL = PL.at[i+1,:].set(1/anew*(-aold*PL[i-1,:]+(x-bnew)*PL[i,:]))
                aold = anew
        P = jnp.transpose(PL[N+1,:])
    return P
    
def LegendreD(N):
    '''
    LegendreD compute D = differentiation matrix, r&w are Legendre Gauss-Lobatto
    quadrature points and weights
    '''
    Np = N+1
    r = JacobiGL(0,0,N)
    # weight is equal to 2/(N*(N+1))*1/(L_N(r))^2 where L_N is the Legendre
    # polynomial; JacobiP is the normalized Jacobi polynomial, i.e.,
    # JacobiP(0,0,r)=L_N(r)*sqrt((2*N+1)/2)
    w = (2*N+1)/(N**2+N)/(JacobiP(r,0,0,N))**2
    # Let l_i(r) be the Lagrange interpolation polynomial
    # Next compute the derivative of l_j(r) evaluated at r_i. We have
    # D_{ij}=l'_j(r_i)=omega_j/omega_i/(r_i-r_j), for i not equal to j.
    # see page 81 in D. Kopriva's book on spectral method.
    Distance = (r*jnp.ones([1,N+1])).transpose()-jnp.ones([N+1,1])*r+jnp.eye(N+1)
    omega = jnp.prod(Distance,axis=1)
    D = jnp.matmul(jnp.matmul(jnp.diag(omega),(1/Distance)),jnp.diag(1/omega)) # off diagonal entries
    # compute the diag entries by Negative Sum Tricks, see p.81 in Kopriva's book
    D -= jnp.diag(jnp.diagonal(D))
    D += -jnp.diag(jnp.sum(D,axis=1))
    return D,r,w
    
def cal_matrix(Np,Ncell,BC,L):
    N = Np
    if BC=='dirichlet':
        n = Np*Ncell-1
    elif BC=='periodic':
        n = Np*Ncell
    elif BC=='neumann':
        n = Np*Ncell+1
    [D,r,w] = LegendreD(N)
    # Generate the mesh with Ncell intervals and each interval has GL points
    # Domain is [Left, Right]
    Left = -L; Right = L;
    Length = Right - Left; dx = Length/Ncell
    for j in range(Ncell):
        cell_left = Left + dx*j;
        local_points = cell_left + dx/2 + r*dx/2
        if j==0:
            x = local_points
        else:
            x = jnp.append(x,local_points[1:])
    S_local = jnp.matmul(jnp.matmul(jnp.transpose(D),jnp.diag(w)),D)
    for j in range(Ncell):
        # global stiffness and lumped mass matrices if treating cells sperately
        if j==0:
            S = S_local
            M = jnp.diag(w)
        else:
            S = jsp.linalg.block_diag(S_local,S)
            M = jsp.linalg.block_diag(jnp.diag(w),M)
    # Next step: "glue" the cells
    Np = N + 1 # number of points in each cell
    Glue = jnp.zeros([Ncell*Np-Ncell+1, Ncell*Np])
    for j in range(Ncell):
        rowstart_index = (j)*Np+1-j;
        rowend_index = rowstart_index+Np-1;
        colstart_index = (j)*Np+1;
        colend_index = colstart_index+Np-1;
        Glue = Glue.at[rowstart_index-1:rowend_index,colstart_index-1:colend_index].set(jnp.eye(Np))
    if BC=='periodic':
        Glue = Glue.at[0,-1].set(1.0)
        Glue = Glue.at[-1,0].set(1.0)
    S = jnp.matmul(jnp.matmul(Glue,S),jnp.transpose(Glue))
    M = jnp.matmul(jnp.matmul(Glue,M),jnp.transpose(Glue))
    H = jnp.matmul(jnp.diag(1/jnp.diag(M)),S)
    if BC=='dirichlet':
        S = S[1:-1,1:-1]
        M = M[1:-1,1:-1]
        H = jnp.matmul(jnp.diag(1/jnp.diag(M)),S)
        x = x[1:-1]
        ex = jnp.ones([n,])
    elif BC=='periodic':
        S = S[:-1,:-1]
        M = M[:-1,:-1]
        H = H[:-1,:-1]
        x = x[:-1]
        ex = jnp.ones([n,])
    elif BC=='neumann':
        ex = jnp.ones([n,])
    # the the discrete Laplacian is (H kron I+ I kron H)*vec(U)
    # which is (T kron T)([eigenvalues]*)(T^{-1} kron T^{-1})vec(U)

    # for stability, solve a symmetric eigenvalue problem numerically:
    # H=M^{-1}S=M^{-1/2}S1 M^{1/2}=M^{-1/2}T lambda T^{-1} M^{1/2}=V lambda V^{-1}
    M_half_inv = jnp.diag(1/jnp.sqrt(jnp.diag(M)))
    S1 = jnp.matmul(jnp.matmul(M_half_inv,S),M_half_inv)
    S1 = (S1+jnp.transpose(S1))/2
    eigs,T = jsp.linalg.eigh(S1)
    h = dx/2
    eigs = eigs/((dx/2)**2)
    T = jnp.matmul(M_half_inv,T)
    H = H/(h**2)
    return h,x,ex,T,H,eigs

def cal_matrix_V1(Np,Ncell,BC,L,V_func):
    '''
    Same as cal_matrix but for the operator -Laplacian + V1(x)
    where V1 is a separable potential evaluated at grid points.
    The potential is incorporated into the 1D eigendecomposition:
      S1 = M^{-1/2} S M^{-1/2} + h^2 * diag(V(x))
    so that eigs/h^2 gives physical eigenvalues including the potential.
    '''
    N = Np
    if BC=='dirichlet':
        n = Np*Ncell-1
    elif BC=='periodic':
        n = Np*Ncell
    elif BC=='neumann':
        n = Np*Ncell+1
    [D,r,w] = LegendreD(N)
    Left = -L; Right = L;
    Length = Right - Left; dx = Length/Ncell
    for j in range(Ncell):
        cell_left = Left + dx*j;
        local_points = cell_left + dx/2 + r*dx/2
        if j==0:
            x = local_points
        else:
            x = jnp.append(x,local_points[1:])
    S_local = jnp.matmul(jnp.matmul(jnp.transpose(D),jnp.diag(w)),D)
    for j in range(Ncell):
        if j==0:
            S = S_local
            M = jnp.diag(w)
        else:
            S = jsp.linalg.block_diag(S_local,S)
            M = jsp.linalg.block_diag(jnp.diag(w),M)
    Np = N + 1
    Glue = jnp.zeros([Ncell*Np-Ncell+1, Ncell*Np])
    for j in range(Ncell):
        rowstart_index = (j)*Np+1-j;
        rowend_index = rowstart_index+Np-1;
        colstart_index = (j)*Np+1;
        colend_index = colstart_index+Np-1;
        Glue = Glue.at[rowstart_index-1:rowend_index,colstart_index-1:colend_index].set(jnp.eye(Np))
    if BC=='periodic':
        Glue = Glue.at[0,-1].set(1.0)
        Glue = Glue.at[-1,0].set(1.0)
    S = jnp.matmul(jnp.matmul(Glue,S),jnp.transpose(Glue))
    M = jnp.matmul(jnp.matmul(Glue,M),jnp.transpose(Glue))
    H = jnp.matmul(jnp.diag(1/jnp.diag(M)),S)
    if BC=='dirichlet':
        S = S[1:-1,1:-1]
        M = M[1:-1,1:-1]
        H = jnp.matmul(jnp.diag(1/jnp.diag(M)),S)
        x = x[1:-1]
        ex = jnp.ones([n,])
    elif BC=='periodic':
        S = S[:-1,:-1]
        M = M[:-1,:-1]
        H = H[:-1,:-1]
        x = x[:-1]
        ex = jnp.ones([n,])
    elif BC=='neumann':
        ex = jnp.ones([n,])

    M_half_inv = jnp.diag(1/jnp.sqrt(jnp.diag(M)))
    S1 = jnp.matmul(jnp.matmul(M_half_inv,S),M_half_inv)
    S1 = (S1+jnp.transpose(S1))/2
    h = dx/2
    # Add potential in reference coordinates: h^2 * V(x)
    S1 = S1 + h**2 * jnp.diag(V_func(x))
    eigs,T = jsp.linalg.eigh(S1)
    eigs = eigs/((dx/2)**2)
    T = jnp.matmul(M_half_inv,T)
    H = H/(h**2) + jnp.diag(V_func(x))
    return h,x,ex,T,H,eigs

def cal_matrix_V1_fast(Np,Ncell,BC,L,V_func):
    '''
    Vectorized version of cal_matrix_V1. No Python for-loops over cells.
    Uses scatter-add to assemble global stiffness and mass matrices.
    Returns (h, x, T, invT, eigs) where invT = T0^T @ M^{1/2}
    is computed analytically from the orthogonal eigenvectors T0 of eigh.
    '''
    k = Np  # polynomial order
    if BC=='dirichlet':
        n = k*Ncell-1
    elif BC=='periodic':
        n = k*Ncell
    elif BC=='neumann':
        n = k*Ncell+1
    [D,r,w] = LegendreD(k)
    Left = -L; Right = L
    dx = (Right-Left)/Ncell
    h = dx/2
    Np_local = k+1  # number of points per cell

    # --- Mesh points (vectorized) ---
    cell_lefts = Left + dx*jnp.arange(Ncell)
    cell_points = cell_lefts[:,None] + h + r[None,:]*h  # (Ncell, Np_local)
    # Global: first k nodes from each cell, plus last node of last cell
    x = jnp.concatenate([cell_points[:,:k].ravel(), cell_points[-1:,-1:].ravel()])
    # n_global = k*Ncell + 1 nodes before BC trim

    # --- Global node indices for each cell ---
    cell_node_idx = jnp.arange(Ncell)[:,None]*k + jnp.arange(Np_local)[None,:]  # (Ncell, Np_local)
    n_global = k*Ncell+1
    if BC=='periodic':
        cell_node_idx = cell_node_idx % (k*Ncell)
        n_global = k*Ncell

    # --- Lumped mass diagonal (scatter-add) ---
    m_diag = jnp.zeros(n_global).at[cell_node_idx.ravel()].add(jnp.tile(w, Ncell))

    # --- Stiffness matrix (scatter-add) ---
    S_local = jnp.matmul(jnp.matmul(jnp.transpose(D),jnp.diag(w)),D)
    row_idx = jnp.broadcast_to(cell_node_idx[:,:,None], (Ncell,Np_local,Np_local)).ravel()
    col_idx = jnp.broadcast_to(cell_node_idx[:,None,:], (Ncell,Np_local,Np_local)).ravel()
    S = jnp.zeros((n_global,n_global)).at[row_idx,col_idx].add(jnp.tile(S_local.ravel(), Ncell))

    # --- Apply BC ---
    if BC=='dirichlet':
        S = S[1:-1,1:-1]
        m_diag = m_diag[1:-1]
        x = x[1:-1]
    elif BC=='periodic':
        x = x[:n]

    # --- Symmetric eigendecomposition with potential ---
    m_half_inv = 1.0/jnp.sqrt(m_diag)
    m_half = jnp.sqrt(m_diag)
    S1 = (m_half_inv[:,None]*S)*m_half_inv[None,:]  # M^{-1/2} S M^{-1/2}
    S1 = (S1+S1.T)/2
    S1 = S1 + h**2 * jnp.diag(V_func(x))
    eigs,T0 = jsp.linalg.eigh(S1)
    eigs = eigs/(h**2)
    # T = M^{-1/2} T0,  T^{-1} = T0^T M^{1/2}  (T0 orthogonal from eigh)
    T = m_half_inv[:,None]*T0
    invT = T0.T*m_half[None,:]
    return h,x,T,invT,eigs

