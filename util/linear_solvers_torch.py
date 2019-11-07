import numpy as np
import torch
import time


def cholesky(device_config, K, Y):
    """
    Solve linear system using a cholesky solver.
    Params:
        K - Covariance Matrix
        Y - Target Labels
    """

    with torch.no_grad():
        L = torch.cholesky(K, upper=False)
        solution = torch.cholesky_solve(Y, L, upper=False)

    return solution

def cg(device_config, K, Y, init=None, tol=1e-5, atol=1e-9, max_iterations=15000):
    """
    Solve linear system using the conjugate gradient (CG) method.
    Params:
        K - Covariance Matrix
        Y - Target labels
        init - Initial solution (if None, it is initialized to 0)
        tol, atol - Termination criterion: norm(residual) <= max(tol*norm(y), atol)
        max_iterations - maximum number of iterations the algorithm runs for
            if no earlier termination has been achieved
            
    Returns:
        - The solution to the linear system
        - A list with the iterations needed per output dimension
            
    CAUTION:
        The number of iterations required to converge depends highly on the conditioning of the linear system.
        It makes sense to fit the kernel hyperparameters in a low-scale setting first.
        The right choice of kernel scale and diagonal noise improves convergence significantly!
    """

    N = K.shape[0]
    num_gpus = len(device_config['active_gpus'])
    gpu_ids = device_config['active_gpus']

    if num_gpus > 0:
        main_gpu = torch.device('cuda:' + str(device_config['active_gpus'][0]))
    
    if init is None:
        init = torch.zeros(Y.shape)

    X = init
    R = Y - torch.matmul(K, X) # initialise residuals
    
    if num_gpus > 0:
        split_size = K.shape[0] // num_gpus
        
        Ks = []
        for i in range(num_gpus):
            # split the kernel among gpus
            # this allows us to store very large kernel matrices
            if i < (num_gpus-1):
                Ks.append(
                    K[i*split_size:(i+1)*split_size].to('cuda:' + str(gpu_ids[i]))
                )
            else:
                Ks.append(
                    K[i*split_size:].to('cuda:' + str(gpu_ids[i]))
                )

    iterations = []
    residual_norms = []
    solutions = []

    # We have to solve one linear system (Kx=y) for every output dimension
    for dim in range(Y.shape[1]):
        print('Starting CG for dimension {}'.format(dim))
        # get current residual vector
        r = R[:, dim][:, None]
        # get current solution
        x = X[:, dim][:, None]

        p = r

        t = 0

        if num_gpus > 0:
            # we copy p to every gpu unit
            x = x.to(main_gpu)
            r = r.to(main_gpu)
            ps = [p.to('cuda:' + str(gpu_ids[i])) for i in range(num_gpus)]
        else:
            ps = [p]

        while True:
            with torch.no_grad():
                if num_gpus > 0:
                    # we compute one split of Kp on every GPU
                    # apart from memory savings, this gives a bit of acceleration
                    Kps = [Ks[i].mm(ps[i]).to(main_gpu) for i in range(num_gpus)]
                    Kp = torch.cat(Kps, dim=0)
                else:
                    Kp = K.mm(ps[0])
                    
                pKp = ps[0].t().mm(Kp)

                alpha = r.t().mm(r) / pKp
                x = x + alpha*ps[0]
                r_prev = r

                r = r - alpha * Kp

                residual_norm_sq = r.t().mm(r)
                if ((torch.sqrt(residual_norm_sq).item() <= max(tol*np.linalg.norm(Y[:, dim]), atol*N)) or (t>max_iterations)):
                    residual_norms.append(torch.sqrt(residual_norm_sq).item())
                    break

                beta = residual_norm_sq / r_prev.t().mm(r_prev)
                ps[0] = r + beta*ps[0]

                if num_gpus > 0:
                    # we need to send the updated p to each gpu
                    ps = [ps[0].to('cuda:' + str(gpu_ids[i])) for i in range(num_gpus)]

                t = t + 1

        print('Iterations needed: {}'.format(t))
        print('Residual norm: {}'.format(residual_norms[-1]))
        iterations.append(t)
        solutions.append(x)

    solution = torch.cat(solutions, dim=1)

    if device_config['use_cpu_memory']:
        solution = solution.cpu()

    return solution, iterations, residual_norms
