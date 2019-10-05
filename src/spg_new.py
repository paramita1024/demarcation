"""TODO."""

import numpy as np

# import matplotlib.pyplot as plt
class spg:
    def __init__(self):
        pass

    def solve(self,x0, f, g, proj, m, eps, maxit, callback=None):
        """TODO."""
        # print "inside spg"
        alpha_min = 1e-3
        alpha_max = 1e3

        f_hist = np.zeros(maxit)

        results = {
            'feval': 0,
            'geval': 0,
            
        }
        #***********
        # print "init my buffer"
        # my_buffer = list([])
        #*************
        def linesearch(x_k, f_k, g_k, d_k, k):
            gamma = 1e-4
            sigma_1 = 0.1
            sigma_2 = 0.9

            f_max = np.max(f_hist[max(0, k - m + 1):k + 1])
            delta = np.dot(g_k, d_k)

            x_p = x_k + d_k
            lam = 1

            f_p = f(x_p)
            results['feval'] = results['feval'] + 1
            while f_p > f_max + gamma * lam * delta:
                lam_t = 0.5 * (lam**2) * delta / (f_p - f_k - lam * delta)
                if lam_t >= sigma_1 and lam_t <= sigma_2 * lam:
                    lam = lam_t
                else:
                    lam = lam / 2.0
                x_p = x_k + lam * d_k
                f_p = f(x_p)
                results['feval'] = results['feval'] + 1

            return lam

        # If x_0 \not\in \Omega, replace x_0 by P(x_0)
        x = proj(np.copy(x0))

        f_new = f(x)
        g_new = g(x)
        results['feval'] = results['feval'] + 1
        results['geval'] = results['geval'] + 1
        d = proj(x - g_new) - x
        alpha = min(alpha_max, max(alpha_min, 1 / np.max(d)))

        results['bestF'] = None
        results['bestX'] = np.copy(x)

        for k in range(maxit):
            f_k = f_new
            g_k = g_new
            f_hist[k] = f_k
            if results['bestF'] is None or f_k < results['bestF']:
                results['bestF'] = f_k
                results['bestX'][:] = x

            if callback:
                callback(k, results['bestF'])
            d = proj(x - alpha * g_k) - x
            if np.linalg.norm(d) < eps:
                break

            lam = linesearch(x, f_k, g_k, d, k) or 1.0
            s = lam * d
            x += s
            f_new = f(x)
            g_new = g(x)
            #**********
            # print "append"
            # my_buffer.append(f_new)
            #**********
            results['feval'] = results['feval'] + 1
            results['geval'] = results['geval'] + 1
            y = g_new - g_k
            beta = np.dot(s, y)
            if beta < 0:
                alpha = alpha_max
            else:
                alpha = min(alpha_max, max(alpha_min, np.dot(s, s) / beta))
        #**********************
        # results['buffer']=my_buffer
        # plt.plot(my_buffer)
        # plt.show()
        #*********************
        # print "exiting spg"
        return results
