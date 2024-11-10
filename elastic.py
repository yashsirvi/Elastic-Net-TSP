import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import animation
import time
import os

class ElasticTSP:
    def __init__(self, pts, m=None):
        """
        Parameters
        ----------
        pts : ndarray
            2D array of points in the plane.
        m : int, optional
            Number of points in the elastic band. By default, it is 2.5 times the number of input points.
        """
        self.pts = pts
        self.n = len(pts)
        self.m = m or int(2 * self.n) # num points in the elastic band
        
        # parameters
        self.alpha = 1.0
        self.beta = 0.2
        self.temp = 0.2
        self.decay = 0.99
        self.min_temp = 0.01
        
        # making a ring of points as an initial state for the elastic band
        centre = np.mean(pts, axis=0)
        radius = 0.5 * np.std(pts)
        t = np.linspace(0, 2*np.pi, self.m, endpoint=False)
        self.ring = centre + radius * np.column_stack((np.cos(t), np.sin(t)))
        
    def get_weights(self, K):
        """
        Parameters
        ----------
        K : float
            Length Parameter
        ;;;;
        Returns
        -------
        w : ndarray
            2D array with shape (n, m), where n is the number of points and m is the number of points in the elastic band.
        """
        p1 = self.pts[:, np.newaxis, :]
        p2 = self.ring[np.newaxis, :, :]
        
        d = np.sum((p1 - p2) ** 2, axis=2) # pairwise distances^2
        w = np.exp(-d / (2 * K * K))    # gaussian function exp(-||x-y||^2 / (2 * K^2))
        
        w /= (w.sum(axis=1, keepdims=True) + 1e-10) # pairwise weights
        return w
    
    def step(self, w):
        f1 = np.zeros_like(self.ring)
        for i in range(self.n):
            d = self.pts[i, np.newaxis, :] - self.ring
            f1 += w[i, :, np.newaxis] * d
        
        prev = np.roll(self.ring, 1, axis=0)
        nxt = np.roll(self.ring, -1, axis=0)
        f2 = prev + nxt - 2 * self.ring
        
        self.ring += self.alpha * f1 + self.beta * f2
        
    def solve(self, iters=1000, tol=1e-4):
        t = self.temp
        hist = [self.ring.copy()]
        
        for _ in tqdm(range(iters)):
            prev = self.ring.copy()
            w = self.get_weights(t)
            self.step(w)
            hist.append(self.ring.copy())
            
            if np.max(np.abs(self.ring - prev)) < tol:
                break
                
            t = max(t * self.decay, self.min_temp)
            
        return hist
    
    def get_path(self):
        d = np.sum((self.pts[:, np.newaxis, :] - self.ring[np.newaxis, :, :]) ** 2, axis=2)
        return np.argsort(np.argmin(d, axis=1))
    
    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.scatter(self.pts[:, 0], self.pts[:, 1], c='red', s=50, label='Points')
        plt.plot(self.ring[:, 0], self.ring[:, 1], 'bo-', alpha=0.5, label='Elastic Band')
        plt.plot(self.ring[[0, -1], 0], self.ring[[0, -1], 1], 'b-', alpha=0.5)
        plt.gca().set_aspect('equal')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Elastic Band Plot')
        plt.legend()
        plt.show()

def animate_solution(history, coords, save=True, save_path="."):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(coords[:,0], coords[:,1], c='red', s=100, label='Cities')
    
    band_line, = ax.plot([], [], 'bo-', alpha=0.5, label='Elastic Band')
    band_line_closed, = ax.plot([], [], 'bo-', alpha=0.5)
    
    def init():
        band_line.set_data([], [])
        band_line_closed.set_data([], [])
        return band_line, band_line_closed
    
    def update(frame):
        band = history[frame]
        band_line.set_data(band[:,0], band[:,1])
        band_line_closed.set_data([band[-1,0], band[0,0]], [band[-1,1], band[0,1]])
        return band_line, band_line_closed
    
    ani = animation.FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True, repeat=False, interval=1)
    
    plt.title("Elastic Band TSP Solution")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    time_ymdhs = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    filename = "solved_eg_" + time_ymdhs + ".gif"
    save_path = os.path.join(save_path, filename)
    # Save the animation as a GIF
    if save:
        ani.save(save_path, writer='pillow', fps=300)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    cities = np.random.rand(100, 2)
    # npzfile = "tsp225.npz"
    # data = np.load(npzfile)
    # cities = data["coords"][0]
    
    solver = ElasticTSP(cities)
    hist = solver.solve(iters=500)
    path = solver.get_path()
    
    solver.plot()
    
    anim = animate_solution(hist, cities, save=False)