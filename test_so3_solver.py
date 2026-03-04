import torch

from flow_matching.path.geodesic import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils.manifolds import SO3, FlatTorus, Product


scheduler = CondOTScheduler()


manifold = Product(input_dim=5, manifolds=[(SO3(), 4, 3), (FlatTorus(), 1)])


x_0 = manifold.projx(torch.randn(1, 5))
x_1 = manifold.projx(torch.randn(1, 5))

# manifold = FlatTorus()
# x_0 = torch.tensor([[0.0]])
# x_1 = torch.tensor([[1.0]])

path = GeodesicProbPath(scheduler, manifold)


def velocity_model(x, t):
    t = t.unsqueeze(0)
    return path.sample(x_0, x_1, t, space="tangent_space").dx_t


solver = RiemannianODESolver(manifold, velocity_model)

result = solver.sample(
    x_0, step_size=0.5, time_grid=torch.tensor([0.0, 1.0]), method="euler_riemannian"
)

print(x_1.squeeze(0))
print(result.squeeze(0))

print(manifold.dist(x_1.squeeze(0), result.squeeze(0)))

# tensor([-0.0281, -0.6316, 0.7450, 0.2129])
# tensor([-0.0281, -0.6316, 0.7450, 0.2129])
# tensor([1.8781])
