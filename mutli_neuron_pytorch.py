import torch
import torch.nn as nn

X = torch.tensor([
    [1., 2.],
    [2., 1.],
    [2., 3.],
    [3., 2.],
    [4., 5.],
    [5., 4.],
    [5., 6.],
    [6., 5.]
])

y = torch.tensor([
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [1.],
    [1.]
])

model = nn.Sequential(
    nn.Linear(2, 4),   # X -> hidden layer
    nn.ReLU(),
    nn.Linear(4, 1),   # hidden -> output
    nn.Sigmoid()
)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):

    optimizer.zero_grad()
    Y = model(X)
    loss = loss_fn(Y, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")