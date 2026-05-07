import torch
import torch.nn as nn
import torch.optim as optim


class XorModel(nn.Module):
    def __init__(self, layers: list[int]) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def main() -> None:
    torch.manual_seed(42)

    x = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=torch.float32)
    y_true = torch.tensor([1, 0, 1, 0], dtype=torch.float32)

    layers = [2, 4, 1]
    model = XorModel(layers)
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    epochs = 10000

    for epoch in range(epochs):
        output = model(x)
        loss = loss_fn(output, y_true.view(-1, 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.6f}")

    with torch.no_grad():
        pred = (model(x) > 0.5).int()

    print(y_true.int().view(-1, 1), pred)


if __name__ == "__main__":
    main()
