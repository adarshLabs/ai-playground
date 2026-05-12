import torch


class CustomRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0

        return grad_input


class CustomSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        return x**2

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors

        return grad_output * 2 * input


def main():
    x = torch.tensor([-2.0, -1.0, 0.0, 2.0, 3.0], requires_grad=True)
    y = CustomRelu.apply(x)
    loss = y.sum()
    loss.backward()

    print("ReLU forward:", y)
    print("ReLU gradient:", x.grad)

    square_input = torch.tensor([-2.0, -1.0, 0.0, 2.0, 3.0], requires_grad=True)
    z = CustomSquare.apply(square_input)
    square_loss = z.sum()
    square_loss.backward()

    print("Square forward:", z)
    print("Square gradient:", square_input.grad)


if __name__ == "__main__":
    main()
