import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
print(x.grad_fn)

y = x**2
print(y.grad_fn)

z = 3 * y + 1
print(z.grad_fn)

loss = z.sum()
print(loss.grad_fn)

loss.backward()
print(x.grad)
