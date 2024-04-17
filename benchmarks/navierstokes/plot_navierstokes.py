import torch
import matplotlib.pyplot as plt
from continuiti.benchmarks import NavierStokes
from continuiti.operators import FourierNeuralOperator

# FFT not available on MPS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ns = NavierStokes()

operator = FourierNeuralOperator(
    ns.train_dataset.shapes,
    width=32,
    depth=4,
    device=device,
)

operator.load(
    "mlruns/949079562496325102/ae8779c648d941caa0304b2e0a5c4ee4/artifacts/final.pt"
)
operator.eval()

# Compute train loss
loss_fn = ns.losses[0]


def compute_loss(dataset):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    avg_loss = 0
    max_loss, min_loss = 0, 1e10
    max_i, min_i = 0, 0
    for i, xuyv in enumerate(train_loader):
        x, u, y, v = [t.to(device) for t in xuyv]
        loss = loss_fn(operator, x, u, y, v)
        avg_loss += loss.detach()
        if loss > max_loss:
            max_loss = loss
            max_i = i
        if loss < min_loss:
            min_loss = loss
            min_i = i
    avg_loss = avg_loss / len(train_loader)
    return avg_loss, max_loss, max_i, min_loss, min_i


loss_train, max_loss, max_i_train, min_loss, min_i_train = compute_loss(
    ns.train_dataset
)
print(f"rel. error train = {loss_train:.4e}")
print(f"min loss = {min_loss:.4e} at index {min_i_train}")
print(f"max loss = {max_loss:.4e} at index {max_i_train}")

# Compute test loss
loss_test, max_loss, max_i_test, min_loss, min_i_test = compute_loss(ns.test_dataset)
print(f"rel. error test =  {loss_test:.4e}")
print(f"min loss = {min_loss:.4e} at index {min_i_test}")
print(f"max loss = {max_loss:.4e} at index {max_i_test}")


# Plot
def plot_sample(split, sample):
    dataset = ns.train_dataset if split == "train" else ns.test_dataset
    x, u, y, v = [t.to(device) for t in dataset[sample : sample + 1]]
    v_pred = operator(x, u, y)
    v = v.reshape(1, 64, 64, 10, 1).cpu()
    v_pred = v_pred.reshape(1, 64, 64, 10, 1).detach().cpu()

    fig, axs = plt.subplots(10, 3, figsize=(4, 16))

    axs[0][0].set_title("Truth")
    axs[0][1].set_title("Prediction")
    axs[0][2].set_title("Error")
    for t in range(10):
        axs[t][0].imshow(v[0, :, :, t, 0], cmap="jet")
        axs[t][1].imshow(v_pred[0, :, :, t, 0], cmap="jet")
        e = v - v_pred
        elim = max(abs(e.min()), abs(e.max()))
        im = axs[t][2].imshow(e[0, :, :, t, 0], cmap="jet", clim=(-elim, elim))
        fig.colorbar(im, ax=axs[t][2])
        axs[t][0].axis("off")
        axs[t][1].axis("off")
        axs[t][2].axis("off")

    plt.tight_layout()
    plt.savefig(f"navierstokes/ns_{split}_{sample}.png", dpi=500)


plot_sample("train", min_i_train)
plot_sample("train", max_i_train)
plot_sample("test", min_i_test)
plot_sample("test", max_i_test)
