import numpy as np
from matplotlib import pyplot as plt

def edit_axes(row, index):
    for col in row:
        col.axis("off")
    if index == 0:
        row[0].set_title("Input")
        row[1].set_title("GT")
        row[2].set_title("Trimap")
        if len(row) > 3: row[3].set_title("Predicted")
        if len(row) > 3: row[4].set_title("Predicted Alpha")
        if len(row) > 3: row[5].set_title("Trimap Predicted")


def plot_samples(images, images_gt, images_pred=None, trimap_pred=None, n=8, title=None, show=True):
    nrows = min(n, len(images))
    ncols = 6 if images_pred is not None and trimap_pred is not None else 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 15))

    for i, row in enumerate(ax):
        row[0].imshow(images[i])
        row[1].imshow(images_gt["rgba"][i][:,:,:4])
        row[2].imshow(images_gt["trimap"][i], cmap="gray", vmin=0, vmax=2)
        if ncols > 3: row[3].imshow(images_pred[i])
        if ncols > 3: row[4].imshow(images_pred[i][:,:,3:4], cmap="gray")
        if ncols > 3: row[5].imshow(trimap_pred[i], cmap="gray")
        edit_axes(row, i)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        return fig


def plot_samples_without_trimap(images, images_gt, images_pred=None, n=8, title=None, show=True):
    nrows = min(n, len(images))
    ncols = 4 if images_pred is not None else 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 15))

    for i, row in enumerate(ax):
        row[0].imshow(images[i])
        row[1].imshow(images_gt[i])
        if ncols > 3: row[2].imshow(images_pred[i])
        if ncols > 3: row[3].imshow(images_pred[i][:,:,3:4], cmap="gray")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        return fig


# image, GT, Predicted, RGB Diff, Alpha GT, Alpha Predicted, Alpha Diff
def evaluation_plt(images, images_gt=None, images_pred=None,
                   trimap_gt=None, trimap_pred=None,
                   rgb_max_diff_gt=None, rgb_max_diff_pred=None,
                   n=8, title=None, show=True):
    cols = ["Input"]
    if images_gt is not None: cols.append("GT")
    if images_pred is not None: cols.append("Predicted")
    if images_gt is not None and images_pred is not None: cols.append("Pred-GT Diff")
    if images_gt is not None and images_pred is not None: cols.append("Input-GT Diff")
    if images_gt is not None: cols.append("Alpha GT")
    if images_pred is not None: cols.append("Alpha Predicted")
    if images_gt is not None and images_pred is not None: cols.append("Alpha Diff")
    if trimap_gt is not None: cols.append("Trimap GT")
    if trimap_pred is not None: cols.append("Trimap Predicted")
    if rgb_max_diff_gt is not None: cols.append("RGB Deviation GT")
    if rgb_max_diff_pred is not None: cols.append("RGB Deviation Predicted")
    if rgb_max_diff_gt is not None and rgb_max_diff_pred is not None: cols.append("RGB Deviation Diff")

    nrows = min(n, len(images))
    ncols = len(cols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2*nrows, 4*nrows))

    for i, row in enumerate(ax):
        for col, name in zip(row, cols):
            if name == "Input":
                col.imshow(images[i])
            elif name == "GT":
                col.imshow(images_gt[i][:,:,:4])
            elif name == "Predicted":
                col.imshow(images_pred[i])
            elif name == "Pred-GT Diff":
                diff = images_pred[i][:,:,:3] - images_gt[i][:,:,:3]
                col.imshow(0.5 + (diff * images_pred[i][:,:,3:4]))
            elif name == "Input-GT Diff":
                diff = images[i][:,:,:3] - images_gt[i][:,:,:3]
                col.imshow(0.5 + (diff * images_gt[i][:,:,3:4]))
            elif name == "Alpha GT":
                col.imshow(images_gt[i][:,:,3:4], cmap="gray")
            elif name == "Alpha Predicted":
                col.imshow(images_pred[i][:,:,3:4], cmap="gray")
            elif name == "Alpha Diff":
                col.imshow(np.abs(images_pred[i][:,:,3] - images_gt[i][:,:,3]), cmap="gray")
            elif name == "Trimap GT":
                col.imshow(trimap_gt[i][..., 0], cmap="gray", vmin=0, vmax=2)
            elif name == "Trimap Predicted":
                col.imshow(np.argmax(trimap_pred[i], axis=-1, keepdims=True), cmap="gray", vmin=0, vmax=2)
            elif name == "RGB Deviation GT":
                col.imshow(rgb_max_diff_gt[i], cmap="gray", vmin=0, vmax=1)
            elif name == "RGB Deviation Predicted":
                col.imshow(rgb_max_diff_pred[i], cmap="gray", vmin=0, vmax=1)
            elif name == "RGB Deviation Diff":
                col.imshow(np.abs(rgb_max_diff_pred[i] - rgb_max_diff_gt[i]), cmap="gray", vmin=0, vmax=1)

            col.axis("off")
            if i == 0:
                col.set_title(name)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        return fig


def model_comparison(images, images_gt=None, images_pred=None,
                     trimap_gt=None, trimap_pred=None,
                     rgb_max_diff_gt=None, rgb_max_diff_pred=None,
                     n=8, title=None, model_names=None, show=True):

    rows = ["Alpha Diff", "Alpha Predicted", "Predicted", "Pred-GT Diff"]

    nrows = 4
    ncols = len(images_pred) + 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows), squeeze=False, )

    diff = images[0][:, :, :3] - images_gt[0][:, :, :3]

    for j, (row, name) in enumerate(zip(ax, rows)):
        for i, col in enumerate(row):
            col.axis("off")
            if j == 0 and i == 0: col.imshow(images[0], aspect='auto')
            if j == 1 and i == 0: col.imshow(images_gt[0][:, :, 3:4], cmap="gray", aspect='auto')
            if j == 2 and i == 0: col.imshow(images_gt[0][:, :, :4], aspect='auto')
            if j == 3 and i == 0: col.imshow(0.5 + (diff * images_gt[0][:, :, 3:4]), aspect='auto')
            if i == 0: continue
            i = i -1

            if j == 0 and model_names is not None:
                col.set_title(model_names[i], fontsize=40)

            if name == "Predicted":
                col.imshow(images_pred[i], aspect='auto')
            elif name == "Pred-GT Diff":
                diff = images_pred[i][:,:,:3] - images_gt[0][:,:,:3]
                col.imshow(0.5 + (diff * images_pred[i][:,:,3:4]), aspect='auto')
            elif name == "Alpha Predicted":
                col.imshow(images_pred[i][:,:,3:4], cmap="gray", aspect='auto')
            elif name == "Alpha Diff":
                col.imshow(np.abs(images_pred[i][:,:,3] - images_gt[0][:,:,3]), cmap="gray", aspect='auto')


    if title is not None:
        fig.suptitle(title)

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0.0)

    if show:
        plt.show()

    return fig
