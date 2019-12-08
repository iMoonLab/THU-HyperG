# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from thumoon.generation import gen_knn_hg, gen_grid_neigh_hg, concat_multi_hg, fuse_mutli_sub_hg
from thumoon.learning import trans_infer
from thumoon.utils import iou_socre, gather_patch_ft, print_log

from examples.data_helper import load_myocardium


def postprocess(ary):
    out_img = sitk.GetImageFromArray(ary)

    cnt_filter = sitk.ConnectedComponentImageFilter()
    out_img = cnt_filter.Execute(out_img)
    rlb_filter = sitk.RelabelComponentImageFilter()
    out_img = rlb_filter.Execute(out_img)
    bin_filter = sitk.BinaryThresholdImageFilter()
    out_img = bin_filter.Execute(out_img, 1, 1, 1, 0)

    return sitk.GetArrayFromImage(out_img).astype(np.int)


def main():
    print_log("loading data")
    X_train, X_test, y_train, y_test = load_myocardium([3])
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train.reshape(-1), -1 * np.ones_like(y_test).reshape(-1)])

    print_log("generating hypergraph")
    # grid hg
    grid_hg_s = [gen_grid_neigh_hg(X[i].shape) for i in range(X.shape[0])]
    grid_hg = fuse_mutli_sub_hg(grid_hg_s)
    # knn hg
    X_patch_ft = np.stack([gather_patch_ft(X[i][:, :, np.newaxis], (5, 5)) for i in range(X.shape[0])])
    knn_hg = gen_knn_hg(X_patch_ft.reshape(-1, X_patch_ft.shape[-1]), n_neighbors=7)
    # concatfeat hg
    concat_hg = concat_multi_hg([grid_hg, knn_hg])

    print_log("learning on hypergraph")
    y_predict = trans_infer(concat_hg, y, 100)
    print_log("iou: {}".format(iou_socre(y_predict, y_test.reshape(-1))))

    print_log("postprocessing")
    y_postpro = postprocess(y_predict.reshape(y_test.shape))
    print_log("iou(postprocess): {}".format(iou_socre(y_postpro.reshape(-1), y_test.reshape(-1))))

    # visualize
    print_log("visualizing")
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(X_test.squeeze())
    ax1.title.set_text('X_test')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(y_test.squeeze())
    ax2.title.set_text('y_test')
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(y_predict.reshape(y_test.squeeze().shape))
    ax3.title.set_text('y_predict')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(y_postpro.reshape(y_test.squeeze().shape))
    ax4.title.set_text('y_postpro')
    ax4.axis('off')

    plt.show()





if __name__ == "__main__":
    main()
