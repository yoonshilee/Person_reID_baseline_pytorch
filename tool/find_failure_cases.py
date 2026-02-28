import argparse
import os

import numpy as np
import scipy.io
import torch
from torchvision import datasets


def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu().numpy()

    index = np.argsort(score)[::-1]

    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    index = index[np.isin(index, junk_index, invert=True)]
    return index


def main():
    parser = argparse.ArgumentParser(description="Find failure cases from pytorch_result.mat")
    parser.add_argument("--mat", default="pytorch_result.mat", type=str, help="Path to pytorch_result.mat")
    parser.add_argument("--test_dir", default="./data/DukeMTMC-reID/pytorch", type=str, help="Dataset pytorch folder")
    parser.add_argument("--topk", default=10, type=int, help="Top-K for stats")
    parser.add_argument("--max_candidates", default=20, type=int, help="Max candidates to print")
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"], help="Device")
    args = parser.parse_args()

    if not os.path.isfile(args.mat):
        raise FileNotFoundError(f"[ERROR] mat file not found: {args.mat}")

    data_dir = args.test_dir
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"[ERROR] test_dir not found: {data_dir}")

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ["gallery", "query"]
    }

    result = scipy.io.loadmat(args.mat)
    query_feature = torch.FloatTensor(result["query_f"])
    query_cam = result["query_cam"][0]
    query_label = result["query_label"][0]
    gallery_feature = torch.FloatTensor(result["gallery_f"])
    gallery_cam = result["gallery_cam"][0]
    gallery_label = result["gallery_label"][0]

    device = torch.device(args.device)
    query_feature = query_feature.to(device)
    gallery_feature = gallery_feature.to(device)

    failures = []
    for i in range(len(query_label)):
        index = sort_img(
            query_feature[i],
            query_label[i],
            query_cam[i],
            gallery_feature,
            gallery_label,
            gallery_cam,
        )

        topk = min(args.topk, len(index))
        topk_labels = gallery_label[index[:topk]]
        topk_pos = int(np.sum(topk_labels == query_label[i]))

        pos_rows = np.where(gallery_label[index] == query_label[i])[0]
        if len(pos_rows) == 0:
            continue
        first_pos = int(pos_rows[0])

        is_rank1_correct = gallery_label[index[0]] == query_label[i]
        if (not is_rank1_correct) or (first_pos >= 5) or (topk_pos <= 1):
            query_path, _ = image_datasets["query"].imgs[i]
            top1_path, _ = image_datasets["gallery"].imgs[index[0]]
            failures.append(
                {
                    "i": i,
                    "first_pos": first_pos,
                    "topk_pos": topk_pos,
                    "query_label": int(query_label[i]),
                    "query_cam": int(query_cam[i]),
                    "query_path": query_path,
                    "top1_path": top1_path,
                }
            )

    failures.sort(key=lambda x: (x["first_pos"], -x["topk_pos"]), reverse=True)

    print(f"[INFO] total queries: {len(query_label)}")
    print(f"[INFO] failure-like candidates: {len(failures)}")
    print(f"[INFO] showing up to {min(args.max_candidates, len(failures))} candidates")

    for c in failures[: args.max_candidates]:
        print(
            f"[CAND] query_index={c['i']} label={c['query_label']} cam={c['query_cam']} "
            f"first_positive_rank={c['first_pos'] + 1} top{args.topk}_positives={c['topk_pos']}"
        )
        print(f"       query: {c['query_path']}")
        print(f"       top1 : {c['top1_path']}")


if __name__ == "__main__":
    main()
