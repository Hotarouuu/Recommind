from collections import defaultdict
import torch

def evaluate_batch_precision_recall(df, model, k=10, threshold=4, device="cpu"):
    model.eval()
    user_item_scores = defaultdict(list)

    with torch.no_grad():
        for X, y in df:
            X, y = X.to(device), y.to(device)
            item_ids = X[:, 0]
            user_ids = X[:, 1]
            scores_real = y

            scores_pred = model(X).squeeze()

            for uid, iid, pred, real in zip(user_ids, item_ids, scores_pred, scores_real):
                user_item_scores[uid.item()].append((iid.item(), pred.item(), real.item()))

    precisions, recalls = [], []

    for uid, interactions in user_item_scores.items():
        # Ordena por score predito (desc)
        topk = sorted(interactions, key=lambda x: x[1], reverse=True)[:k]
        topk_items = {i[0] for i in topk}
        relevant_items = {i[0] for i in interactions if i[2] >= threshold}

        if not relevant_items:
            continue

        hits = topk_items & relevant_items
        precisions.append(len(hits) / k)
        recalls.append(len(hits) / len(relevant_items))

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    f_score = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if avg_precision + avg_recall > 0
        else 0.0
    )

    return avg_precision, avg_recall, f_score, user_item_scores