

def get_common_items(*lists):
    merged_lists = []
    [merged_lists.extend(i) for i in lists]
    selected = {}
    for (score, feature) in merged_lists:
        if feature in selected:
            selected[feature] += 1
        else:
            selected[feature] = 1

    ranked_features = list(reversed(sorted(selected.items(), key=lambda kv: kv[1])))
    print(*ranked_features, sep="\n")
    return ranked_features


def get_total_items(*lists):
    merged_lists = []
    [merged_lists.extend(i) for i in lists]
    selected = {}
    for (feature, score) in merged_lists:
        if feature in selected:
            selected[feature] += score
        else:
            selected[feature] = score

    ranked_features = reversed(sorted(selected.items(), key=lambda kv: kv[1]))
    print(*ranked_features, sep="\n")
    return ranked_features
