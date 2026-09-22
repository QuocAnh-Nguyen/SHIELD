import json
import re
import argparse


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def parse_yes_no(text):
    """Parse a yes/no answer from free-form model output.

    Free-form models (run without the 'Please answer with yes or no.' suffix)
    may answer e.g. 'No, there is no truck', 'There is no existence of
    scissors', 'Yes, there is a car', or simply 'no'. Rules:

    1. A leading yes/affirmation token wins.
    2. A leading no/negation token wins (no, not, none, never, cannot...).
    3. Otherwise the first standalone yes/no word in the text decides.
    Returns 'yes', 'no', or None when unparseable.
    """
    s = str(text).strip().lower()
    if not s:
        return None
    if re.match(r"^(yes|yeah|yep|y)\b", s):
        return 'yes'
    if re.match(r"^(no|not|none|never|nope|n)\b", s):
        return 'no'
    m = re.search(r"\b(yes|no)\b", s)
    if m:
        return m.group(1)
    if re.search(r"\b(cannot|can't|don't|doesn't|wrong|without)\b", s):
        return 'no'
    return None


def answer_check(beaf_qna, model_answers):
    if len(beaf_qna) != len(model_answers):
        raise ValueError(
            f"answer file has {len(model_answers)} entries but the qna file has "
            f"{len(beaf_qna)} - every question must be answered in id order"
        )

    orig_pairs = {}
    unparseable = []
    for (q, a) in zip(beaf_qna, model_answers):
        assert q['id'] == a['id']
        answer = parse_yes_no(a['answer'])
        if answer is None:
            unparseable.append((q['id'], a['answer']))
            answer = 'no'

        gt = q['gt']

        if gt == 'yes' and answer == 'yes':
            q['answer'] = 'TP'
        elif gt == 'no' and answer == 'no':
            q['answer'] = 'TN'
        elif gt == 'yes' and answer == 'no':
            q['answer'] = 'FN'
        elif gt == 'no' and answer == 'yes':
            q['answer'] = 'FP'

        if q['orig_img']:
            if orig_pairs.get(q['image']) is None:
                orig_pairs[q['image']] = {}
            orig_pairs[q['image']][q['question']] = q['answer']

        total_qna = beaf_qna.copy()

    if unparseable:
        print(f"WARNING: {len(unparseable)} answers were unparseable and defaulted to 'no' "
              f"(e.g. ids {[i for i, _ in unparseable[:10]]})")

    return orig_pairs, total_qna


def metric(orig_pairs, total_qna):
    cnt = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
           'TU': 0, 'IG': 0, 'SBp': 0, 'SBn': 0, 'ID': 0}
    conv = {'TPTN': 'TU', 'FNFP': 'IG', 'TPFP': 'SBp', 'FNTN': 'SBn'}

    id_tot = 0
    for tot in total_qna:
        cnt[tot['answer']] += 1
        if not tot['orig_img']:
            name = tot['image'][:-7] + '.jpg'
            ori_ans = orig_pairs[name][tot['question']]
            if tot['removed_q']:
                if conv.get(ori_ans + tot['answer']) is not None:
                    key = conv[ori_ans + tot['answer']]
                    cnt[key] += 1
            else:
                id_tot += 1
                if ori_ans[0] != tot['answer'][0]:
                    cnt['ID'] += 1

    total = cnt['TP'] + cnt['FP'] + cnt['TN'] + cnt['FN']
    removed_total = cnt['TU'] + cnt['IG'] + cnt['SBp'] + cnt['SBn']

    acc = (cnt['TP'] + cnt['TN']) / total * 100
    precision = cnt['TP'] / (cnt['TP'] + cnt['FP']) * 100 if (cnt['TP'] + cnt['FP']) > 0 else 0
    recall = cnt['TP'] / (cnt['TP'] + cnt['FN']) * 100 if (cnt['TP'] + cnt['FN']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    tu = cnt['TU'] / removed_total * 100 if removed_total > 0 else 0
    ig = cnt['IG'] / removed_total * 100 if removed_total > 0 else 0
    sbp = cnt['SBp'] / removed_total * 100 if removed_total > 0 else 0
    sbn = cnt['SBn'] / removed_total * 100 if removed_total > 0 else 0
    id_ = cnt['ID'] / id_tot * 100 if id_tot > 0 else 0
    f1_tuid = 2 * tu * (100 - id_) / (tu + (100 - id_)) if (tu + (100 - id_)) > 0 else 0
    return acc, precision, recall, f1, tu, ig, sbp, sbn, id_, f1_tuid


def evaluate(args):
    beaf_qna = load_json(f'{args.qna_path}')
    model_answers = load_json(f'{args.model_answers}')
    orig_pairs, total_qna = answer_check(beaf_qna, model_answers)
    ACC, Precision, Recall, F1_PR, TU, IG, SBp, SBn, ID, F1_TUID = metric(orig_pairs, total_qna)

    print("========================================================")
    print("   Accuracy  |  Precision  |    Recall   |    F1(P,R) ")
    print("--------------------------------------------------------")
    print(f"    {ACC:.2f}    |    {Precision:.2f}    |    {Recall:.2f}    |    {F1_PR:.2f}")
    print("=========================================================")
    print("   TU   |   IG   |   SB+  |   SB-  |   ID   | F1(TU,ID)")
    print("---------------------------------------------------------")
    print(f" {TU:.2f}  |  {IG:.2f}  |  {SBp:.2f} |  {SBn:.2f} |  {ID:.2f}  |   {F1_TUID:.2f}")
    print("=========================================================")

    if args.save_path:
        detail = []
        for q in total_qna:
            detail.append({
                'id': q['id'],
                'image': q['image'],
                'question': q['question'],
                'gt': q['gt'],
                'orig_img': q['orig_img'],
                'removed_q': q['removed_q'],
                'verdict': q['answer'],
                'raw_answer': next(a['answer'] for a in model_answers if a['id'] == q['id']),
            })
        with open(args.save_path, 'w') as f:
            json.dump(detail, f, indent=2, ensure_ascii=False)
        print(f"Saved per-question verdicts to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qna-path", type=str, default="./beaf_qna.json")
    parser.add_argument("--model-answers", type=str, default="./answer_gpt4o.json")
    parser.add_argument("--save-path", type=str, default="",
                        help="save per-question TP/FP/TN/FN verdicts to this json file")
    args = parser.parse_args()

    evaluate(args)
