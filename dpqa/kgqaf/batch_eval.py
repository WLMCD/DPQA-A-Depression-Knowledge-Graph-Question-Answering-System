# -*- coding: utf-8 -*-
"""
batch_eval.py — KGQA 批量测试与评测（修正版）
- 读取 JSONL 测试集（每行需含：cid / question / expected_terms / slots / expected_text）
- 逐条调用 qa_test.run_pipeline(question)（直接取 rows，不走答案整合 LLM）
- 抽取预测项（按 cid → 对应列名/字段），与 expected_terms 做集合重合评估
- 支持策略：全局阈值 + per-cid 覆盖（阈值、min_hits、关键词等）
"""
import os, sys, json, time, argparse, statistics, unicodedata, re
from typing import Dict, Any, List, Tuple, Set


import qa_test as system

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="测试集 JSONL 路径（含 question / expected_terms / cid）")
    ap.add_argument("--policy", required=True, help="评测规则 JSON 路径（kgqa_eval_policy.json）")
    ap.add_argument("--limit", type=int, default=0, help="仅测试前 N 条（调试用）")
    ap.add_argument("--pretty", action="store_true", help="每 10 条打印一次简报")
    ap.add_argument("--timeout", type=float, default=None, help="覆盖 OPENAI_TIMEOUT_SEC（可更小以防阻塞）")
    ap.add_argument("--export_detail", default="", help="如指定，则导出逐题明细 JSON（含命中、耗时、错误等）")
    ap.add_argument("--graphdb", default="", help="覆盖 GRAPHDB_URL（可快速切仓）")
    return ap.parse_args()

# ---------------- 评测策略与文本归一化 ----------------
_CN_PUNCT = "，。、；：！？（）【】《》“”‘’—・·、"

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    # 去 HTML 标签（如 <sub>、<sup>）
    s = re.sub(r"<[^>]+>", " ", s)
    # 去中文/英文标点
    s = re.sub(rf"[{re.escape(_CN_PUNCT)}]", " ", s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_terms(s: str) -> List[str]:
    s = normalize_text(s)
    if not s: return []
    parts = re.split(r"\s+", s)
    return [p for p in parts if p]

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    i = len(a & b)
    u = len(a | b)
    return i / max(1, u)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- 结果抽取规则 ----------------
CID_COLS = {
    1: ["indication"],                # 药物适应证
    2: ["side_effect_text"],          # 药物副作用
    3: ["desc", "interaction"],       # 相互作用
    4: ["food_label", "food", "food_good"],
    5: ["food_label", "food", "food_avoid"],
    6: ["food_suitability", "drug_treatments_desc", "checks", "nursing_desc", "nursing", "treatment"],
    7: ["symptom"],
    8: ["check_item", "check"],
    9: ["fact"],                      # 药物描述
    10:["pmid","pubmed_id"],          # 出版物
    11:["description"],               # 疾病描述
}

def collect_predicted_terms(rows: List[Dict[str, Any]], cid: int) -> List[str]:
    cols = CID_COLS.get(cid, [])
    outs: List[str] = []
    for r in rows:
        values = []
        for c in cols:
            if c in r and isinstance(r[c], str):
                values.append(r[c])
        if not values:
            values = [v for v in r.values() if isinstance(v, str)]
        for v in values:
            parts = [p.strip() for p in re.split(r"\|", v) if p.strip()]
            outs.extend(parts)
    return outs

# ---------------- 命中判定 ----------------
def eval_one(record: Dict[str, Any], units: List[Dict[str, Any]], policy: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    cid = int(record.get("cid"))
    expected_terms = [str(x) for x in (record.get("expected_terms") or [])]
    candidates = [u for u in units if u.get("cid") == cid]
    target = None
    if candidates:
        target = sorted(candidates, key=lambda u: len(u.get("rows") or []), reverse=True)[0]
    else:
        return False, {"reason": "cid_not_found", "pred_terms": [], "rows": []}

    rows = target.get("rows") or []
    if not rows:
        return False, {"reason": "no_rows", "pred_terms": [], "rows": []}

    pred_items = collect_predicted_terms(rows, cid)

    exp_set, pred_set = set(), set()
    for t in expected_terms: exp_set.update(split_terms(t))
    for t in pred_items: pred_set.update(split_terms(t))

    mcfg = policy.get("match_strategy", {})
    default_thresh = float(mcfg.get("threshold", 0.5))
    ocfg_root = policy.get("per_cid_overrides", {})
    ocfg = ocfg_root.get(str(cid)) or ocfg_root.get(cid) or {}
    cid_thresh = float(ocfg.get("threshold", default_thresh))
    ok_default = jaccard(exp_set, pred_set) >= cid_thresh

    # ---- cid=2：副作用
    if cid == 2:
        min_hits = int(ocfg.get("min_hits", 0)) if isinstance(ocfg.get("min_hits", 0), (int, float)) else 0
        if min_hits > 0:
            ok2 = len(exp_set & pred_set) >= min_hits
            return ok2, {"reason": "ok" if ok2 else "min_hits_not_met",
                         "pred_terms": list(pred_set), "rows": rows}

    # ---- cid=3：相互作用
    if cid == 3:
        drug_b = (record.get("slots") or {}).get("drug_b") or ""
        must_name = normalize_text(drug_b)
        norm_pred_all = " ".join(split_terms(" ".join(pred_items)))
        has_b = bool(must_name) and (must_name in norm_pred_all)

        kw = ocfg.get("keywords", ['contraindicated','increase','decrease','serotonin','monitor','监测','相互作用','避免','增强','降低'])
        kw_hits = sum(1 for k in kw if k in norm_pred_all)

        sub_ok = False
        substr_len = int(ocfg.get("substring_min_len", 0)) if isinstance(ocfg.get("substring_min_len", 0), (int, float)) else 0
        exp_text = record.get("expected_text") or ""
        if substr_len > 0 and exp_text:
            exp_norm = normalize_text(exp_text)
            for i in range(0, max(0, len(exp_norm) - substr_len + 1)):
                if exp_norm[i:i+substr_len] in norm_pred_all:
                    sub_ok = True
                    break

        ok = has_b and (kw_hits >= 2 or sub_ok or jaccard(exp_set, pred_set) >= cid_thresh)
        return ok, {"reason": "ok" if ok else "rule_3_not_met",
                    "pred_terms": list(pred_set), "rows": rows}

    # ---- cid=9：药物描述
    if cid == 9:
        min_hits9 = int(ocfg.get("min_hits", 0)) if isinstance(ocfg.get("min_hits", 0), (int, float)) else 0
        if min_hits9 > 0:
            ok9 = len(exp_set & pred_set) >= min_hits9
            return ok9, {"reason": "ok" if ok9 else "min_hits_not_met",
                         "pred_terms": list(pred_set), "rows": rows}
        return ok_default, {"reason": "ok" if ok_default else "below_threshold",
                            "pred_terms": list(pred_set), "rows": rows}

    # ---- cid=10：PMID 
    if cid == 10:
        ok = len(exp_set & pred_set) >= 1
        return ok, {"reason": "ok" if ok else "no_pmid_hit",
                    "pred_terms": list(pred_set), "rows": rows}

    # 默认
    return ok_default, {"reason": "ok" if ok_default else "below_threshold",
                        "pred_terms": list(pred_set), "rows": rows}

# ---------------- 主流程 ----------------
def main():
    args = parse_args()
    if args.graphdb:
        os.environ["GRAPHDB_URL"] = args.graphdb
    if args.timeout is not None:
        os.environ["OPENAI_TIMEOUT_SEC"] = str(args.timeout)

    dataset = load_jsonl(args.input)
    policy = load_json(args.policy)
    if args.limit and args.limit > 0:
        dataset = dataset[:args.limit]

    total = len(dataset)
    print(f"[INFO] 加载测试样本：{total} 条")

    per_cid_total, per_cid_ok, latencies, details = {}, {}, [], []
    for idx, rec in enumerate(dataset, 1):
        q = rec.get("question") or rec.get("q") or ""
        cid = int(rec.get("cid"))
        t0 = time.time()
        try:
            units = system.run_pipeline(q)
            ok, info = eval_one(rec, units, policy)
            dt = (time.time() - t0) * 1000.0
            latencies.append(dt)
            per_cid_total[cid] = per_cid_total.get(cid, 0) + 1
            per_cid_ok[cid] = per_cid_ok.get(cid, 0) + (1 if ok else 0)
            details.append({
                "idx": idx, "cid": cid, "question": q,
                "ok": ok, "reason": info.get("reason"),
                "latency_ms": round(dt, 2),
                "pred_terms": info.get("pred_terms"),
                "expected_terms": rec.get("expected_terms"),
                "rows_preview": info.get("rows")[:3]
            })
            if args.pretty and (idx % 10 == 0):
                print(f"[{idx}/{total}] cid={cid}, ok={ok}, latency={dt:.1f}ms")
        except Exception as e:
            dt = (time.time() - t0) * 1000.0
            latencies.append(dt)
            per_cid_total[cid] = per_cid_total.get(cid, 0) + 1
            details.append({
                "idx": idx, "cid": cid, "question": q,
                "ok": False, "reason": f"exception: {e}",
                "latency_ms": round(dt, 2),
                "pred_terms": [], "expected_terms": rec.get("expected_terms"),
                "rows_preview": []
            })
            print(f"[ERR] {idx}/{total} 执行异常：{e}")

    overall_ok = sum(1 for d in details if d["ok"])
    overall_acc = overall_ok / max(1, total)
    mean_ms = statistics.mean(latencies) if latencies else 0.0
    p50 = statistics.median(latencies) if latencies else 0.0
    p90 = statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else p50
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else p90

    print("\n=== 评测结果 ===")
    print(f"总题数：{total}")
    print(f"总体准确率：{overall_acc:.3f}  ({overall_ok}/{total})")
    print(f"平均耗时：{mean_ms:.1f} ms   p50：{p50:.1f} ms   p90：{p90:.1f} ms   p95：{p95:.1f} ms\n")

    print("分CID准确率：")
    for c in sorted(per_cid_total.keys()):
        ok = per_cid_ok.get(c, 0)
        tot = per_cid_total[c]
        print(f"  cid={c:<2d} : {ok:>3d}/{tot:<3d} = {ok/max(1,tot):.3f}")

    if args.export_detail:
        with open(args.export_detail, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total": total,
                    "overall_acc": overall_acc,
                    "mean_ms": mean_ms, "p50": p50, "p90": p90, "p95": p95
                },
                "per_cid_total": per_cid_total,
                "per_cid_ok": per_cid_ok,
                "details": details
            }, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 明细已导出：{args.export_detail}")

if __name__ == "__main__":
    main()
