# -*- coding: utf-8 -*-
"""
qa_system.py —— 主程序：解析 → 渲染 → 查询 → 答案整合（稳健版）
- 一句多问：逐个查询，最后一次性整合输出
- 模板渲染：委托 kgqa_templates.render(cid, slots)
- OpenAI：用于 1) 问题分类/抽槽位 2) 最终答案整合 3)（仅当触发）智能生成SPARQL兜底
- 稳定性：GraphDB 走 POST+超时；LLM 整合带入参裁剪、分批、超时与重试；失败自动降级
- 交互：REPL 模式（不会回答完就退出），输入 exit 退出
"""

import os, json, re, requests, time, sys, math, random
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import kgqa_templates as tpl

# ====================== 基础配置（环境变量可覆盖） ======================
GRAPHDB_URL      = os.environ.get("GRAPHDB_URL",   "http://MacBook-Pro-88.local:7200/repositories/dpkgraph")
MODEL_PARSE      = os.environ.get("MODEL_PARSE",   "gpt-4o")      # 分类/抽槽位
MODEL_ANSWER     = os.environ.get("MODEL_ANSWER",  "gpt-4o")     # 答案整合
MODEL_SPARQL     = os.environ.get("MODEL_SPARQL",  "gpt-5")      # （仅兜底时）SPARQL 生成
OPENAI_API_KEY= os.environ.get("OPENAI_API_KEY","")

HTTP_TIMEOUT     = int(os.environ.get("KG_HTTP_TIMEOUT_SEC", "20"))      # GraphDB HTTP超时（秒）
QUERY_TIMEOUT    = int(os.environ.get("KG_QUERY_TIMEOUT_MS", "20000"))   # GraphDB 评估超时（毫秒）
DEFAULT_LIMIT    = int(os.environ.get("KG_DEFAULT_LIMIT", "50"))         # 模板忘写 LIMIT 时兜底
ENABLE_GDB_RETRY = os.environ.get("KG_HTTP_RETRY", "1") == "1"           # GraphDB 请求失败是否指数退避重试

# —— LLM 整合安全阀（按需调整） —— #
OPENAI_TIMEOUT_SEC   = float(os.environ.get("OPENAI_TIMEOUT_SEC", "60"))
OPENAI_MAX_RETRIES   = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))
SUM_ROWS_LIMIT       = int(os.environ.get("SUM_ROWS_LIMIT", "200"))     # 每子问最多带多少行给整合模型
SUM_FIELD_LEN        = int(os.environ.get("SUM_FIELD_LEN", "600"))      # 单字段最大字符数
SUM_PAYLOAD_MAX_KB   = int(os.environ.get("SUM_PAYLOAD_MAX_KB", "300")) # 单次整合payload上限（KB），超过就分批
OUTPUT_MODE          = os.environ.get("OUTPUT_MODE", "text").lower()    # debug 时可设为 json

if not OPENAI_API_KEY:
    raise RuntimeError("请设置 OPENAI_API_KEY 环境变量。")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE or None,
    timeout=OPENAI_TIMEOUT_SEC,
    max_retries=OPENAI_MAX_RETRIES,
)

# ====================== 彩色日志 ======================
def log(msg: str, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    color = {"INFO":"\033[36m","OK":"\033[32m","WARN":"\033[33m","ERR":"\033[31m"}.get(level,"")
    endc = "\033[0m"
    print(f"{color}[{ts}] {level}: {msg}{endc}")
    sys.stdout.flush()

# ====================== 提示词（分类/整合） ======================
SYSTEM_PROMPT_PARSE = r"""
你是一个“抑郁症医学知识图谱问答系统”的语义解析器。
你的任务是从用户的问题中识别出一个或多个子问题，并对每个子问题输出：
1. 问题所属类别（category ID）
2. 抽取到的关键词槽位（slots）
3. 置信度
你不生成答案、不生成SPARQL，只输出一个严格的JSON数组。

一、分类类别定义
将每个子问题归类为以下之一（用数字标识）：
1. 药物适应证 → 捕捉 drug
2. 药物副作用 → 捕捉 drug
3. 药物间相互作用 → 捕捉 drug_a, drug_b
4. 食物推荐 → 捕捉 disease
5. 食物禁忌 → 捕捉 disease
6. 治疗建议 → 捕捉 disease
7. 疾病症状 → 捕捉 disease
8. 检查项目 → 捕捉 disease
9. 药物描述 → 捕捉 drug
10. 药品相关出版物 → 捕捉 drug
11. 疾病描述 → 捕捉 disease

若一个句子同时问多个内容（如“氟西汀的副作用和适应证是什么？”），请拆分成多个子问题。
若存在上下文省略（如“那副作用呢？”），可继承上一个实体。

二、槽位（slots）规则
不同类别对应不同槽位键：
* drug：单药类问题（1, 2, 9, 10）
* drug_a, drug_b：药物相互作用类（3）
* disease：疾病类问题（4, 5, 6, 7, 8, 11）
槽位值写入原文（可为中文或英文药名/疾病名），无需URI。

三、拆分规则
当用户的问题中包含并列成分或多个疑问焦点时（如“和”、“以及”、“还有”、“分别”、“同时”、“？”、“；”等），请：
* 拆成多个子问题；
* 每个子问题独立分类；
* 继承同一上下文实体（如药名相同则沿用）。

四、输出格式（严格JSON数组）
[
  {
    "cid": 2,
    "q": "文拉法辛有哪些副作用？",
    "slots": { "drug": "文拉法辛" },
    "conf": 0.95
  }
]
字段说明：
* cid：问题类别编号（1–11）
* q：子问题文本（清晰简短）
* slots：包含槽位名与捕捉到的值（如 {"drug": "氟西汀"}）
* conf：分类置信度（0~1）

五、触发线索词汇（辅助分类用）
* 适应证：适应症 / 用于治疗 / indication / 适应证
* 副作用：副作用 / 不良反应 / AE / ADR / 不适
* 相互作用：相互作用 / 合用 / 一起吃 / 同服 / 禁忌 / 交互 / 作用冲突
* 食物推荐：吃什么好 / 饮食推荐 / 适合的食物
* 食物禁忌：忌口 / 避免 / 禁食 / 不宜吃
* 护理建议：护理 / 看护 / 日常照护 / 生活建议
* 疾病症状：症状 / 表现 / 体征 / 临床表现
* 检查项目：检查 / 检验 / 筛查 / 量表 / 评估
* 药物描述：简介 / 介绍 / 说明书 / 用法 / 用量
* 出版物：文献 / 论文 / 研究 / 出版物 / PubMed

六、输出要求
1. 严格返回 纯 JSON 数组。
2. 不要解释、不加任何文本。
3. 每个元素是一个独立子问题，格式必须与上例完全一致。
4. 不要虚构药名或疾病名；如果缺失某个关键词，请不要添加。
5. 若只匹配到部分槽位（如相互作用只找到一个药），仍输出该槽位，其余留空。
6. 若一句话含多问，拆分为多个数组元素。

七、示例
输入：
氟西汀有哪些副作用和适应证？还能和洛拉西泮一起吃吗？
输出：
[
  {
    "cid": 2,
    "q": "氟西汀有哪些副作用？",
    "slots": { "drug": "氟西汀" },
    "conf": 0.95
  },
  {
    "cid": 1,
    "q": "氟西汀的适应证是什么？",
    "slots": { "drug": "氟西汀" },
    "conf": 0.93
  },
  {
    "cid": 3,
    "q": "氟西汀能和洛拉西泮一起吃吗？",
    "slots": { "drug_a": "氟西汀", "drug_b": "洛拉西泮" },
    "conf": 0.98
  }
]"""

SYSTEM_PROMPT_ANSWER = r"""
你是抑郁症知识图谱的答案整合器。你将收到若干“子问题+查询结果行”。
请针对每个子问题，用准确、医学合规的中文回答。
- 输出内容要进行润色，不要过于机械；
- 若无数据，明确说明未在知识图谱中查到相关信息；
- 列表信息（如症状/检查/食物）合并去重后列点；
- 相互作用（cid=3）仅依据“药物A的描述中确实提到药物B”的内容作答；
- 不要编造；不要超出给定数据。
- 如果查到的结果过于模糊，如“物理治疗”、“心理治疗”等，可进行适当的内容细化但绝不能瞎写。"""

# ====================== 智能 SPARQL 生成：系统提示词（最新版，含谓词别名降级） ======================
SYSTEM_PROMPT_SPARQL_GEN = r"""
你是“抑郁症医学知识图谱问答系统”的 SPARQL 生成器。
仅在“模板查询 0 行”或“最初的分类器失败”时被调用，用于一次性兜底生成查询语句。

【KG 前缀】
PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

【统一实体解析（中英双语；三段式；大小写不敏感）】
# 变量名按上下文使用 ?drug / ?drugA / ?drugB / ?dis
{
  BIND(LCASE("{{NAME}}") AS ?_name)
  OPTIONAL { ?e1 rdfs:label ?lbl . FILTER(LCASE(STR(?lbl)) = ?_name) BIND(?e1 AS ?label_hit) }
  OPTIONAL { FILTER(!BOUND(?label_hit)) ?e2 db:has_synonym ?syn .
             FILTER(LCASE(STR(?syn)) = ?_name) BIND(?e2 AS ?syn_hit) }
  OPTIONAL { FILTER(!BOUND(?label_hit) && !BOUND(?syn_hit)) ?e3 ?pp ?oo .
             BIND(STRAFTER(STR(?e3), "#") AS ?local) FILTER(LCASE(?local) = ?_name) BIND(?e3 AS ?local_hit) }
  BIND(COALESCE(?label_hit, ?syn_hit, ?local_hit) AS ?entity)
  FILTER(BOUND(?entity))
}

【类别与谓词（存在即用，优雅降级；只 SELECT）】
- 药物适应证：db:indicated_for 〓 备选 db:indication / db:indications
- 副作用：    db:has_side_effect 〓 备选 db:side_effect / db:adverse_effect
- 相互作用：  db:interaction_description 〓 备选 db:drug_interaction
- 疾病→症状： db:typical_symptom 〓 备选 db:symptom
- 疾病→检查： db:has_check 〓 备选 db:check / db:examination / db:assessment
- 治疗建议：  db:has_food_suitability / db:drugTreatmentDescription / db:has_check / db:nursingDescription
- 饮食推荐：  db:has_food_suitability 〓 备选 db:food_recommendation / db:diet_recommendation
- 饮食禁忌：  db:has_food_taboo

当主谓词不存在时，尝试上述近义谓词，用 OPTIONAL+COALESCE/UNION 组合，避免空集。

【产出要求】
1) 仅输出一段可执行的 SELECT SPARQL（不要解释、不要 Markdown、不要围栏）。
2) 文本匹配统一用 LCASE；优先完整等值（=），退化到包含（CONTAINS）。
3) 结果字段命名语义化：?indication ?side_effect ?interaction ?symptom ?check ?food_good ?food_avoid ?nursing ?treatment 等。
4) 必须带 LIMIT（若未显式指定，则默认 LIMIT 50）。
5) 禁止编造实体或硬编码具体药名/疾病名——统一从 slots / 文本中带入并按“三段式解析”绑定实体变量。
"""

# ====================== GraphDB 查询（稳健版） ======================
def _do_post(query: str, timeout: int) -> requests.Response:
    return requests.post(
        GRAPHDB_URL,
        data={"query": query},
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-GraphDB-Timeout": str(QUERY_TIMEOUT),
        },
        timeout=timeout
    )

def exec_sparql(query: str) -> Dict[str, Any]:
    # 可选指数退避（网络波动/瞬时错误时更稳）
    attempts = 4 if ENABLE_GDB_RETRY else 1
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            r = _do_post(query, HTTP_TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"GraphDB 返回错误 {r.status_code}: {r.text[:200]}")
            try:
                return r.json()
            except Exception:
                raise RuntimeError("GraphDB 返回非JSON")
        except (requests.Timeout, requests.RequestException, RuntimeError) as e:
            last_err = e
            if i < attempts - 1:
                sleep_ms = int(200 * (2 ** i) + random.randint(0, 100))
                log(f"  · GraphDB 请求失败，重试（{i+1}/{attempts-1}）{sleep_ms}ms：{e}", "WARN")
                time.sleep(sleep_ms/1000.0)
            else:
                break
    # 全部失败
    if last_err:
        raise RuntimeError(str(last_err))
    raise RuntimeError("未知 GraphDB 错误")

def extract_rows(data: Dict[str, Any]) -> List[Dict[str, str]]:
    out, head = [], data.get("head", {}).get("vars", [])
    for b in data.get("results", {}).get("bindings", []):
        row = {v: b[v]["value"] for v in head if v in b}
        if row: out.append(row)
    return out

# ====================== 语义解析（多问拆分） ======================
def parse_question(user_query: str) -> List[Dict[str, Any]]:
    log("开始语义解析与问题分类...", "INFO")
    completion = client.chat.completions.create(
        model=MODEL_PARSE,
        temperature=1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_PARSE},
            {"role": "user",   "content": user_query}
        ]
    )
    text = completion.choices[0].message.content.strip()
    try:
        units = json.loads(text)
    except Exception:
        m = re.search(r'\[\s*{.*}\s*\]', text, flags=re.S)
        if not m:
            log("× 解析器未返回有效 JSON，进入兜底生成策略。", "ERR")
            return [{"cid": None, "q": user_query.strip(), "slots": {}, "conf": 0.0, "_parse_failed": True}]
        units = json.loads(m.group(0))
    log(f"分类完成，共检测到 {len(units)} 个子问题。", "OK")
    for i,u in enumerate(units,1):
        log(f"  子问题 {i}: cid={u.get('cid')} | {u.get('q')}", "INFO")
    return units

# ====================== 模板渲染 ======================
def render_sparql(cid: Optional[int], slots: Dict[str, str]) -> str:
    if cid is None:
        return ""  # 分类失败时不走模板
    sparql = tpl.render(cid, slots)
    # 兜底 LIMIT
    if sparql.strip().lower().startswith("select") and re.search(r'\blimit\s+\d+\b', sparql, flags=re.I) is None:
        sparql += f"\nLIMIT {DEFAULT_LIMIT}"
    return sparql

# ====================== 智能 SPARQL 生成（仅兜底时调用） ======================
def _strip_code_fence(s: str) -> str:
    m = re.search(r'```(?:sparql)?\s*(.*?)```', s, flags=re.S|re.I)
    return m.group(1).strip() if m else s.strip()

def _force_limit(sparql: str, default_limit: int) -> str:
    if re.search(r'\blimit\s+\d+\b', sparql, flags=re.I) is None:
        return sparql.rstrip() + f"\nLIMIT {default_limit}"
    return sparql

def build_sparql_user_payload(unit: Dict[str, Any]) -> str:
    cid = unit.get("cid")
    q   = unit.get("q","").strip()
    slots = unit.get("slots") or {}
    payload = {"cid": cid, "question": q, "slots": slots}
    return json.dumps(payload, ensure_ascii=False, indent=2)

def llm_generate_sparql(unit: Dict[str, Any]) -> str:
    user_payload = build_sparql_user_payload(unit)
    log("触发兜底：调用 LLM 生成 SPARQL。", "WARN")
    resp = client.chat.completions.create(
        model=MODEL_SPARQL,
        temperature=1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SPARQL_GEN},
            {"role": "user",   "content": user_payload}
        ]
    )
    text = resp.choices[0].message.content or ""
    sparql = _strip_code_fence(text)
    if "select" not in sparql.lower():
        raise RuntimeError("生成器未返回 SELECT 查询。")
    sparql = _force_limit(sparql, DEFAULT_LIMIT)
    return sparql

def need_llm_fallback(item: Dict[str, Any]) -> bool:
    """
    仅在两种情况触发：
    1) 分类失败（cid=None 或 _parse_failed）
    2) 模板执行成功但 rows==0 （“无法返回结果”）
    """
    if item.get("_parse_failed"):
        return True
    if item.get("cid") is None:
        return True
    if item.get("error"):
        # GraphDB 错误默认不兜底（便于定位）；如需也兜底，可改 True
        return False
    rows = item.get("rows") or []
    return len(rows) == 0

# ====================== 主流程：逐个查询（含兜底） ======================
def run_pipeline(user_query: str) -> List[Dict[str, Any]]:
    log("------ 新一轮查询开始 ------", "INFO")
    units = parse_question(user_query)
    results: List[Dict[str, Any]] = []

    for i,u in enumerate(units, start=1):
        cid, qtext, slots = (u.get("cid", None)), u.get("q",""), (u.get("slots") or {})
        parse_failed = bool(u.get("_parse_failed"))
        log(f"[{i}/{len(units)}] 查询子问题：{qtext}", "INFO")
        item = {"cid": cid, "q": qtext, "slots": slots, "rows": []}
        if parse_failed:
            item["_parse_failed"] = True

        # 1) 先按模板走（仅在分类成功时）
        if cid is not None:
            try:
                sparql = render_sparql(cid, slots)
                item["sparql_template"] = sparql
                if sparql:
                    raw  = exec_sparql(sparql)
                    rows = extract_rows(raw)
                    item["rows"] = rows
                    log(f"  → 模板返回 {len(rows)} 条结果。", "OK")
            except Exception as e:
                item["error"] = str(e)
                log(f"  × 模板查询失败：{e}", "ERR")

        # 2) 判断是否触发兜底（严格条件）
        if need_llm_fallback(item):
            try:
                gen_sparql = llm_generate_sparql(item)
                item["sparql_fallback"] = gen_sparql
                raw2 = exec_sparql(gen_sparql)
                rows2 = extract_rows(raw2)
                item["rows"] = rows2
                log(f"  → 兜底SPARQL返回 {len(rows2)} 条结果。", "OK")
            except Exception as e2:
                item["fallback_error"] = str(e2)
                log(f"  × 兜底生成/查询失败：{e2}", "ERR")

        results.append(item)

    log("所有子问题处理完毕。", "OK")
    return results

# ====================== 本地降级整合（不依赖 LLM） ======================
def _first_sentences(s: str, max_sents=2, max_chars=400):
    if not isinstance(s, str): return s
    import re as _re
    parts = _re.split(r'(?<=[.!?。！？])\s+|\n+', s.strip())
    short = ' '.join(parts[:max_sents]).strip()
    return short if len(short) <= max_chars else (short[:max_chars] + " …")

def local_fallback_summary(units: List[Dict[str, Any]]) -> str:
    parts = []
    for u in units:
        title = u.get("q") or f"(cid={u.get('cid')})"
        if u.get("error") and not u.get("rows"):
            parts.append(f"{title}：查询失败（{u['error']}）")
            continue
        rows = u.get("rows", [])
        if not rows:
            fb = u.get("fallback_error")
            if fb:
                parts.append(f"{title}：未查询到匹配结果（兜底亦失败：{fb}）")
            else:
                parts.append(f"{title}：未查询到匹配结果")
            continue
        cols = list(rows[0].keys()) if rows else []
        if cols:
            col0 = cols[0]
            vals = []
            for r in rows[:20]:
                v = r.get(col0)
                vals.append(_first_sentences(v, 2, 200) if isinstance(v,str) else v)
            preview = "；".join([v for v in vals if v])
            more = "" if len(rows) <= 20 else f"（其余 {len(rows)-20} 条略）"
            parts.append(f"{title}：{preview}{more}")
        else:
            parts.append(f"{title}：{len(rows)} 条结果")
    return "\n".join(parts)

# ====================== LLM 整合（带裁剪/分批/降级） ======================
def _truncate_text(s: Any, limit: int) -> Any:
    if not isinstance(s, str): return s
    return s if len(s) <= limit else (s[:limit] + " …")

def _shrink_rows(rows: List[Dict[str, Any]], max_rows: int, max_field_len: int) -> List[Dict[str, Any]]:
    out = []
    for r in rows[:max_rows]:
        out.append({k: _truncate_text(v, max_field_len) for k,v in r.items()})
    return out

def _build_payload(units: List[Dict[str, Any]]) -> str:
    compact = []
    for u in units:
        compact.append({
            "cid":  u.get("cid"),
            "q":    _truncate_text(u.get("q",""), 200),
            "rows": _shrink_rows(u.get("rows",[]), SUM_ROWS_LIMIT, SUM_FIELD_LEN),
            "error": _truncate_text(u.get("error","") or "", 300),
            "sparql": u.get("sparql_template") or u.get("sparql_fallback") or ""
        })
    return json.dumps(compact, ensure_ascii=False, indent=2)

def _batch_units_by_size(units: List[Dict[str, Any]], max_kb: int) -> List[List[Dict[str, Any]]]:
    batches, cur = [], []
    limit_bytes = max(16, max_kb) * 1024
    for u in units:
        test = _build_payload(cur + [u]).encode("utf-8")
        if len(test) > limit_bytes and cur:
            batches.append(cur)
            cur = [u]
        else:
            cur = cur + [u]
    if cur: batches.append(cur)
    return batches

def llm_summarize(units_with_rows: List[Dict[str, Any]]) -> str:
    batches = _batch_units_by_size(units_with_rows, SUM_PAYLOAD_MAX_KB)
    log(f"进入答案整合阶段...（分批 {len(batches)} 批，每批≤{SUM_PAYLOAD_MAX_KB}KB）", "INFO")
    pieces = []
    for idx, batch in enumerate(batches, 1):
        payload = _build_payload(batch)
        log(f"  · 发送第 {idx}/{len(batches)} 批，payload≈{len(payload.encode('utf-8'))} bytes", "INFO")
        try:
            resp = client.chat.completions.create(
                model=MODEL_ANSWER,
                temperature=1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_ANSWER},
                    {"role": "user",   "content": payload}
                ]
            )
            text = resp.choices[0].message.content.strip()
            log(f"  · 第 {idx} 批整合完成。", "OK")
            pieces.append(text)
        except Exception as e:
            err_type = type(e).__name__
            msg = getattr(e, "message", None) or str(e)
            log(f"  × 第 {idx} 批整合失败：{err_type}: {msg}", "ERR")
            log("    → 使用本地规则对该批降级汇总。", "WARN")
            pieces.append(local_fallback_summary(batch))

    final = "\n\n".join(pieces).strip()
    if not final:
        log("整合阶段返回空内容，整体降级。", "WARN")
        return local_fallback_summary(units_with_rows)
    log("整合完成。", "OK")
    return final

# ====================== 输出（调试友好） ======================
def format_plain_results(units_with_rows, mode: str = None) -> str:
    mode = (mode or OUTPUT_MODE).lower()
    if mode == "json":
        payload = []
        for u in units_with_rows:
            payload.append({
                "cid":   u.get("cid"),
                "q":     u.get("q",""),
                "error": u.get("error"),
                "rows":  u.get("rows", []),
                "sparql_template": u.get("sparql_template"),
                "sparql_fallback": u.get("sparql_fallback"),
                "fallback_error":  u.get("fallback_error")
            })
        return json.dumps(payload, ensure_ascii=False, indent=2)
    # text 预览
    lines = []
    for idx, u in enumerate(units_with_rows, 1):
        head = f"[{idx}] {u.get('q','')}"
        if u.get("error") and not u.get("rows"):
            lines.append(head + f"\n  × 查询失败：{u['error']}")
            continue
        rows = u.get("rows", [])
        lines.append(head + f"\n  ✓ 返回 {len(rows)} 条结果")
        if u.get("sparql_template"):
            lines.append("    · 模板查询已执行")
        if u.get("sparql_fallback"):
            lines.append("    · 兜底SPARQL已执行")
        for i, r in enumerate(rows[:8], 1):
            kv = " | ".join([f"{k}={_first_sentences(v,2,200)}" for k, v in r.items()])
            lines.append(f"    {i}. {kv}")
        if len(rows) > 8:
            lines.append(f"    ……（还有 {len(rows)-8} 条）")
    return "\n".join(lines)

# ====================== 一站式调用 ======================
def answer_query(user_query: str, bypass_llm: bool = False) -> str:
    log("系统开始处理输入问题。", "INFO")
    mids = run_pipeline(user_query)
    if bypass_llm:
        log("跳过 LLM 整合，直接输出查询结果。", "WARN")
        return format_plain_results(mids)
    log("准备交给模型进行最终答案整合...", "INFO")
    return llm_summarize(mids)

# ====================== CLI：REPL ======================
if __name__ == "__main__":
    log(f"连接 GraphDB: {GRAPHDB_URL}", "INFO")
    log(f"MODEL_PARSE={MODEL_PARSE} | MODEL_SPARQL={MODEL_SPARQL} | MODEL_ANSWER={MODEL_ANSWER}", "INFO")
    log(f"整合阈值：SUM_ROWS_LIMIT={SUM_ROWS_LIMIT}, SUM_FIELD_LEN={SUM_FIELD_LEN}, MAX_PAYLOAD={SUM_PAYLOAD_MAX_KB}KB", "INFO")
    print("\n输入问题（支持多问），输入 exit 退出。可在行尾加入 --raw 跳过整合直出。\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            log("用户中断。退出。", "WARN")
            break
        if not q:
            continue
        if q.lower() == "exit":
            log("Bye.", "WARN")
            break

        bypass = False
        if q.endswith("--raw"):
            bypass = True
            q = q[:-5].strip()

        t0 = time.time()
        try:
            ans = answer_query(q, bypass_llm=bypass)
            dt = int((time.time()-t0)*1000)
            print("\n=== 输出 ===\n")
            print(ans)
            log(f"执行完成，用时 {dt} ms。", "OK")
            print()
        except Exception as e:
            log(f"执行失败：{e}", "ERR")
