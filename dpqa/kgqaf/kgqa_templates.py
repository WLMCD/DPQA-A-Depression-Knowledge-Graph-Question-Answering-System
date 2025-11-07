# -*- coding: utf-8 -*-
"""
kgqa_templates.py
—— 知识图谱问答系统的 SPARQL 模版与渲染工具
"""
from typing import Dict

# ==============
# 模版定义（cid → SPARQL字符串）
# ==============
TEMPLATES: Dict[int, str] = {
    # 1. 药物适应证
    1: r'''PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?indication WHERE {
  { SELECT ?drug WHERE {
      BIND(LCASE("{{DRUG}}") AS ?_dname)
      OPTIONAL { ?d1 rdfs:label ?lbl . FILTER(LCASE(STR(?lbl)) = ?_dname) BIND(?d1 AS ?label_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit)) ?d2 db:has_synonym ?syn .
                 FILTER(LCASE(STR(?syn)) = ?_dname) BIND(?d2 AS ?syn_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit) && !BOUND(?syn_hit)) ?d3 ?p ?o .
                 BIND(STRAFTER(STR(?d3), "#") AS ?local) FILTER(LCASE(?local) = ?_dname) BIND(?d3 AS ?local_hit) }
      BIND(COALESCE(?label_hit, ?syn_hit, ?local_hit) AS ?drug)
    } LIMIT 1 }
  ?drug db:indicated_for ?txt .
  BIND(STR(?txt) AS ?indication)
}
ORDER BY LCASE(?indication)''',

    # 2. 药物副作用
    2: r'''PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?side_effect_text WHERE {
  { SELECT ?drug WHERE {
      BIND(LCASE("{{DRUG}}") AS ?_dname)
      OPTIONAL { ?d1 rdfs:label ?lbl . FILTER(LCASE(STR(?lbl)) = ?_dname) BIND(?d1 AS ?label_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit)) ?d2 db:has_synonym ?syn .
                 FILTER(LCASE(STR(?syn)) = ?_dname) BIND(?d2 AS ?syn_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit) && !BOUND(?syn_hit)) ?d3 ?p ?o .
                 BIND(STRAFTER(STR(?d3), "#") AS ?local) FILTER(LCASE(?local) = ?_dname) BIND(?d3 AS ?local_hit) }
      BIND(COALESCE(?label_hit, ?syn_hit, ?local_hit) AS ?drug)
    } LIMIT 1 }
  ?drug db:has_side_effect ?se .
  BIND(STR(?se) AS ?side_effect_text)
}''',

    # 3. 药物相互作用（只读A描述，先label命中B，失败再synonym；两步不同时进行）
    3: r'''PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT
  (COALESCE(STR(?lblA), STRAFTER(STR(?drugA), "#")) AS ?drugA_label)
  (COALESCE(STR(?lblB), STRAFTER(STR(?drugB), "#")) AS ?drugB_label)
  ?desc
WHERE {
  # 药物A
  { SELECT ?drugA WHERE {
      BIND(LCASE("{{DRUG_A}}") AS ?_nameA)
      OPTIONAL { ?d1 rdfs:label ?la . FILTER(LCASE(STR(?la)) = ?_nameA) BIND(?d1 AS ?label_hitA) }
      OPTIONAL { FILTER(!BOUND(?label_hitA)) ?d2 db:has_synonym ?sa .
                 FILTER(LCASE(STR(?sa)) = ?_nameA) BIND(?d2 AS ?syn_hitA) }
      OPTIONAL { FILTER(!BOUND(?label_hitA) && !BOUND(?syn_hitA)) ?d3 ?p ?o .
                 BIND(STRAFTER(STR(?d3), "#") AS ?ua) FILTER(LCASE(?ua) = ?_nameA) BIND(?d3 AS ?local_hitA) }
      BIND(COALESCE(?label_hitA, ?syn_hitA, ?local_hitA) AS ?drugA)
    } LIMIT 1 }
  OPTIONAL { ?drugA rdfs:label ?lblA }

  # 药物B（仅用于匹配字符串）
  { SELECT ?drugB WHERE {
      BIND(LCASE("{{DRUG_B}}") AS ?_nameB)
      OPTIONAL { ?e1 rdfs:label ?lb . FILTER(LCASE(STR(?lb)) = ?_nameB) BIND(?e1 AS ?label_hitB) }
      OPTIONAL { FILTER(!BOUND(?label_hitB)) ?e2 db:has_synonym ?sb .
                 FILTER(LCASE(STR(?sb)) = ?_nameB) BIND(?e2 AS ?syn_hitB) }
      OPTIONAL { FILTER(!BOUND(?label_hitB) && !BOUND(?syn_hitB)) ?e3 ?p2 ?o2 .
                 BIND(STRAFTER(STR(?e3), "#") AS ?ub) FILTER(LCASE(?ub) = ?_nameB) BIND(?e3 AS ?local_hitB) }
      BIND(COALESCE(?label_hitB, ?syn_hitB, ?local_hitB) AS ?drugB)
    } LIMIT 1 }
  OPTIONAL { ?drugB rdfs:label ?lblB }

  # 只查A的交互描述
  ?drugA db:interaction_description ?d1 .

  { OPTIONAL { ?drugB rdfs:label ?bLabel }
    FILTER(BOUND(?bLabel))
    FILTER(CONTAINS(LCASE(STR(?d1)), LCASE(STR(?bLabel))))
    BIND(?d1 AS ?desc)
  }
  UNION
  { OPTIONAL { ?drugB rdfs:label ?bLabel2 }
    FILTER(!BOUND(?bLabel2))
    ?drugB db:has_synonym ?bSyn .
    FILTER(CONTAINS(LCASE(STR(?d1)), LCASE(STR(?bSyn))))
    BIND(?d1 AS ?desc)
  }
}''',

    # 4. 食物推荐
    4: r'''PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT (COALESCE(STR(?fl), STRAFTER(STR(?food), "#")) AS ?food_label) WHERE {
  { SELECT ?dis WHERE {
      BIND(LCASE("{{DISEASE}}") AS ?_dis_name)
      OPTIONAL { ?x1 rdfs:label ?dl . FILTER(LCASE(STR(?dl)) = ?_dis_name) BIND(?x1 AS ?label_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit)) ?x3 ?pp ?oo .
                 BIND(STRAFTER(STR(?x3), "#") AS ?dlocal) FILTER(LCASE(?dlocal) = ?_dis_name) BIND(?x3 AS ?local_hit) }
      BIND(COALESCE(?label_hit, ?local_hit) AS ?dis)
    } LIMIT 1 }
  ?dis db:has_food_suitability ?food .
  OPTIONAL { ?food rdfs:label ?fl }
}
ORDER BY LCASE(COALESCE(STR(?fl), STRAFTER(STR(?food), "#")))''',

    # 5. 食物禁忌
    5: r'''PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT (COALESCE(STR(?fl), STRAFTER(STR(?food), "#")) AS ?food_label) WHERE {
  { SELECT ?dis WHERE {
      BIND(LCASE("{{DISEASE}}") AS ?_dis_name)
      OPTIONAL { ?x1 rdfs:label ?dl . FILTER(LCASE(STR(?dl)) = ?_dis_name) BIND(?x1 AS ?label_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit)) ?x3 ?pp ?oo .
                 BIND(STRAFTER(STR(?x3), "#") AS ?dlocal) FILTER(LCASE(?dlocal) = ?_dis_name) BIND(?x3 AS ?local_hit) }
      BIND(COALESCE(?label_hit, ?local_hit) AS ?dis)
    } LIMIT 1 }
  ?dis db:has_food_taboo ?food .
  OPTIONAL { ?food rdfs:label ?fl }
}
ORDER BY LCASE(COALESCE(STR(?fl), STRAFTER(STR(?food), "#")))''',

    # 6. 护理建议（占位）
    6: r'''PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT
  (GROUP_CONCAT(DISTINCT STR(?food);    separator="|") AS ?food_suitability)
  (GROUP_CONCAT(DISTINCT STR(?dtreat);  separator="|") AS ?drug_treatments_desc)
  (GROUP_CONCAT(DISTINCT STR(?check);   separator="|") AS ?checks)
  (GROUP_CONCAT(DISTINCT STR(?nurs);    separator="|") AS ?nursing_desc)
WHERE {
  { SELECT ?dis WHERE {
      BIND(LCASE("{{DISEASE}}") AS ?_dis_name)
      OPTIONAL { ?d1 rdfs:label ?dl .
                 FILTER(LCASE(STR(?dl)) = ?_dis_name)
                 BIND(?d1 AS ?label_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit))
                 ?d2 db:has_synonym ?syn .
                 FILTER(LCASE(STR(?syn)) = ?_dis_name)
                 BIND(?d2 AS ?syn_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit) && !BOUND(?syn_hit))
                 ?d3 ?pp ?oo .
                 BIND(STRAFTER(STR(?d3), "#") AS ?dlocal)
                 FILTER(LCASE(?dlocal) = ?_dis_name)
                 BIND(?d3 AS ?local_hit) }
      BIND(COALESCE(?label_hit, ?syn_hit, ?local_hit) AS ?dis)
    } LIMIT 1 }

  OPTIONAL { ?dis db:has_food_suitability      ?food . }
  OPTIONAL { ?dis db:drugTreatmentDescription  ?dtreat . }
  OPTIONAL { ?dis db:has_check                 ?check . }
  OPTIONAL { ?dis db:nursingDescription        ?nurs . }
}
''',

    # 7. 疾病症状
    7: r'''PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT (COALESCE(STR(?sl), STRAFTER(STR(?sym), "#")) AS ?symptom) WHERE {
  { SELECT ?dis WHERE {
      BIND(LCASE("{{DISEASE}}") AS ?_dis_name)
      OPTIONAL { ?x1 rdfs:label ?dl . FILTER(LCASE(STR(?dl)) = ?_dis_name) BIND(?x1 AS ?label_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit)) ?x3 ?pp ?oo .
                 BIND(STRAFTER(STR(?x3), "#") AS ?dlocal) FILTER(LCASE(?dlocal) = ?_dis_name) BIND(?x3 AS ?local_hit) }
      BIND(COALESCE(?label_hit, ?local_hit) AS ?dis)
    } LIMIT 1 }
  ?dis db:typical_symptom ?sym .
  OPTIONAL { ?sym rdfs:label ?sl }
}
ORDER BY LCASE(COALESCE(STR(?sl), STRAFTER(STR(?sym), "#")))''',

    # 8. 检查项目
    8: r'''PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT (COALESCE(STR(?cl), STRAFTER(STR(?chk), "#")) AS ?check_item) WHERE {
  { SELECT ?dis WHERE {
      BIND(LCASE("{{DISEASE}}") AS ?_dis_name)
      OPTIONAL { ?x1 rdfs:label ?dl . FILTER(LCASE(STR(?dl)) = ?_dis_name) BIND(?x1 AS ?label_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit)) ?x3 ?pp ?oo .
                 BIND(STRAFTER(STR(?x3), "#") AS ?dlocal) FILTER(LCASE(?dlocal) = ?_dis_name) BIND(?x3 AS ?local_hit) }
      BIND(COALESCE(?label_hit, ?local_hit) AS ?dis)
    } LIMIT 1 }
  ?dis db:has_check ?chk .
  OPTIONAL { ?chk rdfs:label ?cl }
}
ORDER BY LCASE(COALESCE(STR(?cl), STRAFTER(STR(?chk), "#")))''',

    # 9. 药物描述
    9: r'''PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dc:   <http://purl.org/dc/elements/1.1/>
SELECT DISTINCT ?fact WHERE {
  { SELECT ?drug WHERE {
      BIND(LCASE("{{DRUG}}") AS ?_dname)
      OPTIONAL { ?d1 rdfs:label ?lbl . FILTER(LCASE(STR(?lbl)) = ?_dname) BIND(?d1 AS ?label_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit)) ?d2 db:has_synonym ?syn .
                 FILTER(LCASE(STR(?syn)) = ?_dname) BIND(?d2 AS ?syn_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit) && !BOUND(?syn_hit)) ?d3 ?p ?o .
                 BIND(STRAFTER(STR(?d3), "#") AS ?local) FILTER(LCASE(?local) = ?_dname) BIND(?d3 AS ?local_hit) }
      BIND(COALESCE(?label_hit, ?syn_hit, ?local_hit) AS ?drug)
    } LIMIT 1 }
  { ?drug dc:description ?d . BIND(STR(?d) AS ?fact) }
  UNION { ?drug rdfs:comment ?c . BIND(STR(?c) AS ?fact) }
  UNION { ?drug db:drug_type ?t . BIND(CONCAT("Drug type: ", STR(?t)) AS ?fact) }
  UNION { ?drug db:atc_code ?a . BIND(CONCAT("ATC code: ", STR(?a)) AS ?fact) }
  UNION { ?drug db:off_label_use ?o . BIND(CONCAT("Off-label: ", STR(?o)) AS ?fact) }
}''',

    # 10. 药品相关出版物
    10: r'''PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT
  (COALESCE(STR(?pl), STRAFTER(STR(?pub), "#")) AS ?pub_node)
  ?pmid
WHERE {
  { SELECT ?drug WHERE {
      BIND(LCASE("{{DRUG}}") AS ?_dname)
      OPTIONAL { ?d1 rdfs:label ?lbl . FILTER(LCASE(STR(?lbl)) = ?_dname) BIND(?d1 AS ?label_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit)) ?d2 db:has_synonym ?syn .
                 FILTER(LCASE(STR(?syn)) = ?_dname) BIND(?d2 AS ?syn_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit) && !BOUND(?syn_hit)) ?d3 ?p ?o .
                 BIND(STRAFTER(STR(?d3), "#") AS ?local) FILTER(LCASE(?local) = ?_dname) BIND(?d3 AS ?local_hit) }
      BIND(COALESCE(?label_hit, ?syn_hit, ?local_hit) AS ?drug)
    } LIMIT 1 }
  ?drug db:has_publication ?pub .
  OPTIONAL { ?pub rdfs:label   ?pl }
  OPTIONAL { ?pub db:pubmed_id ?pmid }
}
ORDER BY ?pmid''',

#11 疾病描述
    11: r'''PREFIX db:   <http://kgqa.cn/depression#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT (GROUP_CONCAT(DISTINCT STR(?desc); separator="|") AS ?description)
WHERE {
  { SELECT ?dis WHERE {
      BIND(LCASE("{{DISEASE}}") AS ?_dis_name)
      OPTIONAL { ?d1 rdfs:label ?dl .
                 FILTER(LCASE(STR(?dl)) = ?_dis_name)
                 BIND(?d1 AS ?label_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit))
                 ?d2 db:has_synonym ?syn .
                 FILTER(LCASE(STR(?syn)) = ?_dis_name)
                 BIND(?d2 AS ?syn_hit) }
      OPTIONAL { FILTER(!BOUND(?label_hit) && !BOUND(?syn_hit))
                 ?d3 ?pp ?oo .
                 BIND(STRAFTER(STR(?d3), "#") AS ?dlocal)
                 FILTER(LCASE(?dlocal) = ?_dis_name)
                 BIND(?d3 AS ?local_hit) }
      BIND(COALESCE(?label_hit, ?syn_hit, ?local_hit) AS ?dis)
    } LIMIT 1 }

  ?dis db:description ?desc .
}'''
}


# ==============
# 渲染函数
# ==============
def _escape(s: str) -> str:
    return (s or "").replace('\\', '\\\\').replace('"', '\\"')

def render(cid: int, slots: dict) -> str:
    """根据 cid + 槽位字典 生成可执行 SPARQL 查询字符串"""
    tmpl = TEMPLATES.get(cid)
    if not tmpl:
        raise KeyError(f"未知 cid 模版：{cid}")
    for k, v in {
        "{{DRUG}}": _escape(slots.get("drug", "")),
        "{{DRUG_A}}": _escape(slots.get("drug_a", "")),
        "{{DRUG_B}}": _escape(slots.get("drug_b", "")),
        "{{DISEASE}}": _escape(slots.get("disease", "")),
    }.items():
        tmpl = tmpl.replace(k, v)
    return tmpl
