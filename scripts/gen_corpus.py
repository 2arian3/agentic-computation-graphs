#!/usr/bin/env python3
"""Procedural fictional knowledge base + shape-diverse task generator.

Why: the canonical corpus is 16 docs / 3.5 KB and the tasks are almost all linear
multi-hop, so measured width is ~always 1. To *characterize* agentic application graphs
we need graphs that actually differ in shape. This generator builds a larger fictional
world from a knowledge-graph backbone and emits task families that each REQUIRE a
different realized-graph topology, while staying fully owned (no memorization) and
deterministic (seeded) -- so the only stochastic part of the system is still the model.

Task families (each induces a distinct ACG shape):
  linear_bridge        chain a relation path, ask the terminal attribute      -> depth
  fan_out_superlative  gather a numeric attr over N entities, take max/min     -> width (+compare)
  counting             count entities satisfying a relation                    -> width (+compare)
  numeric_diff         difference of two numeric attributes                    -> calculator node
  unanswerable         ask about a relation absent from the KB                 -> early-stop/failure

Outputs (under data/, override with --outdir):
  corpus_large.json         list[{id,title,text}]
  distractors_large.json    list[{id,title,text}]   (near-duplicate confusers)
  tasks_families.jsonl      {task_id, family, hops, question, answers, supporting}

Usage:
  ./.venv/bin/python scripts/gen_corpus.py --scale 12 --seed 1234
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# --- fictional name generation ------------------------------------------------
_ONSET = ["V", "K", "Br", "Dr", "Th", "Orr", "Sund", "Cor", "Vel", "Mor", "Pir", "El",
          "Gal", "Tarn", "Wex", "Zan", "Fenn", "Lum", "Ryd", "Quin", "Aur", "Nyx"]
_MID = ["a", "e", "o", "u", "el", "an", "or", "in", "ell", "ar", "un", "im", "ov"]
_END = ["ne", "st", "dar", "mora", "past", "vane", "th", "ium", "is", "os", "en", "wick",
        "gate", "fell", "mont", "dell", "ra", "va", "ke"]


def _namer(rng: random.Random):
    seen = set()
    def make(nsyl: int = 2) -> str:
        for _ in range(50):
            s = rng.choice(_ONSET) + "".join(rng.choice(_MID) for _ in range(nsyl - 1)) + rng.choice(_END)
            s = s.capitalize()
            if s not in seen:
                seen.add(s)
                return s
        n = f"{rng.choice(_ONSET)}{len(seen)}"
        seen.add(n)
        return n
    return make


def build_world(scale: int, seed: int) -> dict:
    """Build a small fictional KG. `scale` roughly sets entity counts per type."""
    rng = random.Random(seed)
    name = _namer(rng)

    n_cont = max(2, scale // 4)
    n_country = scale
    n_city = scale * 2
    n_inst = scale
    n_company = scale
    n_material = max(3, scale // 2)
    n_currency = n_country

    continents = [name(2) for _ in range(n_cont)]
    currencies = [name(2).lower() for _ in range(n_currency)]
    materials = [name(2).lower() for _ in range(n_material)]

    countries = []
    for i in range(n_country):
        countries.append({
            "name": name(2),
            "continent": rng.choice(continents),
            "currency": currencies[i],
            "population": rng.randint(2, 90) * 100000,
            "material": rng.choice(materials),   # a material mined here
        })

    cities = []
    for i in range(n_city):
        c = rng.choice(countries)
        cities.append({
            "name": name(2),
            "country": c["name"],
            "coastal": rng.random() < 0.5,
            "founded": rng.randint(1500, 1980),
        })
    # ensure each country has >=1 city and exactly one capital
    by_country: dict[str, list] = {}
    for ct in cities:
        by_country.setdefault(ct["country"], []).append(ct)
    for c in countries:
        lst = by_country.get(c["name"])
        if not lst:
            ct = {"name": name(2), "country": c["name"], "coastal": rng.random() < 0.5,
                  "founded": rng.randint(1500, 1980)}
            cities.append(ct); lst = [ct]; by_country[c["name"]] = lst
        cap = rng.choice(lst)
        cap["is_capital"] = True
        c["capital"] = cap["name"]
        c["capital_coastal"] = cap["coastal"]

    institutes = []
    for _ in range(n_inst):
        city = rng.choice(cities)
        institutes.append({
            "name": name(2) + " Institute",
            "city": city["name"],
            "founder": "Dr. " + name(2),
            "founded": rng.randint(1900, 2010),
            "field": rng.choice(["tidal-energy", "metallurgy", "cartography", "astronomy",
                                 "marine-biology", "seismology"]),
        })

    companies = []
    for _ in range(n_company):
        companies.append({
            "name": name(2) + " Dynamics",
            "product": name(2),
            "material": rng.choice(materials),   # its product's key material
            "hq_city": rng.choice(cities)["name"],
            "revenue": rng.randint(5, 400) * 10,  # millions
        })

    return dict(continents=continents, currencies=currencies, materials=materials,
                countries=countries, cities=cities, institutes=institutes,
                companies=companies, by_country=by_country)


def render_docs(world: dict) -> tuple[list[dict], list[dict]]:
    """One document per entity, in prose, so multi-hop chains require chaining docs."""
    docs: list[dict] = []
    dist: list[dict] = []
    n = [0]

    def add(title: str, text: str) -> str:
        n[0] += 1
        did = f"D{n[0]:04d}"
        docs.append({"id": did, "title": title, "text": text})
        return did

    for c in world["countries"]:
        add(c["name"],
            f"{c['name']} is a country on the continent of {c['continent']}. Its capital is "
            f"{c['capital']}. The currency used in {c['name']} is the {c['currency']}. Its "
            f"population is about {c['population']:,}. The mineral {c['material']} is mined in {c['name']}.")
    for ct in world["cities"]:
        coastal = "a coastal city" if ct["coastal"] else "an inland city"
        cap = " It is the capital." if ct.get("is_capital") else ""
        add(ct["name"],
            f"{ct['name']} is {coastal} in the country of {ct['country']}, founded in {ct['founded']}.{cap}")
    for it in world["institutes"]:
        add(it["name"],
            f"The {it['name']} is a research institute founded in {it['founded']} by {it['founder']}. "
            f"It is located in the city of {it['city']} and is known for its work on {it['field']} systems.")
    for co in world["companies"]:
        add(co["name"],
            f"{co['name']} is a company headquartered in {co['hq_city']}. It builds a product called "
            f"{co['product']}, whose key component is made of {co['material']}. Its annual revenue is "
            f"about {co['revenue']} million.")

    # near-duplicate distractors for a subset of institutes (confusers: similar name, different facts)
    rng = random.Random(99)
    for it in world["institutes"][: max(1, len(world["institutes"]) // 2)]:
        alt_city = rng.choice(world["cities"])["name"]
        dn = it["name"].replace("Institute", "Foundation")
        dist.append({"id": f"DX{len(dist)+1:04d}", "title": dn,
                     "text": f"The {dn} is a philanthropic foundation located in {alt_city}. "
                             f"It funds arts education and is unrelated to any research institute."})
    return docs, dist


def make_tasks(world: dict, docs: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed + 7)
    title2id = {d["title"]: d["id"] for d in docs}
    tasks: list[dict] = []
    tid = [0]

    def add(family: str, hops: int, question: str, answers: list[str], supporting: list[str]):
        tid[0] += 1
        tasks.append({"task_id": f"G{tid[0]:03d}", "family": family, "hops": hops,
                      "question": question, "answers": [a for a in answers if a],
                      "supporting": [s for s in supporting if s]})

    countries = {c["name"]: c for c in world["countries"]}
    cities = {c["name"]: c for c in world["cities"]}

    # 1) linear_bridge: institute -> city -> country -> currency  (3 hops)
    for it in world["institutes"]:
        city = cities.get(it["city"])
        if not city:
            continue
        country = countries.get(city["country"])
        if not country:
            continue
        add("linear_bridge", 3,
            f"What currency is used in the country where the {it['name']} is located?",
            [country["currency"], "the " + country["currency"]],
            [title2id.get(it["name"]), title2id.get(city["name"]), title2id.get(country["name"])])

    # 2) fan_out_superlative: among the countries on a continent, which has the largest population?
    by_cont: dict[str, list] = {}
    for c in world["countries"]:
        by_cont.setdefault(c["continent"], []).append(c)
    for cont, cs in by_cont.items():
        if len(cs) < 2:
            continue
        top = max(cs, key=lambda c: c["population"])
        add("fan_out_superlative", 2,
            f"Of the countries on the continent of {cont}, which has the largest population?",
            [top["name"]], [title2id.get(c["name"]) for c in cs])

    # 3) counting: how many companies build a product made of <material>?
    for material in world["materials"]:
        matches = [co for co in world["companies"] if co["material"] == material]
        if not matches:
            continue
        add("counting", 2,
            f"How many companies build a product whose key component is made of {material}?",
            [str(len(matches))], [title2id.get(co["name"]) for co in matches])

    # 4) numeric_diff: how many years after institute A was institute B founded?
    insts = world["institutes"]
    for _ in range(min(len(insts), 12)):
        a, b = rng.sample(insts, 2) if len(insts) >= 2 else (insts[0], insts[0])
        diff = b["founded"] - a["founded"]
        add("numeric_diff", 2,
            f"How many years after the {a['name']} was the {b['name']} founded? "
            f"(negative if it was founded earlier)",
            [str(diff)], [title2id.get(a["name"]), title2id.get(b["name"])])

    # 5) unanswerable: ask for a relation the corpus never states
    for it in rng.sample(insts, min(len(insts), 8)):
        add("unanswerable", 1,
            f"What is the annual research budget of the {it['name']}?",
            ["insufficient information", "unknown", "not stated"],
            [title2id.get(it["name"])])

    # 6) constraint_satisfaction: the company satisfying material AND hq-country (a conjunction
    #    -> verify-loops). Only emit pairs matched by EXACTLY ONE company so the gold is unique.
    from collections import defaultdict
    pair_matches: dict = defaultdict(list)
    for co in world["companies"]:
        city = cities.get(co["hq_city"])
        if city:
            pair_matches[(co["material"], city["country"])].append(co)
    n_cs = 0
    for (mat, country_name), matches in pair_matches.items():
        if len(matches) == 1 and n_cs < 8:
            co = matches[0]
            n_cs += 1
            add("constraint_satisfaction", 3,
                f"Which company builds a product made of {mat} and is headquartered in the "
                f"country of {country_name}?",
                [co["name"]],
                [title2id.get(co["name"]), title2id.get(co["hq_city"]), title2id.get(country_name)])

    # 7) conditional: branch on a boolean attribute (coastal) -> different answer path
    #    (conditional routing). The gold depends on the branch, so the model must read the fact
    #    and route accordingly.
    for it in rng.sample(insts, min(len(insts), 8)):
        city = cities.get(it["city"])
        if not city:
            continue
        q = (f"If the {it['name']} is located in a coastal city, answer with its founding year; "
             f"otherwise answer with the name of the city it is located in.")
        ans = str(it["founded"]) if city["coastal"] else it["city"]
        add("conditional", 2, q, [ans], [title2id.get(it["name"]), title2id.get(city["name"])])

    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=12, help="rough entity count per type")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    world = build_world(args.scale, args.seed)
    docs, dist = render_docs(world)
    tasks = make_tasks(world, docs, args.seed)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus_large.json").write_text(json.dumps(docs, indent=1), encoding="utf-8")
    (out / "distractors_large.json").write_text(json.dumps(dist, indent=1), encoding="utf-8")
    with (out / "tasks_families.jsonl").open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    from collections import Counter
    fam = Counter(t["family"] for t in tasks)
    print(f"corpus: {len(docs)} docs, {len(dist)} distractors")
    print(f"tasks : {len(tasks)} across families: {dict(fam)}")
    print(f"wrote -> {out}/corpus_large.json, distractors_large.json, tasks_families.jsonl")


if __name__ == "__main__":
    main()
