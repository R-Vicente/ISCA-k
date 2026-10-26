"""Produce the tables and tests of chapter 5 of the dissertation from the CSV files in resultados/.

Usage:
    python analise/analise_tese.py

Writes to analise/saida/:
    tabela_5_1.csv               mean ranks, wins and errors at MCAR 20%
    ranks_por_condicao.csv       mean rank of each method in each condition (NRMSE and PFC)
    medias_por_condicao.csv      mean over datasets of NRMSE and PFC per condition
    friedman.csv                 Friedman test with the Iman-Davenport F statistic, Holm per metric and scope
    wilcoxon_holm.csv            exact Wilcoxon, ISCA-k as control, Holm over 40 contrasts per metric
    diferenca_melhor.csv         percentage difference between ISCA-k and the best other method (Figure 5.2)
    descritores_datasets.csv     descriptors of Table 4.1, computed from the snapshots
    tabela_5_2_spearman.csv      Spearman association between descriptors and the percentage difference
    tempos_comparacao.csv        dataset and rate pairs in which each method is faster than ISCA-k
    tabela_5_8.csv               distribution preservation measures
    distribuicoes_testes.csv     Friedman (Iman-Davenport F) and Wilcoxon with Holm over 10 contrasts
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
OUT = RAIZ / "analise" / "saida"
CONDICOES = ["mcar_05", "mcar_10", "mcar_20", "mcar_30", "mcar_40", "mcar_50", "blockwise_20", "mar_20"]
MCAR = CONDICOES[:6]
METODOS = ["Mean/Mode", "k-NN", "MICE", "missForest", "SoftImpute", "ISCA-k"]
CONTROLO = "ISCA-k"


def wilcoxon_exato(x, y):
    """Two-sided Wilcoxon test with exact p by enumerating the 2^n sign patterns (mid-ranks for ties)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    r = stats.rankdata(np.abs(nz))
    obs = r[nz > 0].sum()
    vals = np.array([np.dot(s, r) for s in product((0, 1), repeat=n)])
    return float(min(1.0, 2 * min(np.mean(vals <= obs + 1e-12), np.mean(vals >= obs - 1e-12))))


def holm(p):
    p = np.asarray(p, float)
    ordem = np.argsort(p)
    m = len(p)
    out = np.empty(m)
    ant = 0.0
    for k, i in enumerate(ordem):
        ant = max(ant, min(1.0, (m - k) * p[i]))
        out[i] = ant
    return out


def friedman_id(tabela):
    """Friedman (chi-squared) and the Iman-Davenport F statistic, as in Demšar (2006)."""
    chi, p = stats.friedmanchisquare(*[tabela[c] for c in tabela.columns])
    N, k = tabela.shape
    F = (N - 1) * chi / (N * (k - 1) - chi)
    pF = stats.f.sf(F, k - 1, (k - 1) * (N - 1))
    return chi, p, F, pF


def estudo_principal():
    b = pd.read_csv(RES / "estudo_principal" / "benchmark_repeticoes.csv")
    medias = b.groupby(["dataset", "condition", "method"])[["nrmse", "pfc"]].mean().reset_index()
    cat = medias[medias.pfc.notna()].dataset.unique()
    ranks, vit, fr, wil = [], [], [], []
    for met in ("nrmse", "pfc"):
        familia = []
        for c in CONDICOES:
            t = medias[medias.condition == c].pivot(index="dataset", columns="method", values=met)[METODOS]
            if met == "pfc":
                t = t.loc[[d for d in t.index if d in cat]]
            r = t.rank(axis=1)
            melhor = t.idxmin(axis=1)
            for m in METODOS:
                ranks.append(dict(metrica=met, condicao=c, metodo=m, rank_medio=r[m].mean()))
                vit.append(dict(metrica=met, condicao=c, metodo=m, vitorias=int((melhor == m).sum())))
            for ambito, cols in (("seis_metodos", METODOS), ("sem_media_moda", METODOS[1:])):
                chi, p, F, pF = friedman_id(t[cols])
                fr.append(dict(metrica=met, condicao=c, ambito=ambito, n_datasets=len(t),
                               qui_quadrado=chi, p_qui_quadrado=p, F_iman_davenport=F, p_F=pF))
            for m in METODOS:
                if m != CONTROLO:
                    familia.append(dict(metrica=met, condicao=c, metodo=m,
                                        diferenca_media=float((t[CONTROLO] - t[m]).mean()),
                                        p=wilcoxon_exato(t[CONTROLO], t[m])))
        for x, ph in zip(familia, holm([x["p"] for x in familia])):
            x["p_holm"] = ph
        wil += familia
    fr = pd.DataFrame(fr)
    fr["p_F_holm"] = np.nan
    for _, g in fr.groupby(["metrica", "ambito"]):
        fr.loc[g.index, "p_F_holm"] = holm(g.p_F.values)
    ranks, vit = pd.DataFrame(ranks), pd.DataFrame(vit)
    ranks.to_csv(OUT / "ranks_por_condicao.csv", index=False)
    fr.to_csv(OUT / "friedman.csv", index=False)
    pd.DataFrame(wil).to_csv(OUT / "wilcoxon_holm.csv", index=False)
    medias.groupby(["condition", "method"])[["nrmse", "pfc"]].mean().reset_index().to_csv(
        OUT / "medias_por_condicao.csv", index=False)

    m20 = medias[medias.condition == "mcar_20"].groupby("method")[["nrmse", "pfc"]].mean()
    t51 = pd.DataFrame({
        "rank_nrmse": ranks[ranks.metrica == "nrmse"].groupby("metodo").rank_medio.mean(),
        "rank_pfc": ranks[ranks.metrica == "pfc"].groupby("metodo").rank_medio.mean(),
        "vitorias_nrmse": vit[vit.metrica == "nrmse"].groupby("metodo").vitorias.sum(),
        "nrmse_mcar20": m20.nrmse, "pfc_mcar20": m20.pfc}).loc[METODOS]
    t51.round(4).to_csv(OUT / "tabela_5_1.csv")

    p = medias.pivot_table(index=["dataset", "condition"], columns="method", values="nrmse")
    outros = p.drop(columns=CONTROLO)
    dif = (100 * (p[CONTROLO] / outros.min(axis=1) - 1)).rename("diferenca_percent").to_frame()
    dif["melhor_dos_outros"] = outros.idxmin(axis=1)
    dif.reset_index().to_csv(OUT / "diferenca_melhor.csv", index=False)
    return dif["diferenca_percent"].unstack()


def descritores():
    """Descriptors of Table 4.1, over the non-constant numeric variables of the complete tables."""
    man = json.loads((RAIZ / "dados" / "manifest.json").read_text(encoding="utf-8"))
    linhas = []
    for nome in man["dataset_order"]:
        e = man["datasets"][nome]
        with np.load(RAIZ / "dados" / e["snapshot_file"], allow_pickle=False) as z:
            X = np.asarray(z["X"], float)
        tipos = e["var_types"]
        idx = [j for j, t in enumerate(tipos) if t == "cont" and X[:, j].std() > 0]
        A = X[:, idx]
        corr = np.abs(np.corrcoef(A, rowvar=False)[np.triu_indices(len(idx), 1)])
        G = np.abs(stats.skew(A, axis=0, bias=False))
        rac = np.ptp(A, axis=0) / A.std(axis=0, ddof=1)
        linhas.append(dict(dataset=nome, linhas=X.shape[0], variaveis=X.shape[1],
                           numericas=tipos.count("cont"), categoricas=tipos.count("cat"),
                           prop_categoricas=tipos.count("cat") / X.shape[1],
                           r_medio=corr.mean(), r_maximo=corr.max(),
                           assimetria_media=G.mean(), assimetria_maxima=G.max(),
                           cv_racio_amplitude_dp=rac.std(ddof=0) / rac.mean()))
    d = pd.DataFrame(linhas).set_index("dataset")
    d.to_csv(OUT / "descritores_datasets.csv")
    return d


def spearman(dif, desc):
    rotulos = {"linhas": "Número de observações", "assimetria_media": "Assimetria absoluta média",
               "cv_racio_amplitude_dp": "Heterogeneidade do rácio amplitude/desvio-padrão",
               "variaveis": "Número de variáveis", "prop_categoricas": "Proporção de variáveis categóricas",
               "r_medio": "Correlação absoluta média"}
    linhas = []
    for col, rot in rotulos.items():
        rho = {c: stats.spearmanr(desc[col], dif.loc[desc.index, c]).statistic for c in MCAR}
        linhas.append(dict(caracteristica=rot, minimo=min(rho.values()), maximo=max(rho.values()),
                           **{f"rho_{c}": v for c, v in rho.items()}))
    pd.DataFrame(linhas).round(3).to_csv(OUT / "tabela_5_2_spearman.csv", index=False)


def tempos():
    t = pd.read_csv(RES / "tempos" / "tempos_repeticoes.csv")
    md = t.groupby(["dataset", "condition", "method"]).seconds.median().unstack()
    linhas = [dict(metodo=m, pares=len(md), mais_rapido_que_iscak=int((md[m] < md[CONTROLO]).sum()),
                   mediana_razao_tempo_metodo_sobre_iscak=float((md[m] / md[CONTROLO]).median()))
              for m in METODOS if m != CONTROLO]
    pd.DataFrame(linhas).to_csv(OUT / "tempos_comparacao.csv", index=False)


def distribuicoes():
    d = pd.read_csv(RES / "distribuicoes" / "distribuicoes_repeticoes.csv")
    pv = pd.read_csv(RES / "distribuicoes" / "distribuicoes_por_variavel_celulas_ocultas.csv")
    g = d.groupby(["dataset", "condition", "method"])[
        ["missing_wasserstein_sd", "missing_total_variation", "pearson_mae"]].mean()
    # R: per replicate, mean of the SD ratios of the numeric columns; then mean over the 75 replicates.
    g["R"] = (pv[pv.type == "cont"].groupby(["dataset", "condition", "method", "replicate"]).sd_ratio.mean()
              .groupby(["dataset", "condition", "method"]).mean())
    tab = g.groupby(["condition", "method"]).mean().unstack(0)
    tab.columns = [f"{a}_{b}" for a, b in tab.columns]
    tab.loc[METODOS].round(4).to_csv(OUT / "tabela_5_8.csv")
    linhas = []
    for met in ("missing_wasserstein_sd", "missing_total_variation", "pearson_mae"):
        familia = []
        for c in ("mcar_20", "mcar_40"):
            t = g[met].xs(c, level="condition").unstack()[METODOS].dropna()
            chi, p, F, pF = friedman_id(t)
            rk = t.rank(axis=1).mean()
            for m in METODOS:
                linhas.append(dict(medida=met, condicao=c, tipo="rank_medio", metodo=m, valor=rk[m]))
            linhas.append(dict(medida=met, condicao=c, tipo="friedman_p_F", metodo="", valor=pF))
            for m in METODOS[:-1]:
                familia.append(dict(medida=met, condicao=c, tipo="wilcoxon_p", metodo=m,
                                    valor=wilcoxon_exato(t[CONTROLO], t[m])))
        for x, ph in zip(familia, holm([x["valor"] for x in familia])):
            linhas.append(x)
            linhas.append({**x, "tipo": "wilcoxon_p_holm", "valor": ph})
    pd.DataFrame(linhas).to_csv(OUT / "distribuicoes_testes.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dif = estudo_principal()
    spearman(dif, descritores())
    tempos()
    distribuicoes()
    print("Tabelas escritas em", OUT)
