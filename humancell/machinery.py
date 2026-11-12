"""Molecular machines: which genes build them and which process each one limits.

Each machine is a list of *groups*; a group is a set of interchangeable genes (paralogues, summed).
The capacity of a machine is the scarcest group relative to the reference cell (Liebig's law):

    level  = min_groups( sum_g functional_copies / sum_g reference_copies ) / relative_cell_size
    factor = min(1, level / RESERVE)

RESERVE < 1 encodes that machines are not run at limiting stoichiometry: CRISPRi titration of
essential genes (Jost et al. 2020) shows growth is typically unaffected until a gene product is
depleted by more than ~70-80%. It also makes the cell robust to the 20-40% cell-to-cell noise that
low-copy mRNAs impose on individual subunits, as real cells are.

This list is hand-curated from textbook composition (the same kind of curation every whole-cell
model rests on). Genes not expressed in the cell line are dropped from their group.
"""
from __future__ import annotations

import re

import numpy as np

RESERVE = 0.2

# name -> list of groups; a group is "A|B|C" (paralogues) and may use a regex when prefixed with "re:"
MACHINES = {
    "rnap2": "POLR2A POLR2B POLR2C POLR2D POLR2E POLR2F POLR2G POLR2H POLR2I POLR2J POLR2K POLR2L GTF2B GTF2E1 "
             "GTF2E2 GTF2F1 GTF2F2 GTF2H1 GTF2H4 TBP ERCC3 ERCC2 CDK7 CCNH MNAT1 SUPT5H SUPT6H SUPT16H SSRP1 "
             "GTF2A1 TAF1 TAF2 TAF5 TAF6 TAF10 TAF12 MED4 MED6 MED7 MED8 MED11 MED14 MED17 MED20 MED21 MED30 "
             "CDK9 CCNT1 NELFB",
    "splicing": "SF3B1 SF3B2 SF3B3 SF3B4 SF3B5 SF3A1 SF3A2 SF3A3 SNRPA1 SNRPB|SNRPN SNRPD1 SNRPD2 SNRPD3 SNRPE "
                "SNRPF SNRPG SNRNP200 SNRNP70 SNRNP27 EFTUD2 PRPF3 PRPF4 PRPF6 PRPF8 PRPF19 PRPF31 PRPF38A U2AF1 "
                "U2AF2 LSM2 LSM3 LSM4 LSM5 LSM6 LSM7 LSM8 CDC5L PLRG1 BUD31 SNW1 DHX15 DHX16 DDX23 AQR XAB2 "
                "PHF5A SART1 SART3 USP39 SRSF1 SRSF2 SRSF3 HNRNPC HNRNPK HNRNPU HNRNPL HNRNPM",
    "mrna_export": "NXF1 NCBP1 NCBP2 CPSF1 CPSF2 CPSF3 CPSF4 CSTF1 CSTF3 NUDT21 PABPN1 THOC1 THOC2 THOC3 THOC5 "
                   "THOC7 DDX39B ALYREF|POLDIP3 GLE1 WDR33 SYMPK",
    "rnap1": "POLR1A POLR1B POLR1C POLR1D POLR1E POLR1F POLR1G UBTF RRN3 TAF1B TAF1C",
    "rnap3": "POLR3A POLR3B POLR3C POLR3D POLR3E POLR3F POLR3H POLR3K GTF3C1 GTF3C2 BRF1 BDP1",
    "ribosome": "re:^RP[LS]\\d+[AXY]?$ RPLP0 RPLP1 RPLP2 RPSA RACK1 UBA52 FAU",
    "ribosome_biogenesis": "FBL NOP56 NOP58 DKC1 NOP2 NAT10 BOP1 PES1 WDR12 UTP3 UTP4 UTP6 UTP15 UTP18 UTP20 "
                           "NOC4L NOL6 RRP1 RRP7A RRP9 RRP12 BMS1 TSR1 TSR2 RIOK2 LTV1 NOB1 WDR3 WDR36 WDR43 "
                           "WDR46 WDR74 WDR75 DDX10 DDX18 DDX21 DDX47 DDX49 DDX52 DDX54 DDX56 HEATR1 NOL11 "
                           "NIP7 SDAD1 GNL2 GNL3 NLE1 NVL MDN1 EIF6 NMD3 XPO1 RPF1 RPF2 BRIX1 EBNA1BP2 NOP16",
    "trna_charging": "AARS1|AARS CARS1|CARS DARS1|DARS EPRS1|EPRS FARSA FARSB GARS1|GARS HARS1|HARS IARS1|IARS "
                     "KARS1|KARS LARS1|LARS MARS1|MARS NARS1|NARS QARS1|QARS RARS1|RARS SARS1|SARS TARS1|TARS "
                     "VARS1|VARS WARS1|WARS YARS1|YARS",
    "translation_factors": "EIF1AX EIF2S1 EIF2S2 EIF2S3 EIF2B1 EIF2B2 EIF2B3 EIF2B4 EIF2B5 EIF3A EIF3B EIF3C EIF3D "
                           "EIF3E EIF3G EIF3I EIF4A1 EIF4E EIF4G1 EIF5 EIF5B EEF1A1 EEF2 EEF1B2 EEF1G ETF1 GSPT1 "
                           "PABPC1 EIF5A DHPS",
    "proteasome": "PSMA1 PSMA2 PSMA3 PSMA4 PSMA5 PSMA6 PSMA7 PSMB1 PSMB2 PSMB3 PSMB4 PSMB5 PSMB6 PSMB7 PSMC1 "
                  "PSMC2 PSMC3 PSMC4 PSMC5 PSMC6 PSMD1 PSMD2 PSMD3 PSMD4 PSMD6 PSMD7 PSMD11 PSMD12 PSMD13 PSMD14 "
                  "UBA1 UBB|UBC|RPS27A|UBA52 NPLOC4 UFD1 VCP",
    "chaperones": "TCP1 CCT2 CCT3 CCT4 CCT5 CCT6A CCT7 CCT8 HSPA8 HSP90AB1|HSP90AA1 HSPD1 HSPE1 PFDN2 VBP1",
    "er_folding": "HSPA5 HSP90B1 CANX|CALR PDIA3|P4HB|PDIA6",
    "translocon": "SEC61A1 SEC61G SEC62 SEC63 SRP14 SRP19 SRP54 SRP68 SRP72 SRPRA SRPRB SPCS2 SPCS3 SEC11A "
                  "DAD1 DDOST RPN1 RPN2 STT3A|STT3B",
    "vesicle_transport": "SEC13 SEC16A SEC23A|SEC23B SAR1A|SAR1B COPA COPB1 COPB2 ARCN1 COPG1 COPE COPZ1 ARF1|ARF4 "
                         "NSF NAPA STX5 YKT6 SEC22B TRAPPC1 TRAPPC3 TRAPPC8 RAB1A|RAB1B GBF1 DNM2 CLTC AP2M1 AP2S1",
    "nuclear_transport": "KPNB1 RAN RCC1 RANGAP1 NUTF2 NUP93 NUP205 NUP98 NUP133 NUP160 NUP214 NUP62 NUP85 NUP88 "
                         "NUP107 NUP153 NUP155 XPO1 CSE1L TNPO1 KPNA2|KPNA3|KPNA4|KPNA6",
    "mito_ribosome": "re:^MRP[LS]\\d+$ TUFM GFM1 GFM2 TSFM MTIF2 POLRMT TFAM TFB2M MTPAP",
    "mito_import": "TOMM20 TOMM22 TOMM40 TIMM23 TIMM17A|TIMM17B TIMM44 TIMM10 TIMM13 TIMM22 PAM16 DNAJC19|DNAJC15 "
                   "HSPA9 GRPEL1 PMPCA PMPCB SAMM50 CHCHD4",
    "mrna_decay": "EXOSC1 EXOSC2 EXOSC3 EXOSC4 EXOSC5 EXOSC6 EXOSC7 EXOSC8 EXOSC9 EXOSC10 DIS3 CNOT1 CNOT2 CNOT3 "
                  "XRN1|XRN2 DCP2 DCP1A",
    "replication": "MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 CDC45 GINS1 GINS2 GINS3 GINS4 POLA1 POLA2 PRIM1 PRIM2 POLD1 "
                   "POLD2 POLD3 POLE POLE2 PCNA RFC1 RFC2 RFC3 RFC4 RFC5 RPA1 RPA2 RPA3 LIG1 FEN1 TOP2A|TOP2B "
                   "TOP1 DNA2 MCM10 TOPBP1 TICRR CLSPN TIMELESS DUT",
    "licensing": "ORC1 ORC2 ORC3 ORC4 ORC5 ORC6 CDC6 CDT1 CDC7 DBF4",
    "chromatin": "re:^H2AC\\d+$|re:^H2AZ\\d$|H2AX re:^H2BC\\d+$|H2BU1 re:^H3C\\d+$|re:^H3-3[AB]$ re:^H4C\\d+$|H4-16 CHAF1A CHAF1B "
                 "ASF1A|ASF1B NASP SMARCA4|SMARCA2 SMARCB1 HDAC1|HDAC2 RBBP4 RBBP7 DNMT1 UHRF1 KAT5 TRRAP",
    "cohesin_condensin": "SMC1A SMC3 RAD21 STAG1|STAG2 NIPBL SMC2 SMC4 NCAPD2 NCAPG NCAPH ESPL1 PTTG1",
    "spindle": "TUBB|TUBB4B|TUBB2A|TUBB2B|TUBB6|TUBB3 TUBA1B|TUBA1C|TUBA1A|TUBA4A TUBG1 TUBGCP2 TUBGCP3 KIF11 "
               "NDC80 NUF2 SPC24 SPC25 CENPA CENPE CENPC CENPM CENPN CENPW KNL1 MIS12 DSN1 BUB1B BUB3 MAD2L1 TTK "
               "PLK1 AURKB INCENP BIRC5 CDCA8 AURKA TPX2 CKAP5 CLASP1|CLASP2 KIF18A|KIF18B NUMA1 HAUS1 HAUS5 "
               "SKA1 SKA3 CCNB1|CCNB2 CDK1",
    "apc": "ANAPC1 ANAPC2 ANAPC4 ANAPC5 ANAPC10 ANAPC11 CDC16 CDC23 CDC27 CDC20 UBE2C|UBE2S",
    "cytokinesis": "KIF23 RACGAP1 ECT2 PRC1 ANLN RHOA CIT|ROCK1|ROCK2 MYH9|MYH10 ACTB|ACTG1 KIF20A CEP55 CHMP2A "
                   "CHMP4B|CHMP4A CHMP6 VPS4A|VPS4B PDCD6IP|TSG101",
    "nhej": "XRCC5 XRCC6 PRKDC LIG4 XRCC4 NHEJ1",
    "hr": "BRCA1 BRCA2 RAD51 PALB2 RAD50 MRE11 NBN RBBP8 RAD51C XRCC2|XRCC3 BARD1",
    "na_k_atpase": "ATP1A1|ATP1A2|ATP1A3 ATP1B1|ATP1B3",
    "serca": "ATP2A2|ATP2A3|ATP2A1",
    "pmca": "ATP2B1|ATP2B4",
    "v_atpase": "ATP6V1A ATP6V1B2 ATP6V1C1 ATP6V1D ATP6V1E1 ATP6V1F ATP6V1H ATP6V0C ATP6V0D1 ATP6AP1 ATP6AP2",
    "antioxidant": "SOD1 SOD2 PRDX1|PRDX2 TXN TXNRD1 GPX4|GPX1 GSR GCLC GCLM GSS",
    "caspases": "CASP3|CASP7 CASP9 APAF1 CYCS",
    "iron_sulfur": "NFS1 ISCU FXN LYRM4 HSCB GLRX5 CIAO1 NUBP1 NUBP2 CIAPIN1 NARFL",
    "cap_methylation": "RNGTT RNMT CMTR1",
    "trna_processing": "RTCB DDX1 TSEN2 TSEN54 TRMT112 TRNT1 RPP14 RPP21 RPP30 RPP38 RPP40 POP1 POP4 POP5 POP7 "
                       "ELP1 ELP2 ELP3 CTU1 CTU2 NSUN2|NSUN3 TRMT5 TRMT61A",
    "gpi_glycosylation": "ALG1 ALG2 ALG11 ALG13 ALG14 DPAGT1 DPM1 DOLK",
}
# membrane / secreted proteins need the translocon and vesicle traffic, nuclear proteins need import, etc.
COMPARTMENT_MACHINE = {"er": ["translocon"], "golgi": ["translocon", "vesicle_transport"],
                       "plasma_membrane": ["translocon", "vesicle_transport"], "secreted": ["translocon", "vesicle_transport"],
                       "vesicles": ["translocon", "vesicle_transport"], "nucleus": ["nuclear_transport"],
                       "mitochondria": ["mito_import"]}


class Machines:
    def __init__(self, symbols, ref_protein):
        self.sym_idx = {}
        for i, s in enumerate(symbols):
            self.sym_idx.setdefault(s, i)
        symbols = np.asarray(symbols)
        self.idx, self.starts, self.ref, self.members = {}, {}, {}, {}
        for name, spec in MACHINES.items():
            groups = []
            for token in spec.split():
                ids = []
                for alt in token.split("|"):
                    if alt.startswith("re:"):
                        rx = re.compile(alt[3:])
                        ids += [i for i, s in enumerate(symbols) if rx.match(s)]
                    elif alt in self.sym_idx:
                        ids.append(self.sym_idx[alt])
                ids = [i for i in dict.fromkeys(ids) if ref_protein[i] > 0]
                if not ids:
                    continue
                if token.startswith("re:") and "|" not in token:
                    groups += [[i] for i in ids]          # a regex family lists distinct subunits
                else:
                    groups.append(ids)
            flat = np.array([i for g in groups for i in g], dtype=int)
            starts = np.cumsum([0] + [len(g) for g in groups[:-1]]) if groups else np.array([], dtype=int)
            self.idx[name], self.starts[name] = flat, np.asarray(starts, dtype=int)
            self.ref[name] = np.add.reduceat(ref_protein[flat], starts) if len(flat) else np.array([])
            self.members[name] = groups
        self.all_genes = sorted({i for v in self.idx.values() for i in v})

    def factors(self, functional, size):
        out = {}
        for name, flat in self.idx.items():
            if len(flat) == 0:
                out[name] = 1.0
                continue
            level = np.add.reduceat(functional[flat], self.starts[name]) / (self.ref[name] * size)
            out[name] = float(min(level.min() / RESERVE, 1.0))
        return out
