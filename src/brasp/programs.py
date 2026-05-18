"""B-RASP programs for all 6 syntactic phenomena.

Each program consists of exactly 2 operations:
  1. HA_left[selector; value] -- hard attention (atomic variable)
  2. Pointwise Boolean        -- compatibility check (compositional variable)
"""

def ha_left(selector, value, tokens):
    """Hard-attention leftward: return value(j) for leftmost j <= i with selector(j)=1."""
    for j, tok in enumerate(tokens):
        if selector(tok):
            return value(tok)
    return False

def agreement_program(tokens, verb_pos):
    """SUBJ_NUM = HA_left[Q_noun; Q_n_pl]; VERB_FORM = SUBJ_NUM <-> Q_v_pl."""
    from data.alphabet import Q_noun, Q_n_pl, Q_v_pl
    subj_num = ha_left(Q_noun, Q_n_pl, tokens[:verb_pos + 1])
    verb_form = (subj_num == Q_v_pl(tokens[verb_pos]))
    return {"subj_num": subj_num, "verb_form": verb_form}

def npi_program(tokens, npi_pos):
    """HAS_LIC = HA_left[Q_lic; Q_lic]; NPI_OK = ~Q_npi | HAS_LIC."""
    from data.alphabet import Q_lic, Q_npi
    has_lic = ha_left(Q_lic, Q_lic, tokens[:npi_pos + 1])
    npi_ok = (not Q_npi(tokens[npi_pos])) or has_lic
    return {"has_lic": has_lic, "npi_ok": npi_ok}

def reflexive_program(tokens, refl_pos):
    """ANTE_NUM = HA_left[Q_noun; Q_n_pl]; REFL_OK = ANTE_NUM <-> Q_refl_pl."""
    from data.alphabet import Q_noun, Q_n_pl, Q_refl_pl
    ante_num = ha_left(Q_noun, Q_n_pl, tokens[:refl_pos + 1])
    refl_ok = (ante_num == Q_refl_pl(tokens[refl_pos]))
    return {"ante_num": ante_num, "refl_ok": refl_ok}

def concord_program(tokens, noun_pos):
    """DET_NUM = HA_left[Q_det; Q_det_pl]; CONC_OK = DET_NUM <-> Q_n_pl."""
    from data.alphabet import Q_noun, Q_n_pl
    Q_det = lambda x: x in ("det_sg", "det_pl")
    Q_det_pl = lambda x: x == "det_pl"
    det_num = ha_left(Q_det, Q_det_pl, tokens[:noun_pos + 1])
    conc_ok = (det_num == Q_n_pl(tokens[noun_pos]))
    return {"det_num": det_num, "conc_ok": conc_ok}

PROGRAMS = dict(agreement=agreement_program, npi=npi_program,
                reflexive=reflexive_program, concord=concord_program,
                mc_npi=npi_program, cc_binding=reflexive_program)
VARIABLES = {
    "agreement": ["subj_num", "verb_form"], "npi": ["has_lic", "npi_ok"],
    "reflexive": ["ante_num", "refl_ok"], "concord": ["det_num", "conc_ok"],
    "mc_npi": ["has_lic", "npi_ok"], "cc_binding": ["ante_num", "refl_ok"],
}
