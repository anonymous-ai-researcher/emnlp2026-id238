"""11-symbol token alphabet for all syntactic phenomena."""

ALPHABET = ["n_sg", "n_pl", "v_sg", "v_pl", "det_sg", "det_pl",
            "refl_sg", "refl_pl", "npi", "lic", "other"]
SYM2ID = {s: i for i, s in enumerate(ALPHABET)}
ID2SYM = {i: s for i, s in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET)
PAD_ID = VOCAB_SIZE  # padding token

def Q(sym, x_i): return x_i == sym
def Q_noun(x_i):   return x_i in ("n_sg", "n_pl")
def Q_n_pl(x_i):   return x_i == "n_pl"
def Q_v_pl(x_i):   return x_i == "v_pl"
def Q_det_pl(x_i): return x_i == "det_pl"
def Q_refl_pl(x_i):return x_i == "refl_pl"
def Q_npi(x_i):    return x_i == "npi"
def Q_lic(x_i):    return x_i == "lic"
