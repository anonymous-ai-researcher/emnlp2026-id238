"""Compile B-RASP programs to structural causal models (SCMs).

Each 2-operation B-RASP program yields a 2-variable SCM:
  U -> Atomic -> Compositional
with Compositional also receiving a direct exogenous input.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List

@dataclass
class CausalVariable:
    name: str
    var_type: str  # "atomic" or "compositional"
    parents: List[str] = field(default_factory=list)
    function: Callable = None

@dataclass
class SCM:
    exogenous: List[str]
    endogenous: Dict[str, CausalVariable] = field(default_factory=dict)

    def predict(self, u_values: dict) -> dict:
        """Forward pass through the SCM given exogenous values."""
        values = dict(u_values)
        for name, var in self.endogenous.items():
            parent_vals = [values[p] for p in var.parents]
            values[name] = var.function(*parent_vals)
        return values

    def counterfactual(self, u_base, u_source, variable):
        """Compute counterfactual: replace `variable` with its value under u_source."""
        val_source = self.predict(u_source)
        val_base = dict(u_base)
        val_base[variable] = val_source[variable]
        result = {}
        for name, var in self.endogenous.items():
            parent_vals = [val_base.get(p, self.predict(u_base).get(p)) for p in var.parents]
            result[name] = var.function(*parent_vals)
            val_base[name] = result[name]
        return result

def compile_agreement() -> SCM:
    scm = SCM(exogenous=["u_subj", "u_verb"])
    scm.endogenous["subj_num"] = CausalVariable(
        "subj_num", "atomic", parents=["u_subj"], function=lambda u: u)
    scm.endogenous["verb_form"] = CausalVariable(
        "verb_form", "compositional", parents=["subj_num", "u_verb"],
        function=lambda s, v: int(s == v))
    return scm

def compile_npi() -> SCM:
    scm = SCM(exogenous=["u_lic", "u_npi"])
    scm.endogenous["has_lic"] = CausalVariable(
        "has_lic", "atomic", parents=["u_lic"], function=lambda u: u)
    scm.endogenous["npi_ok"] = CausalVariable(
        "npi_ok", "compositional", parents=["has_lic", "u_npi"],
        function=lambda h, n: int((not n) or h))
    return scm

COMPILE = dict(agreement=compile_agreement, npi=compile_npi,
               reflexive=compile_agreement, concord=compile_agreement,
               mc_npi=compile_npi, cc_binding=compile_agreement)
