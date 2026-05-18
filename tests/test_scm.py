"""Unit tests for SCM compilation and counterfactual computation."""
import pytest
from src.brasp.scm import compile_agreement, compile_npi

class TestAgreementSCM:
    def setup_method(self):
        self.scm = compile_agreement()

    def test_forward_match(self):
        vals = self.scm.predict({"u_subj": 1, "u_verb": 1})
        assert vals["subj_num"] == 1
        assert vals["verb_form"] == 1  # match

    def test_forward_mismatch(self):
        vals = self.scm.predict({"u_subj": 1, "u_verb": 0})
        assert vals["verb_form"] == 0  # mismatch

    def test_counterfactual(self):
        u_base = {"u_subj": 0, "u_verb": 0}    # sg subj, sg verb -> match
        u_source = {"u_subj": 1, "u_verb": 1}  # pl subj
        cf = self.scm.counterfactual(u_base, u_source, "subj_num")
        assert cf["verb_form"] == 0  # pl subj + sg verb -> mismatch

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
