"""Unit tests for B-RASP programs: verify correctness on known examples."""
import pytest
from src.brasp.programs import agreement_program, npi_program, reflexive_program, concord_program

class TestAgreement:
    def test_plural_subject_no_attractors(self):
        tokens = ["n_pl", "v_pl"]
        result = agreement_program(tokens, verb_pos=1)
        assert result["subj_num"] == True   # plural
        assert result["verb_form"] == True   # match

    def test_singular_subject_with_attractors(self):
        tokens = ["n_sg", "other", "n_pl", "other", "n_pl", "v_sg"]
        result = agreement_program(tokens, verb_pos=5)
        assert result["subj_num"] == False  # singular (leftmost noun)
        assert result["verb_form"] == True  # match

    def test_agreement_mismatch(self):
        tokens = ["n_pl", "v_sg"]
        result = agreement_program(tokens, verb_pos=1)
        assert result["subj_num"] == True   # plural
        assert result["verb_form"] == False  # mismatch

class TestNPI:
    def test_licensed_npi(self):
        tokens = ["lic", "other", "npi", "other"]
        result = npi_program(tokens, npi_pos=2)
        assert result["has_lic"] == True
        assert result["npi_ok"] == True

    def test_unlicensed_npi(self):
        tokens = ["other", "other", "npi", "other"]
        result = npi_program(tokens, npi_pos=2)
        assert result["has_lic"] == False
        assert result["npi_ok"] == False

class TestReflexive:
    def test_matching_reflexive(self):
        tokens = ["n_pl", "other", "n_sg", "other", "refl_pl"]
        result = reflexive_program(tokens, refl_pos=4)
        assert result["ante_num"] == True   # leftmost noun is plural
        assert result["refl_ok"] == True    # reflexive matches

class TestConcord:
    def test_matching_concord(self):
        tokens = ["det_pl", "n_pl", "other"]
        result = concord_program(tokens, noun_pos=1)
        assert result["det_num"] == True
        assert result["conc_ok"] == True

    def test_mismatching_concord(self):
        tokens = ["det_sg", "n_pl", "other"]
        result = concord_program(tokens, noun_pos=1)
        assert result["det_num"] == False
        assert result["conc_ok"] == False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
