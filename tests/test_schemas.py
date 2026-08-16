from clinical_rag.config import collection_name
from clinical_rag.schemas import LegalFlags


def test_legal_complete_requires_all_four():
    flags = LegalFlags()
    assert flags.complete() is False
    flags.open_access_or_reusable = True
    flags.redistribution_ok_for_indexing = True
    flags.edition_current = True
    assert flags.complete() is False
    flags.attribution_documented = True
    assert flags.complete() is True


def test_collection_name_slugs_model():
    name = collection_name("demo", "section_aware", "BAAI/bge-m3")
    assert name == "demo__section_aware__BAAI_bge-m3"
    assert "/" not in name
    assert len(name) <= 63
