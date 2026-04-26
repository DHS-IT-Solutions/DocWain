from src.intelligence.adapters.detect import detect_domain, DetectionResult


def test_high_confidence_returns_classifier_label():
    def fake_classifier(text):
        return ("insurance", 0.9)

    r = detect_domain("Policy SYN-001 ...", classifier=fake_classifier)
    assert isinstance(r, DetectionResult)
    assert r.domain == "insurance"
    assert r.confidence == 0.9
    assert r.fallback_to_generic is False


def test_low_confidence_falls_back_to_generic():
    def fake_classifier(text):
        return ("medical", 0.3)

    r = detect_domain("ambiguous text", classifier=fake_classifier)
    assert r.domain == "generic"
    assert r.fallback_to_generic is True
    assert r.confidence == 0.3


def test_threshold_is_0_7():
    def at_threshold(text):
        return ("hr", 0.7)

    r = detect_domain("x", classifier=at_threshold)
    assert r.domain == "hr"
    assert r.fallback_to_generic is False


def test_keyword_override_routes_insurance():
    text = "Policy number: X. Policyholder: Y. Premium: $1800. Deductible: $500. Liability limit: $25K."
    r = detect_domain(text)
    assert r.domain == "insurance"
    assert r.fallback_to_generic is False


def test_keyword_override_routes_medical():
    text = "Patient: J. MRN: M-001. Chief complaint: chest pain. Diagnosis: hypertension. Vitals: BP 145/95."
    r = detect_domain(text)
    assert r.domain == "medical"


def test_keyword_override_routes_hr():
    text = "Employee Jane. Hire date 2024-01-15. PTO balance 14 days. At-will employment. Performance review cycle: annual."
    r = detect_domain(text)
    assert r.domain == "hr"


def test_keyword_override_routes_procurement():
    text = "RFP-001. Vendor: ACME. Net 30 days. SLA: 99.9% uptime. MOQ: 5 units. Early-pay discount: 2%."
    r = detect_domain(text)
    assert r.domain == "procurement"


def test_classifier_label_policy_maps_to_insurance():
    def fake(text):
        return ("policy", 0.85)
    r = detect_domain("text without strong keywords for override", classifier=fake)
    assert r.domain == "insurance"


def test_unknown_label_falls_back():
    def fake(text):
        return ("totally-unknown-label", 0.95)
    r = detect_domain("ambiguous", classifier=fake)
    assert r.domain == "generic"
    assert r.fallback_to_generic is True
