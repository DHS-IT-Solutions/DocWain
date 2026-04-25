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
