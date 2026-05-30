"""
Tests for evaluation utilities.
"""


class TestEvaluationDirection:
    """Regression tests for score orientation handling."""

    def test_evaluate_detector_handles_inverted_scores(self):
        """Perfectly inverted scores should still evaluate as perfect."""
        from src.evaluate import evaluate_detector

        labels = [0, 0, 1, 1]
        scores = [0.9, 0.8, 0.2, 0.1]

        result = evaluate_detector(labels, scores)

        assert result.auroc == 1.0
        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_comprehensive_evaluation_uses_oriented_scores_for_accuracy_ci(self):
        """Accuracy CI should be based on the same score direction as accuracy."""
        from src.evaluate import comprehensive_evaluation

        labels = [0, 0, 1, 1]
        scores = [0.9, 0.8, 0.2, 0.1]

        result = comprehensive_evaluation(labels, scores, n_bootstrap=50)

        assert result["auroc"] == 1.0
        assert result["accuracy"] == 1.0
        assert result["accuracy_ci_lower"] == 1.0
        assert result["accuracy_ci_upper"] == 1.0
