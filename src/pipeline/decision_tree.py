"""Clinical Decision Tree

Risk stratification based on BiomedCLIP classification scores
and continuous patient risk factors (tobacco, smoking).

Formula:
    risk_score = clip_score * 10 + alpha * tobacco_norm + beta * smoking_norm

Where:
    tobacco_norm = min(tobacco_years / tobacco_significant_years, 1.0)
    smoking_norm = min(smoking_years / smoking_significant_years, 1.0)

YOLO confidence is NOT part of the risk score. YOLO acts as a binary gate
in the fusion module (approve/reject detection). Once approved, only
BiomedCLIP classification + clinical factors determine risk.
"""


class ClinicalDecisionTree:
    """Clinical risk assessment.

    Combines BiomedCLIP classification score with continuous
    tobacco/smoking risk factors to produce a risk level.

    All parameters are configurable via ClinicalConfig.
    """

    def __init__(self, config=None):
        """Initialize with clinical config.

        Args:
            config: ClinicalConfig dataclass instance, or None for defaults.
        """
        if config is not None:
            self.high_threshold = config.high_risk_threshold
            self.medium_threshold = config.medium_risk_threshold
            self.alpha = config.alpha
            self.beta = config.beta
            self.tobacco_significant_years = config.tobacco_significant_years
            self.smoking_significant_years = config.smoking_significant_years
        else:
            self.high_threshold = 8.0
            self.medium_threshold = 5.0
            self.alpha = 2.0
            self.beta = 1.5
            self.tobacco_significant_years = 15
            self.smoking_significant_years = 15

    def _normalize(self, years, significant_years):
        """Normalize years of usage to 0-1 range.

        0 years -> 0.0, significant_years -> 1.0, capped at 1.0.
        """
        if significant_years <= 0:
            return 0.0
        return min(years / significant_years, 1.0)

    def assess_risk(self, vision_score, tobacco_years=0, smoking_years=0, **kwargs):
        """Assess clinical risk level.

        Risk formula:
            risk_score = vision_score * 10
                       + alpha * tobacco_norm
                       + beta * smoking_norm

        Args:
            vision_score: BiomedCLIP classification score (0-1).
                With gate fusion, this is the raw CLIP score.
            tobacco_years: Years of smokeless tobacco use (int)
            smoking_years: Years of smoking (int)
            **kwargs: Ignored (backward-compatible with extra keys)

        Returns:
            dict: Risk assessment containing:
                - 'level': "HIGH", "MEDIUM", or "LOW"
                - 'recommendation': Clinical recommendation
                - 'score': Computed risk score
                - 'breakdown': Score component details
        """
        tobacco_norm = self._normalize(tobacco_years, self.tobacco_significant_years)
        smoking_norm = self._normalize(smoking_years, self.smoking_significant_years)

        base_score = vision_score * 10
        tobacco_contribution = self.alpha * tobacco_norm
        smoking_contribution = self.beta * smoking_norm

        risk_score = base_score + tobacco_contribution + smoking_contribution

        breakdown = {
            'clip_base': round(base_score, 2),
            'tobacco_norm': round(tobacco_norm, 2),
            'tobacco_contribution': round(tobacco_contribution, 2),
            'smoking_norm': round(smoking_norm, 2),
            'smoking_contribution': round(smoking_contribution, 2),
        }

        if risk_score > self.high_threshold:
            return {
                'level': 'HIGH',
                'recommendation': 'Urgent referral to specialist within 2 weeks',
                'score': risk_score,
                'breakdown': breakdown,
            }
        elif risk_score > self.medium_threshold:
            return {
                'level': 'MEDIUM',
                'recommendation': 'Non-urgent referral within 4 weeks',
                'score': risk_score,
                'breakdown': breakdown,
            }
        else:
            return {
                'level': 'LOW',
                'recommendation': 'Routine monitoring recommended',
                'score': risk_score,
                'breakdown': breakdown,
            }

    def get_risk_factors(self, tobacco_years=0, smoking_years=0, **kwargs):
        """Identify active risk factors for a patient.

        Args:
            tobacco_years: Years of smokeless tobacco use
            smoking_years: Years of smoking
            **kwargs: Ignored for backward compatibility

        Returns:
            list: List of active risk factor descriptions
        """
        risk_factors = []

        tobacco_norm = self._normalize(tobacco_years, self.tobacco_significant_years)
        smoking_norm = self._normalize(smoking_years, self.smoking_significant_years)

        if tobacco_norm > 0:
            level = "high" if tobacco_norm >= 0.7 else "moderate" if tobacco_norm >= 0.3 else "low"
            risk_factors.append(
                f"Tobacco use: {tobacco_years}y ({level} risk, "
                f"norm={tobacco_norm:.2f}, contributes +{self.alpha * tobacco_norm:.1f})"
            )

        if smoking_norm > 0:
            level = "high" if smoking_norm >= 0.7 else "moderate" if smoking_norm >= 0.3 else "low"
            risk_factors.append(
                f"Smoking: {smoking_years}y ({level} risk, "
                f"norm={smoking_norm:.2f}, contributes +{self.beta * smoking_norm:.1f})"
            )

        return risk_factors
