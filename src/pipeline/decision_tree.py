"""
Clinical Decision Tree

Risk stratification based on vision scores and patient metadata.
"""


class ClinicalDecisionTree:
    """
    Clinical risk assessment using decision tree logic.

    Combines AI vision scores with patient risk factors (age, tobacco use,
    lesion duration) to provide clinical risk stratification.
    """

    def assess_risk(self, vision_score, age, tobacco_years, lesion_duration,
                    *, detection_available=True, concerning_class=None,
                    assessment_issues=()):
        """
        Return a research-demo concern flag, not a clinical diagnosis.

        Risk Calculation:
            - Base score: OCA/OPMD evidence × 10
            - Age > 40: +1.5
            - Tobacco use > 10 years: +2.0
            - Lesion duration > 6 weeks: +1.5

        Missing regions or unresolved model outputs yield INCONCLUSIVE and no
        numeric score. Agreed OCA receives HIGH; agreed OPMD receives at least
        MEDIUM, so a concerning model label cannot be diluted into LOW by fusion.
        Scores and cutoffs are uncalibrated software heuristics.

        Args:
            vision_score: Detector-weighted OCA/OPMD relative score (0-1)
            age: Patient age in years
            tobacco_years: Years of tobacco use
            lesion_duration: Lesion duration category
                ("< 2 weeks", "2-6 weeks", "> 6 weeks")

        Returns:
            dict: Risk assessment containing:
                - 'level': HIGH, MEDIUM, LOW or INCONCLUSIVE
                - 'recommendation': Clinical recommendation (str)
                - 'score': Bounded demo score, or None when not assessed
        """
        if not detection_available or assessment_issues:
            return {
                'level': 'INCONCLUSIVE',
                'recommendation': (
                    'The image could not be assessed reliably. This does not rule '
                    'out disease. A persistent or concerning lesion needs an '
                    'in-person dental or oral-medicine assessment.'
                ),
                'score': None,
            }

        # Base risk score from vision
        risk_score = vision_score * 10

        # Add clinical risk factors
        if age > 40:
            risk_score += 1.5

        if tobacco_years > 10:
            risk_score += 2.0

        if lesion_duration == "> 6 weeks":
            risk_score += 1.5

        risk_score = min(10.0, max(0.0, risk_score))

        # Classify risk level and provide recommendation
        if concerning_class == 'OCA' or (concerning_class == 'OPMD' and risk_score > 8.0):
            return {
                'level': 'HIGH',
                'recommendation': (
                    'The models flagged concerning features. An in-person '
                    'dental or oral-medicine assessment is needed; this research '
                    'result is not a diagnosis.'
                ),
                'score': risk_score
            }
        elif concerning_class == 'OPMD':
            return {
                'level': 'MEDIUM',
                'recommendation': (
                    'The models flagged a potentially malignant appearance. '
                    'Arrange an in-person dental or oral-medicine assessment; '
                    'this research result is not a diagnosis.'
                ),
                'score': risk_score
            }
        else:
            return {
                'level': 'LOW',
                'recommendation': (
                    'The models agreed on non-concerning labels in the detected '
                    'regions. This is not a negative screening result. A '
                    'persistent or concerning lesion still needs an in-person '
                    'dental or oral-medicine assessment.'
                ),
                'score': risk_score
            }

    def get_risk_factors(self, age, tobacco_years, lesion_duration):
        """
        Identify active risk factors for a patient.

        Args:
            age: Patient age
            tobacco_years: Years of tobacco use
            lesion_duration: Lesion duration category

        Returns:
            list: List of active risk factor descriptions
        """
        risk_factors = []

        if age > 40:
            risk_factors.append(f"Age over 40 ({age} years)")

        if tobacco_years > 10:
            risk_factors.append(f"Heavy tobacco use ({tobacco_years} years)")

        if lesion_duration == "> 6 weeks":
            risk_factors.append("Persistent lesion (> 6 weeks)")

        return risk_factors
