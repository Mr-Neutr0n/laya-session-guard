import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from evaluate_jev import validate, metrics


class JevEvaluationTests(unittest.TestCase):
    def answers(self):
        return {"content":{"choice":"suspicious","probabilities":{"benign":.01,"suspicious":.99}},
                "action":{"choice":"allow","probabilities":{"allow":.93,"block":.04,"review":.02}}}

    def test_real_two_decimal_api_response_is_valid(self):
        validate(self.answers())

    def test_invalid_probabilities_fail(self):
        for value in (float("nan"),-.1,1.5,.5):
            answer=self.answers();answer["action"]["probabilities"]["allow"]=value
            with self.assertRaises(ValueError):validate(answer)

    def test_metrics_normalize_rounding_without_mutation(self):
        answers=self.answers()
        result=metrics([{"gold":{"content":"suspicious","action":"allow"},"answers":answers}])
        self.assertEqual(result["action"]["accuracy"],1)
        expected=(.93/.99-1)**2+(.04/.99)**2+(.02/.99)**2
        self.assertAlmostEqual(result["action"]["brier"],expected)
        self.assertEqual(answers["action"]["probabilities"]["allow"],.93)
