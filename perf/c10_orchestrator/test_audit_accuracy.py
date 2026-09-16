"""Known rigid-transform control and strict archived-fixture evidence checks."""
import unittest

import numpy as np

from audit_accuracy import kabsch_rmsd, replay, validate_atoms


class AccuracyEvidenceTest(unittest.TestCase):
    def test_known_rigid_transform_and_reflection(self):
        points = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=np.float64)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        self.assertLess(kabsch_rmsd(points, points @ rotation + [7, 11, 13]), 1e-12)
        self.assertGreater(kabsch_rmsd(points, points * [-1, 1, 1]), 0.5)

    def test_invalid_identity_and_nonfinite_coordinates(self):
        keys = [('A', '1', 'CA', 'GLY'), ('A', '2', 'CA', 'GLY')]
        xyz = np.zeros((2, 3), dtype=np.float64)
        validate_atoms(keys, xyz, 2)
        for bad_keys, bad_xyz, size in ((keys, xyz, 3), (keys[:1]*2, xyz, 2),
                                        (keys, xyz + np.nan, 2), (keys, xyz[:1], 2),
                                        ([k[:3] for k in keys], xyz, 2),
                                        ([keys[0], ('B','2','CA','GLY')], xyz, 2)):
            with self.subTest(keys=bad_keys, size=size), self.assertRaises(ValueError):
                validate_atoms(bad_keys, bad_xyz, size)

    def test_archived_reference_spread_matches_cifs(self):
        report = replay()
        for size, mean in [('298', 0.80128), ('512', 1.66454)]:
            result = report['sizes'][size]
            self.assertEqual(result['pairs'], 6)
            self.assertAlmostEqual(result['mean_A'], mean, places=5)
            self.assertTrue(result['matches_archived_five_decimal_scores'])


if __name__ == '__main__':
    unittest.main()
