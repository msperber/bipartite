'''
Created on Mar 31, 2014

@author: Matthias Sperber
'''

from source.topics.infer_topics_updates_metropolis import *
import unittest


class TestUpdateGenerated(unittest.TestCase):
    
    def test_drawProposalTypeProportions(self):
        p1, p2, p3 = drawProposalTypeProportions()
        assert p1 >= 0.0
        assert p2 >= 0.0
        assert p3 >= 0.0
        assert_almost_equal(1.0, p1 + p2 + p3)
    
    def test_drawProposalType(self):
        for _ in range(100):
            assert drawProposalType(0.3, 0.3, 0.4) in [PROPOSE_CREATE, PROPOSE_ADD, PROPOSE_DELETE]

        for _ in range(100):
            assert drawProposalType(1.0, 0.0, 0.0) == PROPOSE_CREATE