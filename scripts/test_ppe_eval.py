import unittest
from ppe_eval import box_iou, evaluate_image

def g(box, cls=6):
    return dict(cls=cls, box=box)

def p(box, score=0.9, cls=6):
    return dict(cls=cls, box=box, conf=score)

class EvaluationTest(unittest.TestCase):
    def test_duplicate_stays_fp(self):
        gs, ps = evaluate_image([g([0,0,10,10])], [p([0,0,10,10]), p([0,0,10,10],0.8)])
        self.assertEqual([r['status'] for r in ps], ['TP','FP'])
        self.assertEqual(ps[1]['kind'], 'FP_duplicate_candidate')

    def test_wrong_class_is_fp_and_fn(self):
        gs, ps = evaluate_image([g([0,0,10,10])], [p([0,0,10,10],cls=0)])
        self.assertEqual((gs[0]['status'],ps[0]['status']), ('FN','FP'))

    def test_no_rounding_at_boundary(self):
        b = [0,0,4.999,10]
        self.assertLess(box_iou(b,[0,0,10,10]),0.5)
        gs, ps = evaluate_image([g([0,0,10,10])],[p(b)])
        self.assertEqual(ps[0]['status'],'FP')

    def test_second_available_gt(self):
        gs, ps = evaluate_image([g([0,0,10,10]),g([1,0,11,10])],
                               [p([0,0,10,10]),p([0,0,10,10],0.8)])
        self.assertEqual(sum(r['status']=='TP' for r in ps),2)

    def test_empty(self):
        self.assertEqual(evaluate_image([],[]),([],[]))
        self.assertEqual(evaluate_image([], [p([0,0,1,1])])[1][0]['status'],'FP')

    def test_below_confidence(self):
        gs, ps = evaluate_image([g([0,0,10,10])],[p([0,0,10,10],0.24)])
        self.assertEqual((gs[0]['status'],len(ps)),('FN',0))

    def test_no_helmet_prediction_on_helmet_gt(self):
        gs, ps = evaluate_image([g([0,0,10,10],cls=0)],[p([0,0,10,10])])
        self.assertEqual(ps[0]['kind'],'FP_other_class_overlap')
        self.assertEqual(ps[0]['other_cls'],0)

    def test_exact_threshold(self):
        gs, ps = evaluate_image([g([0,0,10,10])],[p([0,0,5,10],0.25)])
        self.assertEqual(ps[0]['status'],'TP')

if __name__ == '__main__':
    unittest.main()
