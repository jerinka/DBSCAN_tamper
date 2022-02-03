from dbscan import DbscanTamper, get_files
import cv2

import unittest

sbscan_obj = DbscanTamper(showflag = False)
org_path = 'data/org'
tamp_path = 'data/tamp'

org_files = get_files(org_path)
tamp_files = get_files(tamp_path)

print('org_files',org_files)
print('tamp_files',tamp_files)


class TestDbscanTamper(unittest.TestCase):

    def test_det_forgery1(self):
        file_  = tamp_files[1]
        tampered = cv2.imread(file_)
        print('file:', file_)
        forg_flag = sbscan_obj.det_forgery(tampered)
        print('Forgery:', forg_flag)

        self.assertEqual(forg_flag, True, "Should be True")

    def test_det_forgery2(self):
        file_  = org_files[1]
        tampered = cv2.imread(file_)
        print('file:', file_)
        forg_flag = sbscan_obj.det_forgery(tampered)
        print('Forgery:', forg_flag)

        self.assertEqual(forg_flag, False, "Should be False")

if __name__ == '__main__':
    unittest.main()