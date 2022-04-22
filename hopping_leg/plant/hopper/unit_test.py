import unittest
from hopper_plant import HopperPlant


class KinTest(unittest.TestCase):
    def testPos(self):
        p = HopperPlant()
        self.assertAlmostEqual(p.forward_kinematics(*p.inverse_kinematics(.1, .2))[0], .1)
        self.assertAlmostEqual(p.forward_kinematics(*p.inverse_kinematics(.1, .2))[1], .2)

        self.assertAlmostEqual(p.forward_kinematics(*p.inverse_kinematics(.1, .2, 1))[0], .1)
        self.assertAlmostEqual(p.forward_kinematics(*p.inverse_kinematics(.1, .2, 1))[1], .2)

        self.assertAlmostEqual(p.forward_kinematics(*p.inverse_kinematics(-.1, .2))[0], -.1)
        self.assertAlmostEqual(p.forward_kinematics(*p.inverse_kinematics(-.1, .2))[1], .2)

        self.assertAlmostEqual(p.forward_kinematics(*p.inverse_kinematics(-.1, -.2, 1))[0], -.1)
        self.assertAlmostEqual(p.forward_kinematics(*p.inverse_kinematics(-.1, -.2, 1))[1], -.2)

        p = HopperPlant(link_length=[4,5.2])
        self.assertAlmostEqual(
            p.forward_kinematics(*p.inverse_kinematics(4, 2))[0], 4)
        self.assertAlmostEqual(
            p.forward_kinematics(*p.inverse_kinematics(4, 2))[1], 2)
        
        self.assertAlmostEqual(
            p.forward_kinematics(*p.inverse_kinematics(4, -2))[0], 4)
        self.assertAlmostEqual(
            p.forward_kinematics(*p.inverse_kinematics(4, -2))[1], -2)

        self.assertAlmostEqual(
            p.forward_kinematics(*p.inverse_kinematics(-4, 2))[0], -4)
        self.assertAlmostEqual(
            p.forward_kinematics(*p.inverse_kinematics(-4, 2))[1], 2)

    def testVel(self):
        p = HopperPlant()
        self.assertAlmostEqual(p.forward_velocity(
            1, 2, *p.inverse_velocity(1, 2, 3, 4))[0], 3)
        self.assertAlmostEqual(p.forward_velocity(
            4, 2, *p.inverse_velocity(4, 2, 3, 4))[1], 4)

        self.assertAlmostEqual(p.forward_velocity(
            1, -3, *p.inverse_velocity(1, -3, 3, -4))[0], 3)
        self.assertAlmostEqual(p.forward_velocity(
            1, -1, *p.inverse_velocity(1, -1, -3, 4))[1], 4)

        self.assertAlmostEqual(p.forward_velocity(
            1, 2, *p.inverse_velocity(1, 2, -3, 4))[0], -3)
        self.assertAlmostEqual(p.forward_velocity(
            1, 2, *p.inverse_velocity(1, 2, 3, -4))[1], -4)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is.ignored"])
