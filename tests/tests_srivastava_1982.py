from PySDM.physics import constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis

from PySDM_examples.Srivastava_1982.equations import Equations


class TestEquations:
    def test_eq10(self):
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            frag_mass = 1 * si.kg
            eqs = Equations(
                alpha_star=Equations.eq6_alpha_star(
                    alpha=1 / si.s, c=1 / si.s, M=1 * si.kg / frag_mass
                )
            )

            # act
            m_e = eqs.eq10(
                M=1 * si.kg / frag_mass,
                dt=1 * si.s,
                c=1 / si.s,
                total_volume=1 * si.m**3,
                total_number_0=1,
                rho=1 * si.kg / si.m**3,
                x=1,
                frag_mass=frag_mass,
            )

            # assert
            assert m_e.check("[]")

    def test_eq12(self):
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            frag_mass = 1 * si.kg
            eqs = Equations(
                alpha_star=Equations.eq6_alpha_star(
                    alpha=1 / si.s, c=1 / si.s, M=1 * si.kg / frag_mass
                )
            )

            # act
            m_e = eqs.eq12()

            # assert
            assert m_e.check("[]")

    def test_eq13(self):
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            frag_mass = 1 * si.kg
            eqs = Equations(
                beta_star=Equations.eq6_beta_star(beta=1 / si.s, c=1 / si.s)
            )

            # act
            m_e = eqs.eq13(
                M=1 * si.kg / frag_mass,
                dt=1 * si.s,
                c=1 / si.s,
                total_volume=1 * si.m**3,
                total_number_0=1,
                rho=1 * si.kg / si.m**3,
                x=1,
                frag_mass=frag_mass,
            )

            # assert
            assert m_e.check("[]")

    def test_eq14(self):
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            eqs = Equations(
                beta_star=Equations.eq6_beta_star(beta=1 / si.s, c=1 / si.s)
            )

            # act
            m_e = eqs.eq14()

            # assert
            assert m_e.check("[]")
