import numpy as np


class PhysicalConstants:
    c = 3e10
    m_e = 9.109e-28
    e = 4.803e-10
    sigma_T = 6.65e-25
    m_e_c2_keV = 511


class Magnetosphere:
    def __init__(self, B=1e14, n_e=1e14, P=1.5, beta_flow=0.3):
        self.B = B
        self.n_e = n_e
        self.P = P
        self.beta_flow = beta_flow
        self.constants = PhysicalConstants()
        
    @property
    def r_LC(self):
        return self.constants.c * self.P / (2 * np.pi)
    
    @property
    def v_flow(self):
        return self.beta_flow * self.constants.c
    
    @property
    def gamma_flow(self):
        return 1 / np.sqrt(1 - self.beta_flow**2)
    
    @property
    def lambda_T(self):
        return 1 / (self.n_e * self.constants.sigma_T)
    
    def doppler_factor(self, direction):
        if direction == "headon":
            return self.gamma_flow * (1 + self.beta_flow)
        return self.gamma_flow * (1 - self.beta_flow)


class Photon:
    def __init__(self, energy_keV):
        self.energy_keV = energy_keV
        self.constants = PhysicalConstants()
        
    @property
    def x(self):
        return self.energy_keV / self.constants.m_e_c2_keV
    
    @property
    def klein_nishina_ratio(self):
        if self.x < 0.1:
            return 1.0
        elif self.x > 10:
            return 3 / (8 * self.x) * (np.log(2 * self.x) + 0.5)
        return 1 / (1 + 2 * self.x)
    
    def effective_energy(self, magnetosphere, direction):
        return self.energy_keV * magnetosphere.doppler_factor(direction)
    
    def effective_cross_section(self, magnetosphere, direction):
        eff_photon = Photon(self.effective_energy(magnetosphere, direction))
        return eff_photon.klein_nishina_ratio


class PhotonTransport:
    def __init__(self, magnetosphere, r_torus_units=7.0):
        self.magnetosphere = magnetosphere
        self.r_torus_units = r_torus_units
        
    @property
    def r_torus(self):
        return self.r_torus_units * self.magnetosphere.r_LC
    
    @property
    def t_advection(self):
        return self.r_torus / self.magnetosphere.v_flow
    
    def mean_free_path(self, photon):
        return self.magnetosphere.lambda_T / photon.klein_nishina_ratio
    
    def t_diffusion(self, photon):
        mfp = self.mean_free_path(photon)
        return self.r_torus**2 / (mfp * self.magnetosphere.constants.c)
    
    def n_scatterings(self, photon):
        return (self.r_torus / self.magnetosphere.lambda_T) * photon.klein_nishina_ratio
    
    def escape_time(self, photon, flow_direction):
        t_diff = self.t_diffusion(photon)
        if photon.klein_nishina_ratio > 0.5:
            if flow_direction == "outward":
                return max(0, t_diff - self.t_advection)
            return t_diff + self.t_advection
        return t_diff
    
    def spectral_lag(self, soft_photon, hard_photon, flow_direction):
        t_soft = self.escape_time(soft_photon, flow_direction)
        t_hard = self.escape_time(hard_photon, flow_direction)
        return t_soft - t_hard


class SpectralLagAnalysis:
    def __init__(self, magnetosphere, transport, soft_energy=100, hard_energy=1000):
        self.magnetosphere = magnetosphere
        self.transport = transport
        self.soft_photon = Photon(soft_energy)
        self.hard_photon = Photon(hard_energy)
        
    def run(self):
        results = {
            "magnetosphere": self._magnetosphere_properties(),
            "doppler": self._doppler_effects(),
            "cross_sections": self._cross_section_analysis(),
            "transport": self._transport_analysis(),
            "lags": self._lag_analysis()
        }
        return results
    
    def _magnetosphere_properties(self):
        return {
            "P": self.magnetosphere.P,
            "r_LC": self.magnetosphere.r_LC,
            "beta_flow": self.magnetosphere.beta_flow,
            "gamma_flow": self.magnetosphere.gamma_flow,
            "lambda_T": self.magnetosphere.lambda_T
        }
    
    def _doppler_effects(self):
        return {
            "D_headon": self.magnetosphere.doppler_factor("headon"),
            "D_following": self.magnetosphere.doppler_factor("following"),
            "ratio": (self.magnetosphere.doppler_factor("headon") / 
                     self.magnetosphere.doppler_factor("following"))
        }
    
    def _cross_section_analysis(self):
        return {
            "soft": {
                "energy": self.soft_photon.energy_keV,
                "x": self.soft_photon.x,
                "sigma_ratio": self.soft_photon.klein_nishina_ratio,
                "sigma_against": self.soft_photon.effective_cross_section(
                    self.magnetosphere, "headon"),
                "sigma_with": self.soft_photon.effective_cross_section(
                    self.magnetosphere, "following")
            },
            "hard": {
                "energy": self.hard_photon.energy_keV,
                "x": self.hard_photon.x,
                "sigma_ratio": self.hard_photon.klein_nishina_ratio,
                "sigma_against": self.hard_photon.effective_cross_section(
                    self.magnetosphere, "headon"),
                "sigma_with": self.hard_photon.effective_cross_section(
                    self.magnetosphere, "following")
            }
        }
    
    def _transport_analysis(self):
        return {
            "r_torus": self.transport.r_torus,
            "t_advection": self.transport.t_advection,
            "t_diff_soft": self.transport.t_diffusion(self.soft_photon),
            "t_diff_hard": self.transport.t_diffusion(self.hard_photon),
            "n_scatt_soft": self.transport.n_scatterings(self.soft_photon),
            "n_scatt_hard": self.transport.n_scatterings(self.hard_photon)
        }
    
    def _lag_analysis(self):
        return {
            "outward": {
                "t_soft": self.transport.escape_time(self.soft_photon, "outward"),
                "t_hard": self.transport.escape_time(self.hard_photon, "outward"),
                "lag": self.transport.spectral_lag(
                    self.soft_photon, self.hard_photon, "outward")
            },
            "inward": {
                "t_soft": self.transport.escape_time(self.soft_photon, "inward"),
                "t_hard": self.transport.escape_time(self.hard_photon, "inward"),
                "lag": self.transport.spectral_lag(
                    self.soft_photon, self.hard_photon, "inward")
            }
        }


class ResultsPrinter:
    def __init__(self, results):
        self.results = results
        
    def print_all(self):
        self._print_header("MAGNETIC FIELD FLOW EFFECT ON PHOTON ESCAPE")
        self._print_magnetosphere()
        self._print_doppler()
        self._print_cross_sections()
        self._print_transport()
        self._print_lags()
        self._print_conclusion()
        
    def _print_header(self, title):
        print("=" * 70)
        print(title)
        print("=" * 70)
        
    def _print_section(self, title):
        print(f"\n{title}")
        print("-" * 70)
        
    def _print_magnetosphere(self):
        self._print_section("1. PLASMA FLOW VELOCITY IN MAGNETOSPHERE")
        m = self.results["magnetosphere"]
        print(f"Rotation period: P = {m['P']} s")
        print(f"Light cylinder radius: r_LC = {m['r_LC']:.2e} cm = {m['r_LC']/1e5:.0f} km")
        print(f"Characteristic plasma flow velocity: v_flow = {m['beta_flow']:.1f}c")
        
    def _print_doppler(self):
        self._print_section("2. DOPPLER EFFECT ON SCATTERING")
        d = self.results["doppler"]
        m = self.results["magnetosphere"]
        print(f"Flow Lorentz factor: gamma = {m['gamma_flow']:.2f}")
        print(f"\nDoppler factor (photon AGAINST flow): D = {d['D_headon']:.2f}")
        print(f"Doppler factor (photon WITH flow): D = {d['D_following']:.2f}")
        print(f"Ratio: {d['ratio']:.2f}x difference in effective energy")
        
    def _print_cross_sections(self):
        self._print_section("3. KLEIN-NISHINA EFFECT")
        cs = self.results["cross_sections"]
        
        print(f"Soft photons ({cs['soft']['energy']} keV):")
        print(f"  E/m_e c^2 = {cs['soft']['x']:.2f}")
        print(f"  sigma/sigma_T = {cs['soft']['sigma_ratio']:.3f}")
        
        print(f"\nHard photons ({cs['hard']['energy']} keV):")
        print(f"  E/m_e c^2 = {cs['hard']['x']:.2f}")
        print(f"  sigma/sigma_T = {cs['hard']['sigma_ratio']:.3f}")
        
        ratio = cs['soft']['sigma_ratio'] / cs['hard']['sigma_ratio']
        print(f"\nCross-section ratio (soft/hard): {ratio:.1f}x")
        
        self._print_section("4. COMBINED EFFECT: FLOW + ENERGY DEPENDENCE")
        d = self.results["doppler"]
        
        print("SOFT PHOTONS (100 keV):")
        E_against = cs['soft']['energy'] * d['D_headon']
        E_with = cs['soft']['energy'] * d['D_following']
        print(f"  Against flow: E_eff = {E_against:.0f} keV, sigma/sigma_T = {cs['soft']['sigma_against']:.3f}")
        print(f"  With flow:    E_eff = {E_with:.0f} keV, sigma/sigma_T = {cs['soft']['sigma_with']:.3f}")
        
        print("\nHARD PHOTONS (1 MeV):")
        E_against = cs['hard']['energy'] * d['D_headon']
        E_with = cs['hard']['energy'] * d['D_following']
        print(f"  Against flow: E_eff = {E_against:.0f} keV, sigma/sigma_T = {cs['hard']['sigma_against']:.3f}")
        print(f"  With flow:    E_eff = {E_with:.0f} keV, sigma/sigma_T = {cs['hard']['sigma_with']:.3f}")
        
    def _print_transport(self):
        self._print_section("5. DIFFERENTIAL DELAY MECHANISM")
        t = self.results["transport"]
        m = self.results["magnetosphere"]
        
        print(f"Thomson mean free path: lambda_T = {m['lambda_T']:.2e} cm = {m['lambda_T']/1e5:.0f} km")
        print(f"Compare to r_LC = {m['r_LC']/1e5:.0f} km")
        print(f"Ratio lambda_T/r_LC = {m['lambda_T']/m['r_LC']:.2f}")
        
        print(f"\nScatterings to escape from r = 7 r_LC:")
        print(f"  Soft photons: N ~ {t['n_scatt_soft']:.1f}")
        print(f"  Hard photons: N ~ {t['n_scatt_hard']:.1f}")
        
        self._print_section("6. THE KEY MECHANISM: ADVECTION VS DIFFUSION")
        print(f"Soft photon diffusion time: t_diff = {t['t_diff_soft']:.2f} s")
        print(f"Advection time: t_adv = {t['t_advection']:.2f} s")
        print(f"Hard photon diffusion time: t_diff = {t['t_diff_hard']:.2f} s")
        print(f"\nRatio t_diff_soft / t_adv = {t['t_diff_soft']/t['t_advection']:.2f}")
        
    def _print_lags(self):
        self._print_section("7. LAG MAGNITUDE ESTIMATE")
        l = self.results["lags"]
        
        print(f"Hard photon escape time: {l['outward']['t_hard']:.2f} s")
        
        print(f"\nOutward flow (North pole):")
        print(f"  Soft escape time: {l['outward']['t_soft']:.2f} s")
        print(f"  Lag = t_soft - t_hard = {l['outward']['lag']:.2f} s (NEGATIVE)")
        
        print(f"\nInward flow (South pole):")
        print(f"  Soft escape time: {l['inward']['t_soft']:.2f} s")
        print(f"  Lag = t_soft - t_hard = {l['inward']['lag']:.2f} s (POSITIVE)")
        
        print(f"\n|Lag| magnitude: {abs(l['inward']['lag']):.1f} s")
        
    def _print_conclusion(self):
        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print("""
NORTH POLE (field outward, mu dot L > 0):
  - Soft photons advected outward -> arrive FIRST
  - Hard photons diffuse independently -> arrive SECOND  
  - NEGATIVE LAG

SOUTH POLE (field inward, mu dot L < 0):
  - Soft photons advected inward -> delayed
  - Hard photons diffuse independently -> arrive FIRST
  - POSITIVE LAG
""")


class RefinedAnalysis:
    def __init__(self, P=1.47, r_torus_units=9.7, N_wind=1.69):
        self.P = P
        self.r_torus_units = r_torus_units
        self.N_wind = N_wind
        self.constants = PhysicalConstants()
        
    @property
    def r_LC(self):
        return self.constants.c * self.P / (2 * np.pi)
    
    @property
    def r_torus(self):
        return self.r_torus_units * self.r_LC
    
    @property
    def t_wind(self):
        return self.N_wind * 2 * np.pi * self.r_torus / self.constants.c
    
    @property
    def v_needed(self):
        return self.r_torus / self.t_wind
    
    @property
    def beta_needed(self):
        return self.v_needed / self.constants.c
    
    def print_results(self):
        print("\n" + "=" * 70)
        print("8. REFINED CALCULATION WITH POPULATION PARAMETERS")
        print("=" * 70)
        
        print(f"Median parameters from Fermi inversion:")
        print(f"  P = {self.P} s")
        print(f"  r_torus = {self.r_torus_units} r_LC = {self.r_torus:.2e} cm")
        print(f"  N_wind = {self.N_wind}")
        print(f"\nWinding delay (from Eq. 12): {self.t_wind:.1f} s")
        print(f"\nFlow velocity to match: v = {self.beta_needed:.2f}c")


if __name__ == "__main__":
    magnetosphere = Magnetosphere()
    transport = PhotonTransport(magnetosphere)
    analysis = SpectralLagAnalysis(magnetosphere, transport)
    
    results = analysis.run()
    printer = ResultsPrinter(results)
    printer.print_all()
    
    refined = RefinedAnalysis()
    refined.print_results()
