"""
project_2group_xs_BOL_generator_FINAL.py

Final beginning-of-life (BOL) 2-group cross-section generator for the ME388D
microreactor project.

This script uses the corrected class Sn-code package unchanged and defines the
project materials/geometry in one driver file.

Required files in the same directory:
    mpact_reader.py
    mpact_material.py
    mpact_geometry.py
    sn_transport.py
    xs_postprocessor.py
    mpact_data_8g.fmt

Output:
    project_2group_xs_BOL.json

Material basis:
    Fuel:      TRIGA progression problem Table 3 U-ZrH1.6 composition
    Reflector: TRIGA progression problem Table 6 graphite composition
    Shield:    TRIGA progression problem Table 5 stainless-steel composition

Group convention:
    Group 1 = fast    = original MPACT groups 1-4
    Group 2 = thermal = original MPACT groups 5-8
"""

import json
from pathlib import Path
import numpy as np

from mpact_reader import MPACTLibrary
from mpact_material import Material
from mpact_geometry import OneDimensionalCartesianGeometryAndMesh
from sn_transport import LDFESNTransportSolver
from xs_postprocessor import CrossSectionPostProcessor

AVOGADRO = 6.02214076e23
BARN_CM2 = 1.0e-24


def require_nuclide(lib, names):
    """Return the first nuclide found by name; raise a clear error if none exist."""
    if isinstance(names, str):
        names = [names]

    for name in names:
        nuc = lib.find_nuclide_by_name(name)
        if nuc is not None:
            return nuc

    available = ", ".join([name for _, name in lib.list_nuclides()[:60]])
    raise ValueError(
        f"Could not find any of these nuclides in the MPACT library: {names}\n"
        f"First available nuclides include: {available}"
    )


def add_by_weight_fraction(material, nuclide, density_g_cm3, weight_fraction):
    """
    Add a nuclide using weight fraction and bulk material density.

    Number density units are atoms/(barn-cm):
        N = rho * w / A * N_A * 1e-24
    """
    number_density = density_g_cm3 * weight_fraction / nuclide.atomic_mass * AVOGADRO * BARN_CM2
    material.add_nuclide(nuclide, number_density, weight_fraction=weight_fraction)


def add_by_atom_density(material, lib, nuc_name, atom_density):
    """Add a nuclide using an atom density already given in atoms/(barn-cm)."""
    nuc = require_nuclide(lib, nuc_name)
    material.add_nuclide(nuc, atom_density)


def create_uzrh_fuel_EOL(lib):
    """
    END-of-life fuel material based on TRIGA progression-problem
    Table 3 U-ZrH1.6 specifications.

    Source values used here:
        Density: 5.85 g/cm^3
        Composition: weight fractions from Table 3 U-ZrH1.6 specifications
    """
    rho = 5.85
    fuel = Material("U-ZrH1.6 Fuel BOL - TRIGA Table 3", temperature=900.0, density=rho)

    # Table 3 U-ZrH1.6 composition, interpreted as weight fractions.
    table3_weight_fractions = [
        ("H-1", 0.014355),
        ("MN-55", 0.0014287),
        ("U-235", 0.01064),
        ("U-238", 0.066128),
        ("ZR-90", 0.43706),
        ("ZR-91", 0.0942),
        ("ZR-92", 0.14253),
        ("ZR-94", 0.14136),
        ("ZR-96", 0.02228),
        ("CR-NAT", 0.013573),
        ("FE-NAT", 0.049647),
        ("NI-NAT", 0.0067863),
    ]

    for nuc_name, wt_frac in table3_weight_fractions:
        nuc = require_nuclide(lib, nuc_name)
        add_by_weight_fraction(fuel, nuc, rho, wt_frac)

    return fuel


def create_graphite_reflector(lib):
    """Graphite reflector based on TRIGA Table 6: density 1.6 g/cm^3, C-Nat."""
    c = require_nuclide(lib, "C-NAT")
    rho = 1.60
    graphite = Material("Graphite Reflector - TRIGA Table 6", temperature=600.0, density=rho)
    add_by_weight_fraction(graphite, c, rho, 1.0)
    return graphite


def create_stainless_steel_shield(lib):
    """
    Stainless-steel shield based on TRIGA progression-problem Table 5.

    Table 5 gives total density in atoms/(barn-cm) and isotope entries that sum
    to that total. The values below are therefore added directly as number
    densities in atoms/(barn-cm), not converted from weight fractions.
    """
    shield = Material("Stainless-Steel Shield - TRIGA Table 5", temperature=600.0, density=0.0858)

    # Table 5 stainless-steel composition, number densities in atoms/(barn-cm).
    table5_atom_densities = [
        ("C-NAT", 0.00031519),
        ("CR-50", 0.000782),
        ("CR-52", 0.014501),
        ("CR-53", 0.001613),
        ("CR-54", 0.000394),
        ("FE-54", 0.003554),
        ("FE-56", 0.05511),
        ("FE-57", 0.001257),
        ("FE-58", 0.000166),
        ("NI-58", 0.005558),
        ("NI-60", 0.00207),
        ("NI-61", 0.0000885),
        ("NI-62", 0.000278),
        ("NI-64", 0.0000685),
    ]

    for nuc_name, atom_density in table5_atom_densities:
        add_by_atom_density(shield, lib, nuc_name, atom_density)

    return shield


def create_geometry(lib):
    """
    Half-core geometry for cross-section generation.

    Half-core dimensions:
        fuel      = 120 cm
        reflector =  60 cm
        shield    = 100 cm
    """
    fuel = create_uzrh_fuel_EOL(lib)
    reflector = create_graphite_reflector(lib)
    shield = create_stainless_steel_shield(lib)

    geom = OneDimensionalCartesianGeometryAndMesh("BOL Microreactor Half-Core - TRIGA Materials")
    geom.add_region(fuel, length=120.0, n_cells=60, temperature=900.0)
    geom.add_region(reflector, length=60.0, n_cells=30, temperature=600.0)
    geom.add_region(shield, length=100.0, n_cells=50, temperature=600.0)
    geom.finalize()

    return geom


def xs_to_project_dict(xs):
    """Convert a CollapsedXS object to the compact dictionary used by reactor code."""
    sig_tr = np.array(xs.sigma_tr, dtype=float)
    return {
        "D": (1.0 / (3.0 * sig_tr)).tolist(),
        "sigTr": xs.sigma_tr.tolist(),
        "sigA": xs.sigma_a.tolist(),
        "sigT": xs.sigma_t.tolist(),
        "sigS": xs.sigma_s.tolist(),
        "sigF": xs.sigma_f.tolist(),
        "nusigF": xs.nu_sigma_f.tolist(),
        "kapsigF": xs.kappa_sigma_f.tolist(),
        "chi": xs.chi.tolist(),
        "scatter": xs.scatter_matrix.tolist(),
        "sigS12_fast_to_thermal": float(xs.scatter_matrix[0, 1]),
        "sigS21_thermal_to_fast": float(xs.scatter_matrix[1, 0]),
        "flux_for_weighting": xs.flux.tolist(),
        "volume_cm": float(xs.volume),
    }


def print_block(label, xs):
    d = xs_to_project_dict(xs)
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    print("Group 1 = fast, Group 2 = thermal")
    print(f"D       = {d['D']}")
    print(f"sigTr   = {d['sigTr']}")
    print(f"sigA    = {d['sigA']}")
    print(f"nusigF  = {d['nusigF']}")
    print(f"kapsigF = {d['kapsigF']}")
    print(f"chi     = {d['chi']}")
    print("scatter[from_group][to_group] =")
    print(np.array(d["scatter"]))
    print(f"sigS12 fast->thermal = {d['sigS12_fast_to_thermal']}")
    print(f"sigS21 thermal->fast = {d['sigS21_thermal_to_fast']}")


def find_mpact_file():
    """Use mpact_data_8g.fmt if present; otherwise use the uploaded '(2)' filename."""
    candidates = [
        Path("mpact_data_8g.fmt"),
        Path("mpact_data_8g(2).fmt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find mpact_data_8g.fmt. Put this script in the same folder as "
        "the MPACT data file, or rename mpact_data_8g(2).fmt to mpact_data_8g.fmt."
    )


def main():
    mpact_path = find_mpact_file()
    print(f"Using MPACT data file: {mpact_path}")

    lib = MPACTLibrary(str(mpact_path))
    geom = create_geometry(lib)

    solver = LDFESNTransportSolver(
        geom,
        quadrature_order=4,
        left_bc="reflecting",
        right_bc="vacuum",
    )
    solver.tolerance = 1.0e-5
    solver.max_iterations = 5000
    solver.solve()

    pp = CrossSectionPostProcessor(solver)
    results = pp.process_all()

    xs_fuel = results["2g_region_0"]
    xs_reflector = results["2g_region_1"]
    xs_shield = results["2g_region_2"]
    xs_total = results["2g_total"]

    print_block("BOL 2-GROUP XS: FUEL", xs_fuel)
    print_block("BOL 2-GROUP XS: REFLECTOR", xs_reflector)
    print_block("BOL 2-GROUP XS: SHIELD", xs_shield)
    print_block("BOL 2-GROUP XS: HOMOGENIZED TOTAL", xs_total)

    out = {
        "case": "EOL",
        "description": "END-of-life 2-group cross sections generated with corrected Sn postprocessor using TRIGA progression-problem materials.",
        "material_sources": {
            "fuel": "TRIGA progression problem Table 3 U-ZrH1.6: density 5.85 g/cc and listed weight fractions",
            "reflector": "TRIGA progression problem Table 6 graphite: density 1.6 g/cc, C-Nat",
            "shield": "TRIGA progression problem Table 5 stainless steel: listed atom densities in atoms/(barn-cm)",
        },
        "group_convention": {
            "group_1": "fast; original MPACT groups 1-4",
            "group_2": "thermal; original MPACT groups 5-8",
        },
        "geometry_cm_half_core": {
            "fuel": 120.0,
            "reflector": 60.0,
            "shield": 100.0,
        },
        "fuel": xs_to_project_dict(xs_fuel),
        "reflector": xs_to_project_dict(xs_reflector),
        "shield": xs_to_project_dict(xs_shield),
        "homogenized_total": xs_to_project_dict(xs_total),
    }

    out_name = "project_2group_xs_EOL.json"
    with open(out_name, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {out_name}")


if __name__ == "__main__":
    main()
