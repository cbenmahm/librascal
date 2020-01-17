/**
 * @file   test_behler_feature.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   18 Dec 2019
 *
 * @brief  testing Behler-Parinello G-functions
 *
 * Copyright © 2019 Till Junge
 *
 * rascal is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * rascal is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with rascal; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 */

// #include "rascal/representations/behler_feature.hh"
#include "behler_fixtures.hh"
#include "test_structure.hh"

#include "rascal/representations/behler_feature.hh"
#include "rascal/utils/json_io.hh"
#include "rascal/utils/permutation.hh"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#include <memory>

namespace rascal {

  template <SymmetryFunctionType MySymFunType,
            SymmetryFunctionType... SymFunTypes>
  struct BehlerFeatureFixture {
    BehlerFeatureFixture() {}
    using CutFun_t = CutoffFunction<InlCutoffFunctionType::Cosine>;
    const double r_cut{1.1};
    const UnitStyle unit_style{units::metal};
    std::shared_ptr<CutFun_t> cut_fun{std::make_shared<CutFun_t>(
        unit_style,
        json{{"params", {}}, {"r_cut", {{"value", r_cut}, {"unit", "Å"}}}})};
    json raw_params{{"type", "Gaussian"},
                    {"index", 0},
                    {"unit", "eV"},
                    {"params",
                     {{"eta", {{"value", 0.1}, {"unit", "(Å)^(-2)"}}},
                      {"r_s", {{"value", 0.6}, {"unit", "Å"}}}}},
                    {"species", {"Mg", "Si"}},
                    {"r_cut", {{"value", r_cut}, {"unit", "Å"}}}};
    BehlerFeature<MySymFunType, SymFunTypes...> bf{cut_fun, unit_style,
                                                   raw_params};
  };

  // list of all tested BehlerFeatures
  using Features =
      boost::mpl::list<BehlerFeatureFixture<SymmetryFunctionType::Gaussian,
                                            SymmetryFunctionType::Gaussian>>;

  // list of all tested BehlerFeatures defined on pairs
  using PairFeatures =
      boost::mpl::list<BehlerFeatureFixture<SymmetryFunctionType::Gaussian,
                                            SymmetryFunctionType::Gaussian>>;

  BOOST_AUTO_TEST_SUITE(behler_parinello_feature_tests);
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix, Features, Fix) {}

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(eval_test, Fix, Features, Fix) {
    ManagerFixture<StructureManagerLammps> manager_fix{};
    auto manager_ptr{
        make_adapted_manager<AdaptorStrict>(manager_fix.manager, Fix::r_cut)};
    auto & manager{*manager_ptr};
    manager.update();
    using GVals_t =
        Property<double, AtomOrder, AdaptorStrict<StructureManagerLammps>>;
    auto G_vals{std::make_shared<GVals_t>(manager)};

    // Yes, the pairs in this manager do not have the correct species, but this
    // doesn't interfere with testing the compute algo
    Fix::bf.template compute<RepeatedSpecies::All, Permutation<2, 0, 1>>(
        manager, G_vals);
    Fix::bf.template compute<RepeatedSpecies::Not, Permutation<2, 0, 1>>(
        manager, G_vals);

    auto throw_unknown_species_rep{[&manager, &G_vals, this]() {
      this->bf
          .template compute<RepeatedSpecies::FirstTwo, Permutation<2, 0, 1>>(
              manager, G_vals);
    }};

    BOOST_CHECK_THROW(throw_unknown_species_rep(), std::runtime_error);
  }

  /**
   * Tests evaluation of a pair behler feature including permutation, and
   * compares results to independently computed reference values on a half
   * neighbour list
   */
  using GaussianSymFun = BehlerFeatureFixture<SymmetryFunctionType::Gaussian,
                                              SymmetryFunctionType::Gaussian>;
  BOOST_FIXTURE_TEST_CASE(pair_permutation_test, GaussianSymFun) {
    ManagerFixture<StructureManagerLammps> manager_fix{};
    auto half_list_ptr{
        make_adapted_manager<AdaptorHalfList>(manager_fix.manager)};
    auto manager_ptr{
        make_adapted_manager<AdaptorStrict>(half_list_ptr, this->r_cut)};
    auto & manager{*manager_ptr};
    manager.update();
    using GVals_t =
        Property<double, AtomOrder,
                 AdaptorStrict<AdaptorHalfList<StructureManagerLammps>>>;
    using dGVals_t =
        Property<double, AtomOrder,
                 AdaptorStrict<AdaptorHalfList<StructureManagerLammps>>, 3>;
    // results without permutation
    auto G01_vals{std::make_shared<GVals_t>(manager)};
    // results with permutation
    auto G10_vals{std::make_shared<GVals_t>(manager)};
    // results with equal species
    auto G11_vals{std::make_shared<GVals_t>(manager)};

    // results with derivative without permutation
    auto G01_vals2{std::make_shared<GVals_t>(manager)};
    // results with permutation
    auto G10_vals2{std::make_shared<GVals_t>(manager)};
    // results with equal species
    auto G11_vals2{std::make_shared<GVals_t>(manager)};

    // results with derivative without permutation
    auto dG01_derivatives{std::make_shared<dGVals_t>(manager)};
    // results with permutation
    auto dG10_derivatives{std::make_shared<dGVals_t>(manager)};
    // results with equal species
    auto dG11_derivatives{std::make_shared<dGVals_t>(manager)};

    // manual without permutation
    auto G01_ref{std::make_shared<GVals_t>(manager)};
    // manual with permutation
    auto G10_ref{std::make_shared<GVals_t>(manager)};
    // manual with equal species
    auto G11_ref{std::make_shared<GVals_t>(manager)};

    this->bf.template compute<RepeatedSpecies::Not, Permutation<2, 0, 1>>(
        manager, G01_vals);
    this->bf.template compute<RepeatedSpecies::Not, Permutation<2, 1, 0>>(
        manager, G10_vals);
    this->bf.template compute<RepeatedSpecies::All, Permutation<2, 1, 0>>(
        manager, G11_vals);

    this->bf.template compute<RepeatedSpecies::Not, Permutation<2, 0, 1>>(
        manager, G01_vals2, dG01_derivatives);
    this->bf.template compute<RepeatedSpecies::Not, Permutation<2, 1, 0>>(
        manager, G10_vals2, dG10_derivatives);
    this->bf.template compute<RepeatedSpecies::All, Permutation<2, 1, 0>>(
        manager, G11_vals2, dG11_derivatives);

    const double eta{this->raw_params.at("params")
                         .at("eta")
                         .at("value")
                         .template get<double>()};
    const double r_s{this->raw_params.at("params")
                         .at("r_s")
                         .at("value")
                         .template get<double>()};

    G01_ref->resize();
    G10_ref->resize();
    G11_ref->resize();
    for (auto && atom : manager) {
      for (auto && pair : atom.pairs()) {
        double r_ij{manager.get_distance(pair)};
        double f_c{.5 * (std::cos(math::PI * r_ij / this->r_cut) + 1)};
        double G_incr{std::exp(-eta * (r_ij - r_s) * (r_ij - r_s)) * f_c};
        G01_ref->operator[](atom) += G_incr;
        G10_ref->operator[](pair) += G_incr;
        G11_ref->operator[](atom) += G_incr;
        G11_ref->operator[](pair) += G_incr;
      }
    }

    double rel_error{(G01_vals->eigen() - G01_ref->eigen()).norm() /
                     G01_ref->eigen().norm()};
    BOOST_CHECK_EQUAL(rel_error, 0);

    rel_error =
        (G10_vals->eigen() - G10_ref->eigen()).norm() / G10_ref->eigen().norm();
    BOOST_CHECK_EQUAL(rel_error, 0);

    rel_error =
        (G11_vals->eigen() - G11_ref->eigen()).norm() / G11_ref->eigen().norm();
    BOOST_CHECK_EQUAL(rel_error, 0);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace rascal
