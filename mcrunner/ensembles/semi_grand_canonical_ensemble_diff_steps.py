import random 

from mchammer.configuration_manager import SwapNotPossibleError
from mchammer.ensembles import SemiGrandCanonicalEnsemble


class SemiGrandCanonicalEnsemble_Diffsteps(SemiGrandCanonicalEnsemble):
    """
    Adds possible diffusion steps to the SGC Ensemble from mchammer
    """

    def __init__(self, prob_threshold=0.75, **kwargs):
        super().__init__(**kwargs)
        self._prob_threshold = prob_threshold


    def _do_trial_step(self):
        """ Carries out one Monte Carlo trial step. """
        sublattice_index = self.get_random_sublattice_index(
            probability_distribution=self._flip_sublattice_probabilities)

        if random.random() < self._prob_threshold:
            try:
                r = self.do_canonical_swap(
                    sublattice_index=sublattice_index
                )
            except SwapNotPossibleError:
                r = self.do_sgc_flip(
                    self.chemical_potentials, sublattice_index=sublattice_index
                )
            return r
        else:
            return self.do_sgc_flip(
                self.chemical_potentials, sublattice_index=sublattice_index
            )