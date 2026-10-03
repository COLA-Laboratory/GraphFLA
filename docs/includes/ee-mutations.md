## Evolvability-enhancing Mutations

Background dependence can also change the opportunities for subsequent adaptation. A mutation may make other mutations more favorable, even when its own fitness effect is small or negative. [Wagner (2023)](https://doi.org/10.1038/s41467-023-39321-8) describes such mutations as evolvability-enhancing (EE).

GraphFLA compares the mean fitness of the two mutational neighborhoods, excluding changes at the mutated position. For a beneficial mutation, the neighborhood increase must exceed its own fitness gain; for a neutral or deleterious mutation, it must exceed zero. The EE fraction counts statistically supported cases among all observed directed mutations. It measures local opportunities for adaptation, without guaranteeing that evolution will reach a fitter peak.
